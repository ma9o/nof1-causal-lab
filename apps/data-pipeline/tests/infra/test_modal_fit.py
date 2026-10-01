"""Modal is a compute boundary; pinned study values and publication stay local."""

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from nof1_causal_lab.actions import fit as fit_action
from nof1_causal_lab.actions import modal_fit
from nof1_causal_lab.actions.contracts import FitRequest
from nof1_causal_lab.actions.runners import run_action_locally
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec, InferenceReport
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
from nof1_causal_lab.numpyro_json import empirical_atoms, empirical_distribution
from nof1_causal_lab.study.store import ArtifactStore, read_model
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils.config import get_config
from tests.helpers import run_async
from tests.integration.runner_fixtures import (
    panel_frame,
    panel_metadata,
    seed_model,
    seed_panel,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("failure", [None, "remote", "corrupt_array", "missing_array"])
def test_local_fit_transfers_pins_and_retains_outputs_only_after_valid_response(
    tmp_path, monkeypatch, failure
):
    monkeypatch.delenv("DEPLOYMENT_ENV", raising=False)
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    configured = replace(
        get_config(), inference=replace(get_config().inference, compute_backend="modal")
    )
    monkeypatch.setattr("nof1_causal_lab.utils.config.get_config", lambda: configured)
    store = ArtifactStore("MODALTEST")
    model_info = seed_model(store)
    panel_info = seed_panel(store, model_revision=model_info.revision)
    pins: dict[ArtifactId, GitOid] = {"model": model_info.revision, "panel": panel_info.revision}
    model = read_model(store, model_info.revision)
    bindings, _ = parameter_bindings(model)
    layout = JointLawLayout.from_bindings(
        bindings,
        parameters=[p.id for p in model.parameters],
        constructs=model.state_order,
        time_points=(0, 1),
    )
    atoms = np.ones((2, layout.width))
    calls = []

    def numerical_fit(**kwargs):
        # Only the numerical call is stubbed: transfer, validation, array loading,
        # runner persistence and artifact provenance use their real code.
        assert kwargs["data_for_model"].equals(panel_frame())
        assert kwargs["sampler_config"]["num_samples"] == 50
        assert kwargs["sampler_config"]["seed"] == 7
        source = kwargs["model_spec"]
        if source.time_points:
            # The second transfer proves referenced input arrays travel too.
            np.testing.assert_array_equal(
                empirical_atoms(source.distributions[layout.distribution_id]), atoms
            )
        else:
            assert source.model_dump(mode="json") == model.model_dump(mode="json")
        law = empirical_distribution(
            atoms, array_writer=kwargs["array_writer"], array_loader=kwargs["array_loader"]
        )
        conditioned = model.revised(
            parameters=tuple(
                type(p).model_validate(
                    {
                        **p.model_dump(),
                        "distribution": layout.distribution_id,
                        "distribution_transform": "identity",
                        "reference_interval_days": None,
                    }
                )
                for p in model.parameters
            ),
            edges=replace_constructs(
                model.edges,
                tuple(
                    type(c).model_validate(
                        {**c.model_dump(), "distribution": layout.distribution_id}
                    )
                    if c.id in model.state_order
                    else c
                    for c in model.constructs
                ),
            ),
            distributions={layout.distribution_id: law},
            time_points=(0, 1),
        )
        return {
            "_model": conditioned,
            "engine_evidence": {"initialization": "test", "exact": True},
            "time_origin": "2024-01-01T00:00:00Z",
            "inference_metadata": {
                "method": "marginal_particle_gibbs",
                "n_samples": 2,
                "duration_seconds": 1.25,
            },
            "inference_diagnostics": {"mcmc": {"num_chains": 2}},
        }

    def dispatch(payload):
        calls.append(payload)
        if failure == "remote":
            raise RuntimeError("Modal unavailable")
        result = modal_fit.execute_fit_compute(payload)
        if failure == "corrupt_array":
            return replace(result, arrays=dict.fromkeys(result.arrays, b"broken"))
        if failure == "missing_array":
            return replace(result, arrays={})
        return result

    monkeypatch.setattr(fit_action, "fit", numerical_fit)
    monkeypatch.setattr(modal_fit, "_dispatch_fit", dispatch)
    writes = []
    write_array = ArtifactStore.write_array

    def tracked_write(self, values):
        writes.append(values)
        return write_array(self, values)

    monkeypatch.setattr(ArtifactStore, "write_array", tracked_write)
    request = FitRequest(
        model_revision=model_info.revision,
        panel_revision=panel_info.revision,
        settings=FitSettingsSpec(num_samples=50, seed=7),
    )
    if failure is not None:
        message = {
            "remote": "Modal unavailable",
            "corrupt_array": "content identity check",
            "missing_array": "unresolved numerical array",
        }[failure]
        with pytest.raises((ValueError, RuntimeError), match=message):
            run_async(run_action_locally("MODALTEST", request, pins))
        assert not writes
        assert store.list_revisions("model") == [model_info.revision]
        assert len(calls) == 1
        return
    effects = run_async(run_action_locally("MODALTEST", request, pins))
    assert effects.produced[0].derived_from == pins
    assert effects.diagnostics["input_pins"] == pins
    assert effects.diagnostics["engine_evidence"] == {"initialization": "test", "exact": True}
    report = InferenceReport.model_validate(effects.diagnostics["report"])
    assert report.inference_metadata.n_samples == 2
    assert report.inference_metadata.duration_seconds == 1.25
    restored = read_model(store, effects.produced[0].revision)
    np.testing.assert_array_equal(
        empirical_atoms(restored.distributions[layout.distribution_id]), atoms
    )
    assert len(writes) == 2  # the draws and their weights
    assert len(calls) == 1
    transferred = modal_fit.fit_on_modal(
        time_origin=panel_metadata().time_origin,
        model_spec=restored,
        data_for_model=panel_frame(),
        sampler_config=calls[0].sampler_config,
        array_writer=store.write_array,
        array_loader=store.read_array,
        compute_loo_diagnostics=False,
    )
    assert calls[1].arrays
    transferred_model = ModelSpec.model_validate(transferred["_model"])
    assert transferred_model.model_dump(mode="json") == restored.model_dump(mode="json")
