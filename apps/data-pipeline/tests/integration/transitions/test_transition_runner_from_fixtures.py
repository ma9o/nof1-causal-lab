"""Persistence and refit lifecycle with mocked inference, never numerical fitting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from nof1_causal_lab.machine.execution import ExecutionOptions, apply_transition, input_pins
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.inference import inference_is_current
from nof1_causal_lab.machine.runners import execute_transition_locally
from nof1_causal_lab.machine.store import read_model
from nof1_causal_lab.models.ssm.inference.persistence import model_draws
from tests.helpers import run_async
from tests.integration import transition_runner_fixtures as fx
from tests.model_fixtures import parameter_draws

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from nof1_causal_lab.machine.store import ArtifactStore


def test_inference_advances_model_and_uses_the_selected_input(
    integration_workspace: str,
    artifact_store: ArtifactStore,
    monkeypatch,
) -> None:
    import jax.numpy as jnp

    from nof1_causal_lab.flows.transitions.inference import fit as stage5_fit
    from nof1_causal_lab.models.ssm import numerics
    from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

    original = fx.seed_model(artifact_store)
    panel = fx.seed_panel(artifact_store, model_revision=original.revision)
    authored = fx.scientific_model()
    telemetry = {"new_kernel": {"accepted": [True, False], "tuning": {"step": 0.25}}, "note": None}
    parameters = parameter_draws(authored, 4)
    states = tuple(numerics.state_ids(authored))
    paths = jnp.arange(4 * 2 * len(states), dtype=jnp.float32).reshape(4, 2, len(states))
    fitted_inputs = []

    def fake_fit_model(model, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        fitted_inputs.append(model.model_dump(mode="json"))
        return {
            "fitted": True,
            "duration_seconds": 0.01,
            "result": ParticleMCMCPosterior(
                draws=JointPosteriorDraws(
                    parameters=parameters, latent_paths=paths, state_ids=states
                )
            ),
            "runtime": SimpleNamespace(
                times=jnp.array([0.0, 1.0], dtype=jnp.float32),
            ),
            "inference_diagnostics": telemetry,
            "loo_diagnostics": None,
            "posterior_marginals": None,
            "posterior_pairs": None,
        }

    monkeypatch.setattr(stage5_fit, "fit_model", fake_fit_model)
    state = fx.state_from(original, panel)
    spec = transition_spec("posterior")
    revisions = [original.revision]
    for _ in range(2):
        effects = run_async(
            execute_transition_locally(
                integration_workspace,
                "posterior",
                input_pins(state, spec),
                state,
                ExecutionOptions(),
            )
        )
        info = next(info for info in effects.produced if info.artifact_id == "model")
        assert info.revision not in revisions
        revisions.append(info.revision)
        # Both provenance and computation follow the selected model revision.
        assert info.derived_from == {
            "model": state.current["model"].revision,
            "panel": panel.revision,
        }
        assert effects.diagnostics["input_pins"] == info.derived_from
        assert effects.diagnostics["report"]["inference_diagnostics"] == telemetry
        assert effects.diagnostics["report"]["inference_metadata"]["n_samples"] == 4
        assert effects.diagnostics["engine_evidence"]["latent_transition"] == "euler_maruyama"
        conditioned = read_model(artifact_store, info.revision)
        assert type(conditioned) is type(authored)
        assert conditioned.distributions
        assert "inference_diagnostics" not in conditioned.model_dump()
        retained = model_draws(conditioned)
        np.testing.assert_array_equal(retained.latent_paths, paths)
        for name, values in parameters.items():
            np.testing.assert_array_equal(retained.parameters[name], values)
        state = apply_transition(state, effects.produced, effects.retracted)
        assert inference_is_current(state)
    assert fitted_inputs == [
        read_model(artifact_store, revision).model_dump(mode="json") for revision in revisions[:2]
    ]
    assert read_model(artifact_store, original.revision).model_dump(mode="json") == fitted_inputs[0]
    assert artifact_store.list_revisions("model") == revisions
