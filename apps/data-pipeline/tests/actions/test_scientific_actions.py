"""Scientific actions depend on selected inputs, independently of authoring recipes."""

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from nof1_causal_lab.actions.contracts import PrepareDataRequest, SimulateRequest
from nof1_causal_lab.artifacts.checks import NumericCriterionEvidence
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.models.ssm.inference.persistence import condition_model
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from nof1_causal_lab.models.ssm.predictive.parameters import sample_model_laws
from nof1_causal_lab.study.records import ModelEditResult
from nof1_causal_lab.study.state import StudyState
from tests.action_fixtures import applied_record
from tests.git_fixtures import artifact_revision, commit_id, git_oid
from tests.inference_fixtures import particle_posterior
from tests.integration.runner_fixtures import panel_metadata
from tests.model_fixtures import compile_model_fixture, parameter_draws

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid


@pytest.mark.contract
def test_scientific_tool_transport_shares_the_action_endpoint(monkeypatch):
    from uuid import UUID

    from nof1_causal_lab import study_api, tool_server
    from nof1_causal_lab.actions.results import ActionReceipt, RunningPoll

    requests = []

    async def capture(workspace_id, body, clients, *, branch, expected_head):
        assert branch == "alternative"
        assert expected_head == git_oid(123)
        requests.append((workspace_id, body))
        return ActionReceipt(attempt_id=UUID(int=1))

    monkeypatch.setattr(study_api, "execute_scientific_action", capture)
    client = TestClient(tool_server.app)
    contracts = client.get("/api/tools/scientific")
    assert contracts.status_code == 200
    assert {item["name"] for item in contracts.json()} == {
        "edit_model",
        "prepare_data",
        "fit",
        "simulate",
        "poll_action",
    }
    result = client.post(
        "/api/tools/scientific/simulate",
        json={
            "workspace_id": "TEST",
            "branch": "alternative",
            "expected_head": git_oid(123),
            "input": {
                "model_revision": git_oid(1),
                "end": 1,
            },
        },
    )
    assert result.status_code == 200
    assert result.json()["result"] == {"attempt_id": str(UUID(int=1))}
    assert requests[0][1] == SimulateRequest(model_revision=git_oid(1), end=1)
    invalid = client.post(
        "/api/tools/scientific/prepare_data",
        json={"workspace_id": "TEST", "input": {"source": "raw_data"}},
    )
    assert invalid.status_code == 422
    assert len(requests) == 1

    async def poll(workspace_id, attempt_id, clients):
        assert workspace_id == "TEST"
        assert attempt_id == UUID(int=1)
        return RunningPoll()

    monkeypatch.setattr(study_api, "poll_scientific_action", poll)
    response = client.post(
        "/api/tools/scientific/poll_action",
        json={
            "workspace_id": "TEST",
            "input": {"attempt_id": str(UUID(int=1))},
        },
    )
    assert response.json() == {"result": {"kind": "running", "messages": []}}
    assert (
        client.post(
            "/api/tools/scientific/poll_action",
            json={
                "workspace_id": "TEST",
                "input": {"attempt_id": "invalid"},
            },
        ).status_code
        == 422
    )


@pytest.mark.inference(concern="predictive")
def test_current_law_sampling_preserves_joint_parameter_atoms():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_y_model.json"
        ).read_text()
    )
    original = parameter_draws(model, 3)
    # Each atom has distinct coordinated values; independently resampling marginals
    # would produce combinations absent from the joint law.
    original = {
        name: value * jnp.arange(1, 4).reshape((3,) + (1,) * (value.ndim - 1))
        for name, value in original.items()
    }
    conditioned = condition_model(
        model,
        compile_model_fixture(model),
        particle_posterior(
            JointPosteriorDraws(
                original,
                jnp.zeros((3, 2, 2)),
            )
        ),
        times=jnp.array([0.0, 1.0]),
    )
    draws = sample_model_laws(
        compile_model_fixture(conditioned), draws=24, key=jax.random.PRNGKey(4)
    ).parameters
    for draw in range(24):
        assert any(
            all(np.allclose(draws[name][draw], values[atom]) for name, values in original.items())
            for atom in range(3)
        )
    authored = sample_model_laws(
        compile_model_fixture(model), draws=5, key=jax.random.PRNGKey(4)
    ).parameters
    assert all(value.shape[0] == 5 and np.isfinite(value).all() for value in authored.values())


@pytest.mark.inference(concern="simulation")
@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("fitted_laws", "scientific_model_payload"),
    [
        pytest.param(
            False,
            "common/stress_sleep_model.json",
            id="False",
        ),
        pytest.param(
            True,
            "common/stress_sleep_model.json",
            id="True",
        ),
    ],
)
def test_durable_replication_preserves_current_laws_without_comparison(
    tmp_path, monkeypatch, fitted_laws, scientific_model_payload
):
    from nof1_causal_lab.actions.runners import run_action_locally
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import ModelFitResult
    from nof1_causal_lab.study.snapshots import ModelReader
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module
    from tests.helpers import run_async
    from tests.inference_fixtures import inference_log
    from tests.integration.runner_fixtures import panel_frame

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("TEST"), StudyRepository("TEST")
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / scientific_model_payload
        ).read_text()
    )
    definition = store.write_artifact(
        "model",
        produced_by=None,
        derived_from={},
        json_files={"model.json": model.model_dump(mode="json")},
    )
    produced = [definition]
    pins: dict[ArtifactId, GitOid] = {"model": artifact_revision("TEST", "model", 1)}
    produced.append(
        store.write_artifact(
            "panel",
            produced_by="prepare_data",
            derived_from={},
            json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
            parquet_files={"panel.parquet": panel_frame(n_days=4)},
        )
    )
    if fitted_laws:
        from nof1_causal_lab.models.ssm.inference.persistence import condition_model
        from nof1_causal_lab.models.ssm.inference.types import (
            JointPosteriorDraws,
        )
        from tests.model_fixtures import parameter_draws

        model = condition_model(
            model,
            compile_model_fixture(model),
            particle_posterior(
                JointPosteriorDraws(parameter_draws(model, 3), jnp.zeros((3, 5, 2)))
            ),
            times=jnp.arange(-1.0, 4.0),
        )
        fitted = store.write_artifact(
            "model",
            produced_by="fit",
            derived_from={"model": definition.revision, "panel": produced[-1].revision},
            json_files={"model.json": model.model_dump(mode="json")},
        )
        pins["model"] = fitted.revision
        produced.append(fitted)
    state = StudyState().with_artifacts(produced)
    journal.append(
        applied_record(
            ModelFitResult(
                produced=tuple(produced),
                model=GitRef(workspace_id="TEST", revision=definition.revision, path="model.json"),
                panel=GitRef(
                    workspace_id="TEST", revision=produced[1].revision, path="panel.parquet"
                ),
                report=inference_log(model).record.attempt.outcome.result.report,
            )
            if fitted_laws
            else ModelEditResult(produced=tuple(produced)),
            seq=1,
            ts="2026-09-15T12:00:00Z",
        )
    )
    effects = run_async(
        run_action_locally(
            "TEST",
            SimulateRequest(
                model_revision=pins["model"],
                start=-1.0 if fitted_laws else 0.0,
                end=3.0 if fitted_laws else 4.0,
            ),
            pins,
        )
    )
    report = effects.report
    assert store.read_array(report.latent_paths).shape == (report.draws, 5, 2)
    assert store.read_array(report.observations).shape == (report.draws, 5, 2)
    assert {"model": report.model.revision} == pins
    assert set(pins) == {"model"}
    assert "predictive_checks" not in report.model_dump()
    assert "comparison_panel" not in report.model_dump()
    assert report.law is not None
    assert report.law.kind == ("fitted" if fitted_laws else "authored")
    if fitted_laws:
        assert report.law.interpretation == "posterior_predictive"
        assert report.law.fitted_panel_revision == produced[1].revision
    assert not effects.produced
    assert any(finding.subject.check.startswith("C5c") for finding in report.findings)
    assert not any(finding.subject.check.startswith("C5d") for finding in report.findings)
    journal.append(applied_record(effects, seq=2, ts="2026-09-15T12:01:00Z"))
    current = ModelReader("TEST").simulation()
    assert current is not None
    assert current.value == report
    assert current.source.validity == "fresh"
    edited = store.write_artifact(
        "model",
        produced_by=None,
        derived_from={},
        json_files={
            "model.json": model.revised(question="Revised question").model_dump(mode="json")
        },
    )
    journal.append(
        applied_record(
            ModelEditResult(produced=[edited]), seq=3, ts="2026-09-15T12:02:00Z", trace_ids=[]
        )
    )
    revised = ModelReader("TEST").simulation()
    retained = ModelReader("TEST", at=commit_id("TEST", 2)).simulation()
    assert revised is not None
    assert retained is not None
    assert revised.source.validity == "stale"
    assert retained.source.validity == "fresh"

    # A retained generator result can be prepared without running it again.
    from nof1_causal_lab.actions.runners import run_action
    from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef

    preparation = PrepareDataRequest(
        input=SimulationReplicateRef(revision=commit_id("TEST", 2), replicate=1),
    )
    prepared = run_async(run_action("TEST", preparation, state))
    panel_info = next(info for info in prepared.produced if info.artifact_id == "panel")
    panel = store.read_parquet_file("panel", panel_info.revision, "panel.parquet")
    from nof1_causal_lab.models.ssm.runtime import project_observation_data

    wide, _ = project_observation_data(
        panel, model_spec=compile_model_fixture(model), time_origin=report.time_origin
    )
    np.testing.assert_allclose(
        wide.select(["stress_score", "sleep_score"]).to_numpy(),
        store.read_array(report.observations)[1],
        equal_nan=True,
    )


@pytest.mark.inference(concern="simulation")
@pytest.mark.parametrize(
    ("start_time", "complete_test_model_payload"),
    [
        pytest.param(
            None,
            "common/x_y_model.json",
            id="None",
        ),
        pytest.param(
            0.0,
            "common/x_y_model.json",
            id="0.0",
        ),
        pytest.param(
            0.25,
            "common/x_y_model.json",
            id="0.25",
        ),
    ],
)
def test_retained_forecast_and_timed_intervention_preserve_joint_starts(
    tmp_path, monkeypatch, start_time, complete_test_model_payload
):
    from nof1_causal_lab.actions.simulate import simulate
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.artifacts.scenarios import InterventionSpec
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / complete_test_model_payload
        ).read_text()
    )
    from nof1_causal_lab.models.ssm import numerics as numeric

    states = numeric.state_ids(compile_model_fixture(model))
    paths = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    model = condition_model(
        model,
        compile_model_fixture(model),
        particle_posterior(JointPosteriorDraws(parameter_draws(model, 2), paths)),
        times=jnp.array([0.0, 1.0]),
    )
    store = ArtifactStore("TEST")
    start = 1.0 if start_time is None else start_time
    design = SimulationSpec(
        start=start_time,
        end=start + 1,
        interventions=(InterventionSpec(target=states[0], value=2.0, time=start + 0.5),),
    )
    report = simulate(
        model,
        design,
        time_origin=None,
        revision=GitRef(workspace_id="TEST", revision=git_oid(2), path="model.json"),
        write_array=store.write_array,
    )
    action = store.read_array(report.latent_paths)
    assert report.reference_latent_paths is not None
    reference = store.read_array(report.reference_latent_paths)
    np.testing.assert_allclose(action[:, 0], reference[:, 0])
    if start_time in (None, 0.0):
        for initial in action[:, 0]:
            assert any(
                np.allclose(initial, atom)
                for atom in np.asarray(paths[:, 1 if start_time is None else 0])
            )
    np.testing.assert_allclose(action[:, 1, 0], 2.0, atol=1e-5)
    assert not np.allclose(action[:, -1, 0], 2.0)
    assert np.isfinite(action).all()
    emissions = store.read_array(report.observations)
    assert np.isnan(emissions[:, :-1]).all()
    assert np.isfinite(emissions[:, -1]).all()


@pytest.mark.inference(concern="simulation")
def test_causal_action_uses_common_generator_and_requires_matching_engine_evidence(
    tmp_path, monkeypatch
):
    from nof1_causal_lab.actions.scenarios import summarize_causal_simulation
    from nof1_causal_lab.actions.simulate import simulate
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.artifacts.scenarios import InterventionSpec
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module
    from tests.inference_fixtures import inference_log

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_y_model.json"
        ).read_text()
    )
    states = numeric.state_ids(compile_model_fixture(model))
    model = model.revised(default_outcome=states[1])
    model = condition_model(
        model,
        compile_model_fixture(model),
        particle_posterior(JointPosteriorDraws(parameter_draws(model, 2), jnp.zeros((2, 2, 2)))),
        times=jnp.array([0.0, 1.0]),
    )
    design = SimulationSpec(
        end=2,
        interventions=(InterventionSpec(target=states[0], value=1.0, time=1.0),),
    )
    record = inference_log(model)
    store = ArtifactStore("TEST")
    generated = simulate(
        model,
        design,
        time_origin=None,
        revision=GitRef(workspace_id="TEST", revision=git_oid(2), path="model.json"),
        write_array=store.write_array,
    )
    result = summarize_causal_simulation(model, generated, store=store, inference=record)
    assert result.causal_result is not None
    assert result.reference_latent_paths is not None
    assert result.model.revision == git_oid(2)
    from nof1_causal_lab.actions.prepare_data import prepare_simulation_panel
    from nof1_causal_lab.models.ssm.runtime import project_observation_data

    store = ArtifactStore("TEST")
    panel = prepare_simulation_panel(result, 0, read_array=store.read_array)
    wide, _ = project_observation_data(
        panel, model_spec=compile_model_fixture(model), time_origin=panel_metadata().time_origin
    )
    np.testing.assert_allclose(
        wide.select(numeric.observation_names(compile_model_fixture(model))).to_numpy(),
        store.read_array(result.observations)[0],
        equal_nan=True,
    )
    from nof1_causal_lab.artifacts.checks import NotEvaluated

    report = record.record.attempt.outcome.result.report
    bad = inference_log(
        model,
        report=report.revised(
            **{
                "engine": NotEvaluated(
                    subject="production_engine",
                    reason="ARCHIVED_ENGINE_NOT_RETAINED",
                    detail="Engine evidence not retained",
                )
            }
        ),
    )
    rejected = summarize_causal_simulation(model, generated, store=store, inference=bad)
    assert rejected.causal_result is None
    assert rejected.causal_unavailable_reason is not None
    assert "retained exact-engine evidence" in rejected.causal_unavailable_reason
    assert rejected.latent_paths == generated.latent_paths
    unavailable = summarize_causal_simulation(model, generated, store=store, inference=None)
    assert unavailable.causal_result is None
    assert unavailable.causal_unavailable_reason is not None


@pytest.mark.contract
def test_data_profile_survives_model_edits(tmp_path, monkeypatch):
    from nof1_causal_lab.actions.data_checks import evaluate_data_checks, require_data_binding
    from nof1_causal_lab.study.records import DataPreparationResult
    from nof1_causal_lab.study.state import apply_effects
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module
    from tests.integration.runner_fixtures import seed_panel

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    panel = seed_panel(store, model_revision=None)
    effects = evaluate_data_checks("TEST", StudyState(), DataPreparationResult(produced=[panel]))
    state = apply_effects(StudyState(), effects.produced)
    assert state.current["data_profile"].derived_from == {"panel": panel.revision}
    assert not state.has("model")
    revised_model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/models"
            / "common/stress_sleep_model.json"
        ).read_text()
    ).revised(question="Another scientific goal")
    revised = store.write_artifact(
        "model",
        produced_by=None,
        derived_from={},
        json_files={"model.json": revised_model.model_dump(mode="json")},
    )
    require_data_binding(store, revised_model, panel.revision)
    current = state.with_artifacts([revised])
    assert current.current["data_profile"] == state.current["data_profile"]


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("groups", "complete_test_model_payload"),
    [
        pytest.param(
            (),
            "common/x_model.json",
            id="groups0",
        ),
        pytest.param(
            ("dynamics",),
            "common/x_model.json",
            id="groups1",
        ),
        pytest.param(
            ("measurement",),
            "common/x_model.json",
            id="groups2",
        ),
        pytest.param(
            ("dynamics", "measurement"),
            "common/x_model.json",
            id="groups3",
        ),
    ],
)
def test_simulation_selects_checks_before_execution_and_persists_only_parameters(
    monkeypatch, groups, complete_test_model_payload
):
    from importlib import import_module

    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.models.ssm import simulation_checks
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
    from nof1_causal_lab.models.ssm.reachability import CheckResult

    action = import_module("nof1_causal_lab.actions.simulate")
    simulation = import_module("nof1_causal_lab.models.ssm.predictive.simulation")
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / complete_test_model_payload
        ).read_text()
    )
    paths = jnp.zeros((2, 3, 1))
    prediction = PredictiveDraws(
        parameters={"future_parameter_site": jnp.array([1.0, 2.0])},
        trajectory=PredictiveTrajectory(
            paths, paths, paths, jnp.ones_like(paths, dtype=bool), paths
        ),
    )
    monkeypatch.setattr(
        simulation,
        "sample_model_laws",
        lambda *_a, **_k: JointPosteriorDraws(prediction.parameters),
    )
    monkeypatch.setattr(simulation, "simulate_predictive_draws", lambda *_a, **_k: prediction)
    called = []

    def measure(group):
        def execute(*_args, **_kwargs):
            assert group in groups, f"Unselected {group} checks executed"
            called.append(group)
            # Selection cannot depend on C-labels or other presentation text.
            return [
                CheckResult.measured(
                    f"{group} renamed check",
                    "X",
                    "ok",
                    "ok",
                    "ok",
                    outcome="passed",
                    measurements=(NumericCriterionEvidence(criterion="fixture", value=1.0),),
                )
            ], []

        return execute

    monkeypatch.setattr(simulation_checks, "measure_construct_dynamics", measure("dynamics"))
    monkeypatch.setattr(simulation_checks, "measure_construct_measurement", measure("measurement"))
    arrays = {}

    def write(values):
        key = str(len(arrays))
        arrays[key] = values
        return key

    original_measure = action.measure_simulation_batch
    monkeypatch.setattr(
        action,
        "measure_simulation_batch",
        lambda model, batch, **_kwargs: original_measure(
            model, batch, groups=groups, edge_contrasts=True, clock=_kwargs["clock"]
        ),
    )
    report = action.simulate(
        model,
        SimulationSpec(end=2.0),
        time_origin=None,
        revision=GitRef(workspace_id="TEST", revision=git_oid(1), path="model.json"),
        write_array=write,
    )
    assert tuple(called) == groups
    assert [finding.subject.check for finding in report.findings] == [
        f"{group} renamed check" for group in groups
    ]
    assert report.parameter_draws.keys() == prediction.parameters.keys()
    np.testing.assert_array_equal(
        arrays[report.parameter_draws["future_parameter_site"]], [1.0, 2.0]
    )
