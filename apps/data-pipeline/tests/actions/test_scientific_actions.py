"""Scientific actions depend on selected inputs, independently of authoring recipes."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from nof1_causal_lab.actions.commands import action_command
from nof1_causal_lab.actions.contracts import FitRequest, PrepareDataRequest, SimulateRequest
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState
from nof1_causal_lab.machine.moves import validate_move
from nof1_causal_lab.models.ssm.inference.persistence import condition_model
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws, ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.predictive.parameters import sample_model_laws
from tests.helpers import complete_test_model, make_model
from tests.model_fixtures import parameter_draws

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId


def test_action_contracts_require_inputs_without_authoring_stage_or_admission():
    state = EpisodeState().with_versions(
        [
            ArtifactVersionInfo(
                artifact_id="model", version=2, provenance="human", derived_from={}
            ),
            ArtifactVersionInfo(
                artifact_id="panel", version=1, provenance="computed", derived_from={}
            ),
        ]
    )
    fit = action_command(FitRequest(model_version=2, panel_version=1))
    assert validate_move(state, fit.move) is None
    simulation = action_command(
        SimulateRequest(model_version=2, design=SimulationSpec(times=(0, 1)))
    )
    assert validate_move(state.without(["panel"]), simulation.move) is None
    stale = action_command(FitRequest(model_version=1, panel_version=1))
    assert validate_move(state, stale.move) is None
    imported = action_command(PrepareDataRequest(source="files"))
    assert validate_move(EpisodeState(), imported.move) is None


def test_scientific_tool_transport_shares_the_action_endpoint(monkeypatch):
    from nof1_causal_lab import episode_api, tool_server
    from nof1_causal_lab.machine.status import MoveOutcome

    requests = []

    async def capture(workspace_id, body):
        requests.append((workspace_id, body))
        return MoveOutcome(seq=1, status="applied", state=EpisodeState())

    monkeypatch.setattr(episode_api, "execute_scientific_action", capture)
    client = TestClient(tool_server.app)
    contracts = client.get("/api/tools/scientific")
    assert contracts.status_code == 200
    assert {item["name"] for item in contracts.json()} == {
        "edit_model",
        "prepare_data",
        "fit",
        "simulate",
    }
    result = client.post(
        "/api/tools/scientific/simulate",
        json={
            "workspace_id": "TEST",
            "input": {"model_version": 1, "design": {"kind": "trajectory", "times": [0, 1]}},
        },
    )
    assert result.status_code == 200
    assert result.json()["result"]["status"] == "applied"
    assert requests[0][1] == SimulateRequest(model_version=1, design=SimulationSpec(times=(0, 1)))
    invalid = client.post(
        "/api/tools/scientific/prepare_data",
        json={"workspace_id": "TEST", "input": {"source": "raw_data"}},
    )
    assert invalid.status_code == 422
    assert len(requests) == 1


def test_current_law_sampling_preserves_joint_parameter_atoms():
    model = complete_test_model(make_model(["X", "Y"], [("X", "Y")]))
    original = parameter_draws(model, 3)
    # Each atom has distinct coordinated values; independently resampling marginals
    # would produce combinations absent from the joint law.
    original = {
        name: value * jnp.arange(1, 4).reshape((3,) + (1,) * (value.ndim - 1))
        for name, value in original.items()
    }
    conditioned = condition_model(
        model,
        ParticleMCMCPosterior(
            JointPosteriorDraws(
                original,
                jnp.zeros((3, 2, 2)),
            )
        ),
        times=jnp.array([0.0, 1.0]),
    )
    draws = sample_model_laws(conditioned, draws=24, key=jax.random.PRNGKey(4)).parameters
    for draw in range(24):
        assert any(
            all(np.allclose(draws[name][draw], values[atom]) for name, values in original.items())
            for atom in range(3)
        )
    authored = sample_model_laws(model, draws=5, key=jax.random.PRNGKey(4)).parameters
    assert all(value.shape[0] == 5 and np.isfinite(value).all() for value in authored.values())


@pytest.mark.simulation
@pytest.mark.parametrize("compare", [False, True])
def test_durable_replication_and_comparisons_without_fitting(tmp_path, monkeypatch, compare):
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.machine.moves import ExecOptions, RunOperation, WriteArtifact
    from nof1_causal_lab.machine.runners import execute_transition_locally
    from nof1_causal_lab.machine.snapshots import ModelReader
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
    from nof1_causal_lab.utils import data as data_module
    from tests.helpers import run_async
    from tests.integration.transition_runner_fixtures import panel_frame, scientific_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, journal = ArtifactStore("TEST"), EpisodeJournal("TEST")
    model = scientific_model()
    definition = store.write_version(
        "model",
        provenance="human",
        produced_by=None,
        derived_from={},
        json_files={"model.json": model.model_dump(mode="json")},
    )
    produced = [definition]
    pins: dict[ArtifactId, int] = {"model": 1}
    if compare:
        produced.append(
            store.write_version(
                "panel",
                provenance="computed",
                produced_by="run:measurements",
                derived_from={"model": 1},
                parquet_files={"panel.parquet": panel_frame(n_days=4)},
            )
        )
        pins["panel"] = 1
    state = EpisodeState().with_versions(produced)
    journal.append(
        TransitionRecord(
            seq=1,
            ts="2026-09-15T12:00:00Z",
            status="applied",
            trace_ids=[],
            resume=None,
            move=WriteArtifact(artifact_id="model", expected_model_version=0),
            produced=produced,
        )
    )
    effects = run_async(
        execute_transition_locally(
            "TEST",
            "simulate",
            pins,
            state,
            ExecOptions(
                simulation=SimulationSpec(times=(-1.0, 0.0, 1.0, 2.0, 3.0), draws=4),
                comparison_panel_version=1 if compare else None,
            ),
        )
    )
    report = SimulationReport.model_validate(effects.diagnostics["report"])
    assert store.read_array(report.latent_paths).shape == (4, 5, 2)
    assert store.read_array(report.observations).shape == (4, 5, 2)
    assert (report.predictive_checks is not None) == compare
    assert report.comparison_panel_version == (1 if compare else None)
    assert not effects.produced
    assert any(finding.check.startswith("C5c") for finding in report.findings)
    assert not any(finding.check.startswith("C5d") for finding in report.findings)
    journal.append(
        TransitionRecord(
            seq=2,
            ts="2026-09-15T12:01:00Z",
            status="applied",
            trace_ids=[],
            resume=None,
            move=RunOperation(operation_id="simulate"),
            diagnostics=effects.diagnostics,
        )
    )
    current = ModelReader("TEST").simulation()
    assert current.value == report
    assert current.source.validity == "fresh"
    edited = store.write_version(
        "model",
        provenance="human",
        produced_by=None,
        derived_from={},
        json_files={
            "model.json": model.revised(question="Revised question").model_dump(mode="json")
        },
    )
    journal.append(
        TransitionRecord(
            seq=3,
            ts="2026-09-15T12:02:00Z",
            status="applied",
            trace_ids=[],
            resume=None,
            move=WriteArtifact(artifact_id="model", expected_model_version=1),
            produced=[edited],
        )
    )
    assert ModelReader("TEST").simulation().source.validity == "stale"
    assert ModelReader("TEST", at_seq=2).simulation().source.validity == "fresh"


@pytest.mark.simulation
@pytest.mark.parametrize("process_noise", [False, True])
def test_retained_forecast_and_timed_intervention_preserve_joint_starts(
    tmp_path, monkeypatch, process_noise
):
    from nof1_causal_lab.actions.simulate import simulate
    from nof1_causal_lab.artifacts.identity import ModelRevision
    from nof1_causal_lab.artifacts.scenarios import ScenarioClamp
    from nof1_causal_lab.machine.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    model = complete_test_model(make_model(["X", "Y"], [("X", "Y")]))
    from nof1_causal_lab.models.ssm import numerics as numeric

    states = numeric.state_ids(model)
    paths = jnp.asarray([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    model = condition_model(
        model,
        ParticleMCMCPosterior(JointPosteriorDraws(parameter_draws(model, 2), paths)),
        times=jnp.array([0.0, 1.0]),
    )
    store = ArtifactStore("TEST")
    design = SimulationSpec(
        times=(1.0, 1.5, 2.0),
        draws=3,
        initial_state="retained",
        state_time=1.0,
        process_noise=process_noise,
        observation_noise=False,
        checks=(),
        interventions=(ScenarioClamp(target=states[0], mode="set", value=2.0, from_day=0.5),),
    )
    report = simulate(
        model,
        design,
        revision=ModelRevision(workspace_id="TEST", version=2),
        write_array=store.write_array,
    )
    action = store.read_array(report.latent_paths)
    assert report.reference_latent_paths is not None
    reference = store.read_array(report.reference_latent_paths)
    np.testing.assert_allclose(action[:, 0], reference[:, 0])
    for start in action[:, 0]:
        assert any(np.allclose(start, atom) for atom in np.asarray(paths[:, -1]))
    np.testing.assert_allclose(action[:, 1:, 0], 2.0, atol=1e-5)
    assert np.isfinite(action).all()
    emissions = store.read_array(report.observations)
    assert np.isnan(emissions[:, :-1]).all()
    assert np.isfinite(emissions[:, -1]).all()


@pytest.mark.simulation
def test_causal_action_uses_common_generator_and_requires_matching_engine_evidence(
    tmp_path, monkeypatch
):
    from nof1_causal_lab.actions.scenarios import simulate_causal
    from nof1_causal_lab.artifacts.identity import ModelRevision
    from nof1_causal_lab.artifacts.scenarios import (
        ScenarioClamp,
        ScenarioQueryInput,
        ScenarioRequest,
    )
    from nof1_causal_lab.artifacts.simulation import CausalSimulationSpec
    from nof1_causal_lab.machine.store import ArtifactStore
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.utils import data as data_module
    from tests.inference_fixtures import inference_log

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    model = complete_test_model(make_model(["X", "Y"], [("X", "Y")]))
    states = numeric.state_ids(model)
    model = model.revised(default_outcome=states[1])
    model = condition_model(
        model,
        ParticleMCMCPosterior(JointPosteriorDraws(parameter_draws(model, 2), jnp.zeros((2, 2, 2)))),
        times=jnp.array([0.0, 1.0]),
    )
    design = CausalSimulationSpec(
        query=ScenarioRequest(
            clamps=[ScenarioClamp(target=states[0], mode="set", value=1.0)],
            outcome=states[1],
            readout=ScenarioQueryInput(horizon_days=1),
        ),
        draws=2,
    )
    record = inference_log(model)
    result = simulate_causal(
        model,
        design,
        revision=ModelRevision(workspace_id="TEST", version=2),
        store=ArtifactStore("TEST"),
        inference=record,
    )
    assert result.causal_result is not None
    assert result.reference_latent_paths is not None
    assert result.causal_result.model.version == 2
    bad = record.model_copy(
        update={
            "diagnostics": {
                **record.diagnostics,
                "engine_evidence": {"engine": "laplace", "latent_transition": "gaussian"},
            }
        }
    )
    with pytest.raises(ValueError, match="production particle"):
        simulate_causal(
            model,
            design,
            revision=ModelRevision(workspace_id="TEST", version=2),
            store=ArtifactStore("TEST"),
            inference=bad,
        )


def test_data_profile_reuse_and_historical_selection(tmp_path, monkeypatch):
    from nof1_causal_lab.machine.derivations import complete_derivation_cascade
    from nof1_causal_lab.machine.graph import transition_spec
    from nof1_causal_lab.machine.selection import resolve_input_pins
    from nof1_causal_lab.machine.store import ArtifactStore
    from nof1_causal_lab.utils import data as data_module
    from tests.integration.transition_runner_fixtures import panel_frame, scientific_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    model = scientific_model()
    definition = store.write_version(
        "model",
        provenance="human",
        produced_by=None,
        derived_from={},
        json_files={"model.json": model.model_dump(mode="json")},
    )
    panel = store.write_version(
        "panel",
        provenance="computed",
        produced_by="run:measurements",
        derived_from={"model": 1},
        parquet_files={"panel.parquet": panel_frame(n_days=4)},
    )
    effects = complete_derivation_cascade(store, EpisodeState(), [definition, panel])
    state = EpisodeState().with_versions(effects.produced)
    assert state.current["data_profile"].derived_from == {"panel": 1}
    revised = store.write_version(
        "model",
        provenance="human",
        produced_by=None,
        derived_from={},
        json_files={
            "model.json": model.revised(question="Another scientific goal").model_dump(mode="json")
        },
    )
    effects = complete_derivation_cascade(store, state, [revised])
    assert "data_profile" not in [info.artifact_id for info in effects.produced]
    current = state.with_versions(effects.produced)
    assert resolve_input_pins(
        store, current, transition_spec("posterior"), {"model": 1, "panel": 1}
    ) == {"model": 1, "panel": 1}
    with pytest.raises(ValueError, match="measurement definitions"):
        resolve_input_pins(store, current, transition_spec("posterior"), {"model": 2, "panel": 1})


def test_offline_predictive_migration_preserves_source_and_provenance(tmp_path):
    import json
    from pathlib import Path

    from scripts.migrate_scientific_actions import migrate

    source = tmp_path / "source"
    logs = source / "episode" / "journal"
    logs.mkdir(parents=True)
    checks = json.loads(
        (Path(__file__).parents[4] / "data/DEMO/fixture/predictive_checks.json").read_text()
    )
    record = {
        "seq": 3,
        "move": {"operation_id": "posterior"},
        "produced": [{"artifact_id": "model", "version": 2}],
        "diagnostics": {"input_pins": {"panel": 1}, "report": {"ppc": checks}},
    }
    original = json.dumps(record)
    (logs / "000003.json").write_text(original)
    target = tmp_path / "migrated"
    inventory = migrate(source, target, check=True)
    assert not target.exists()
    assert inventory["predictive_archives"] == ["000003.json"]
    migrate(source, target)
    assert (logs / "000003.json").read_text() == original
    assert (
        "ppc"
        not in json.loads((target / "episode/journal/000003.json").read_text())["diagnostics"][
            "report"
        ]
    )
    archive = json.loads((target / "episode/predictive-archive/000003.json").read_text())
    assert archive["checks"] == checks
    assert archive["model_version"] == 2
    assert archive["arrays_available"] is False
