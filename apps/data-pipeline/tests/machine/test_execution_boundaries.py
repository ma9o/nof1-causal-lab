"""Partial models stay inspectable; numerical operations validate their own inputs."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import polars as pl
import pytest

from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState
from nof1_causal_lab.machine.moves import ExecOptions, WriteArtifact
from nof1_causal_lab.machine.runners import execute_transition
from nof1_causal_lab.machine.snapshots import ModelReader
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
from nof1_causal_lab.machine.writes import execute_write
from tests.helpers import complete_test_model, make_model, run_async

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils import storage

    monkeypatch.setenv("DEPLOYMENT_ENV", "development")
    monkeypatch.setattr(storage, "is_remote", lambda: False)
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    return "EXECUTION"


def test_partial_model_revisions_remain_readable_without_execution_checks(workspace):
    partial = make_model(["X", "Y"], [("X", "Y")])
    complete = complete_test_model(partial)
    missing_law = complete.revised(
        distributions={},
        parameters=tuple(p.model_copy(update={"distribution": None}) for p in complete.parameters),
    )
    journal = EpisodeJournal(workspace)
    state = EpisodeState()
    snapshots = []
    for version, model in (
        (1, partial),
        (2, complete),
        (3, missing_law),
    ):
        effects = execute_write(
            workspace,
            "model",
            model.model_dump(mode="json"),
            "human",
            state,
            expected_model_version=version - 1,
        )
        state = state.with_versions(effects.produced)
        journal.append(
            TransitionRecord(
                seq=version,
                ts="2026-09-14T12:00:00Z",
                move=WriteArtifact(artifact_id="model", expected_model_version=version - 1),
                status="applied",
                produced=effects.produced,
                diagnostics=effects.diagnostics,
                trace_ids=[],
                resume=None,
            )
        )
        with patch.object(
            ModelSpec, "check_execution", side_effect=AssertionError("read compiled")
        ):
            snapshot = ModelReader(workspace).snapshot()
        assert snapshot.model is not None
        assert snapshot.model.value == model
        assert snapshot.findings.specification is not None
        execution = snapshot.findings.specification.value.findings[0]
        assert execution.status == ("passed" if version == 2 else "not_evaluated")
        assert "execution" not in snapshot.findings.model_dump()
        assert "execution_readiness" not in snapshot.model.value.model_dump()
        assert not snapshot.context.can_simulate
        snapshots.append(snapshot)
    assert "compiled_ssm" not in ARTIFACT_IDS
    for snapshot in snapshots:
        assert ModelReader(workspace, at_seq=snapshot.context.seq).snapshot() == snapshot


@pytest.mark.parametrize(
    (
        "model_kind",
        "panel_version",
        "specification_current",
        "posterior_needed",
        "specification_needed",
    ),
    [
        ("partial", 1, False, False, True),
        ("complete", 1, False, True, True),
        ("complete", 1, True, True, False),
        ("fitted", 1, True, False, False),
        ("fitted", 2, False, True, True),
    ],
)
def test_auto_navigation_uses_operation_requirements_and_input_versions(
    model_kind, panel_version, specification_current, posterior_needed, specification_needed
):
    from nof1_causal_lab.episode_api import _needs_run
    from nof1_causal_lab.machine.graph import transition_spec

    model = make_model(["X", "Y"], [("X", "Y")])
    if model_kind != "partial":
        model = complete_test_model(model)
    state = EpisodeState().with_versions(
        [
            ArtifactVersionInfo(
                artifact_id="model",
                version=1,
                provenance="computed",
                produced_by="run:posterior" if model_kind == "fitted" else "write:model",
                derived_from={"panel": 1} if model_kind == "fitted" else {},
            ),
            ArtifactVersionInfo(artifact_id="panel", version=panel_version, provenance="computed"),
        ]
    )
    assert _needs_run(state, transition_spec("posterior"), model) is posterior_needed
    assert (
        _needs_run(
            state,
            transition_spec("statistical_model_spec"),
            model,
            specification_current=specification_current,
        )
        is specification_needed
    )


@pytest.mark.parametrize("deployment", ["development", "production"])
def test_incomplete_model_is_rejected_before_local_or_remote_inference(
    workspace, monkeypatch, deployment
):
    from nof1_causal_lab.actions import fit as flow
    from nof1_causal_lab.flows import modal_runners

    monkeypatch.setenv("DEPLOYMENT_ENV", deployment)
    local, remote = Mock(), AsyncMock()
    monkeypatch.setattr(flow, "fit", local)
    monkeypatch.setattr(modal_runners, "run_transition_on_modal", remote)
    store = ArtifactStore(workspace)
    model = store.write_version(
        "model",
        provenance="human",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
    )
    panel = store.write_version(
        "panel",
        provenance="computed",
        derived_from={"model": 1},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": pl.DataFrame({"value": [1.0]})},
    )
    state = EpisodeState().with_versions([model, panel])
    with pytest.raises(IncompleteModelError):
        run_async(execute_transition(workspace, "posterior", state, ExecOptions()))
    local.assert_not_called()
    remote.assert_not_called()
    assert store.list_versions("model") == [1]


def test_model_spec_completion_tracks_prior_lineage_and_data_without_requiring_a_result(workspace):
    from nof1_causal_lab.episode_api import _needs_run
    from nof1_causal_lab.machine.graph import transition_spec
    from nof1_causal_lab.machine.model_spec_results import model_spec_is_current, model_spec_record
    from nof1_causal_lab.machine.moves import RunOperation

    store, journal = ArtifactStore(workspace), EpisodeJournal(workspace)
    model = complete_test_model(make_model(["X", "Y"], [("X", "Y")]))
    panel = store.write_version(
        "panel", provenance="computed", derived_from={}, produced_by="run:measurements"
    )
    original = store.write_version(
        "model",
        provenance="computed",
        derived_from={"panel": 1},
        produced_by="run:statistical_model_spec",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    record = TransitionRecord(
        seq=1,
        ts="2026-09-14T12:00:00Z",
        move=RunOperation(operation_id="statistical_model_spec"),
        status="applied",
        produced=[panel, original],
        diagnostics={"search_queries": {}},
        trace_ids=[],
        resume=None,
    )
    journal.append(record)
    state = EpisodeState().with_versions([panel, original])
    assert model_spec_record(journal.read_all()) == record
    assert model_spec_is_current(record, state, store)
    assert ModelReader(workspace).prior_predictive() is None
    assert not _needs_run(
        state,
        transition_spec("statistical_model_spec"),
        model,
        specification_current=model_spec_is_current(record, state, store),
    )

    checked = store.write_version(
        "model",
        provenance="computed",
        derived_from={"model": 1, "panel": 1},
        produced_by="run:statistical_model_spec",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    samples = {model.indicators[0].id: [0.25, 0.5]}
    record = record.model_copy(
        update={
            "seq": 2,
            "produced": [checked],
            "diagnostics": {"prior_predictive": {"samples": samples, "diagnostics": []}},
        }
    )
    journal.append(record)
    state = state.with_versions([checked])
    historical = ModelReader(workspace).prior_predictive()
    assert historical is not None
    assert historical.value.samples == samples
    assert historical.source.ref.model_dump() == {"seq": 2}
    assert historical.source.validity == "fresh"
    assert ModelReader(workspace, at_seq=1).prior_predictive() is None

    changed = model.revised(
        edges=(model.edges[0].model_copy(update={"description": "Changed mechanism"}),)
    )
    fitted = store.write_version(
        "model",
        provenance="computed",
        derived_from={"model": checked.version, "panel": 1},
        produced_by="run:posterior",
        json_files={"model.json": model.revised(time_points=(0.0, 1.0)).model_dump(mode="json")},
    )
    assert model_spec_is_current(record, state.with_versions([fitted]), store)
    edited = store.write_version(
        "model",
        provenance="human",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": changed.model_dump(mode="json")},
    )
    assert not model_spec_is_current(record, state.with_versions([edited]), store)
    assert not model_spec_is_current(
        record, state.with_versions([panel.model_copy(update={"version": 2})]), store
    )
    assert (
        model_spec_record([record, record.model_copy(update={"seq": 3, "status": "raised"})])
        == record
    )


def test_refit_after_question_edit_uses_selected_model_and_preserves_current_question(
    workspace, monkeypatch
):
    from nof1_causal_lab.actions import fit as flow
    from nof1_causal_lab.machine.derivations import read_model
    from nof1_causal_lab.machine.inference import inference_input_version, read_prior_model
    from nof1_causal_lab.machine.runners import _run_posterior

    prior = complete_test_model(make_model(["X", "Y"], [("X", "Y")])).revised(
        question="Does X change Y?"
    )
    fitted = prior.revised(time_points=(0.0, 1.0))
    edited = fitted.revised(question="How does X change Y?")
    store = ArtifactStore(workspace)
    for version, (model, producer) in enumerate(
        ((prior, "run:statistical_model_spec"), (fitted, "run:posterior"), (edited, None)), 1
    ):
        store.write_version(
            "model",
            provenance="computed" if producer else "human",
            derived_from={"model": version - 1} if version > 1 else {},
            produced_by=producer,
            json_files={"model.json": model.model_dump(mode="json")},
        )
    assert inference_input_version(store, 3) == 1
    expected = prior.revised(question=edited.question)
    assert read_prior_model(store, 3) == expected
    store.write_version(
        "panel",
        provenance="computed",
        derived_from={"model": 3},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": pl.DataFrame({"value": [1.0]})},
    )

    def fit(**kwargs):
        assert kwargs["model_spec"] == edited
        return {
            "_model": kwargs["model_spec"].revised(time_points=(0.0, 1.0)),
            "engine_evidence": {},
            "inference_metadata": {"method": "mock", "n_samples": 1, "duration_seconds": 0.0},
        }

    monkeypatch.setattr(flow, "fit", fit)
    pins: dict[ArtifactId, int] = {"model": 3, "panel": 1}
    effects = run_async(_run_posterior(workspace, store, pins, ExecOptions()))
    assert effects.produced[0].derived_from == pins
    assert effects.diagnostics["input_pins"] == pins
    assert read_model(store, 4).question == edited.question
    assert inference_input_version(store, 4) == 1

    changed_science = edited.revised(measurement_clock="2d")
    store.write_version(
        "model",
        provenance="human",
        derived_from={"model": 4},
        produced_by=None,
        json_files={"model.json": changed_science.model_dump(mode="json")},
    )
    assert inference_input_version(store, 5) == 5
