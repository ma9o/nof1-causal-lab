"""Partial models stay inspectable; numerical operations validate their own inputs."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import polars as pl
import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.execution import FitOperation
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.runners import execute_transition
from nof1_causal_lab.machine.snapshots import ModelReader
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from tests.action_fixtures import edit_and_check
from tests.data_fixtures import metadata_for_model
from tests.git_fixtures import artifact_revision, commit_id
from tests.helpers import complete_test_model, make_model, run_async

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils import storage

    monkeypatch.setenv("DEPLOYMENT_ENV", "development")
    monkeypatch.setattr(storage, "is_remote", lambda: False)
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    return "EXECUTION"


def test_partial_model_revisions_remain_readable_with_capability_findings(workspace):
    partial = make_model(["X", "Y"], [("X", "Y")])
    complete = complete_test_model(partial)
    missing_law = complete.revised(
        distributions={},
        parameters=tuple(p.model_copy(update={"distribution": None}) for p in complete.parameters),
    )
    journal = StudyRepository(workspace)
    state = EpisodeState()
    snapshots = []
    for revision, model in (
        (1, partial),
        (2, complete),
        (3, missing_law),
    ):
        effects = edit_and_check(
            workspace,
            EditModelRequest.model_validate(
                {
                    "model": model.model_dump(mode="json"),
                    "expected_revision": state.current["model"].revision
                    if state.has("model")
                    else None,
                }
            ),
            state,
        )
        state = state.with_artifacts(effects.produced).model_copy(update={"checks": effects.checks})
        journal.append(
            TransitionRecord(
                seq=revision,
                ts="2026-09-14T12:00:00Z",
                action="edit_model",
                inputs={
                    "expected_revision": state.current["model"].revision
                    if state.has("model")
                    else None
                },
                status="applied",
                produced=effects.produced,
                diagnostics=effects.diagnostics,
                checks=effects.checks,
                trace_ids=[],
                resume=None,
            )
        )
        snapshot = ModelReader(workspace).snapshot()
        assert snapshot.model is not None
        assert snapshot.model.value == model
        assert snapshot.findings.specification is not None
        execution = snapshot.findings.specification.value.findings[0]
        assert execution.status == ("passed" if revision == 2 else "not_evaluated")
        assert "execution" not in snapshot.findings.model_dump()
        assert "execution_readiness" not in snapshot.model.value.model_dump()
        assert snapshot.context.can_simulate == (revision == 2)
        snapshots.append(snapshot)
    assert "compiled_ssm" not in ARTIFACT_IDS
    for snapshot in snapshots:
        assert ModelReader(workspace, at=snapshot.context.commit_id).snapshot() == snapshot


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
    model = store.write_artifact(
        "model",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
    )
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:measurements",
        parquet_files={"panel.parquet": pl.DataFrame({"value": [1.0]})},
    )
    state = EpisodeState().with_artifacts([model, panel])
    with pytest.raises(IncompleteModelError):
        run_async(execute_transition(workspace, FitOperation(), state))
    local.assert_not_called()
    remote.assert_not_called()
    assert store.list_revisions("model") == [model.revision]


def test_model_spec_completion_tracks_prior_lineage_and_data_without_requiring_a_result(workspace):
    from nof1_causal_lab.machine.model_spec_results import model_spec_is_current, model_spec_record

    store, journal = ArtifactStore(workspace), StudyRepository(workspace)
    model = complete_test_model(make_model(["X", "Y"], [("X", "Y")]))
    panel = store.write_artifact("panel", derived_from={}, produced_by="run:measurements")
    original = store.write_artifact(
        "model",
        derived_from={"panel": artifact_revision(workspace, "panel", 1)},
        produced_by="run:statistical_model_spec",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    record = TransitionRecord(
        seq=1,
        ts="2026-09-14T12:00:00Z",
        action="edit_model",
        operation_id="statistical_model_spec",
        inputs={},
        status="applied",
        produced=[panel, original],
        diagnostics={"search_queries": {}},
        trace_ids=[],
        resume=None,
    )
    journal.append(record)
    state = EpisodeState().with_artifacts([panel, original])
    selected = model_spec_record(journal.attempts())
    assert selected is not None
    assert selected.seq == record.seq
    assert model_spec_is_current(record, state, store)
    assert ModelReader(workspace).prior_predictive is None

    checked = store.write_artifact(
        "model",
        derived_from={
            "model": artifact_revision(workspace, "model", 1),
            "panel": artifact_revision(workspace, "panel", 1),
        },
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
    state = state.with_artifacts([checked])
    historical = ModelReader(workspace).prior_predictive
    assert historical is not None
    assert historical.value.samples == samples
    assert historical.source.ref.model_dump() == {
        "workspace_id": workspace,
        "revision": commit_id(workspace, 2),
        "path": "logs/transition.json",
    }
    assert historical.source.validity == "fresh"
    assert ModelReader(workspace, at=commit_id(workspace, 1)).prior_predictive is None

    changed = model.revised(
        edges=(model.edges[0].model_copy(update={"description": "Changed mechanism"}),)
    )
    fitted = store.write_artifact(
        "model",
        derived_from={"model": checked.revision, "panel": artifact_revision(workspace, "panel", 1)},
        produced_by="run:posterior",
        json_files={"model.json": model.revised(time_points=(0.0, 1.0)).model_dump(mode="json")},
    )
    assert model_spec_is_current(record, state.with_artifacts([fitted]), store)
    edited = store.write_artifact(
        "model",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": changed.model_dump(mode="json")},
    )
    assert not model_spec_is_current(record, state.with_artifacts([edited]), store)
    assert not model_spec_is_current(
        record,
        state.with_artifacts(
            [panel.model_copy(update={"revision": artifact_revision(workspace, "model", 2)})]
        ),
        store,
    )
    assert (
        model_spec_record([record, record.model_copy(update={"seq": 3, "status": "raised"})])
        == record
    )


def test_refit_after_question_edit_uses_selected_model_and_preserves_current_question(
    workspace, monkeypatch
):
    from nof1_causal_lab.actions import fit as flow
    from nof1_causal_lab.machine.inference import inference_input_revision
    from nof1_causal_lab.machine.runners import _run_posterior
    from nof1_causal_lab.machine.store import read_model

    prior = complete_test_model(make_model(["X", "Y"], [("X", "Y")])).revised(
        question="Does X change Y?"
    )
    fitted = prior.revised(time_points=(0.0, 1.0))
    edited = fitted.revised(question="How does X change Y?")
    store = ArtifactStore(workspace)
    for revision, (model, producer) in enumerate(
        ((prior, "run:statistical_model_spec"), (fitted, "run:posterior"), (edited, None)), 1
    ):
        store.write_artifact(
            "model",
            derived_from={"model": artifact_revision(workspace, "model", revision - 1)}
            if revision > 1
            else {},
            produced_by=producer,
            json_files={"model.json": model.model_dump(mode="json")},
        )
    assert inference_input_revision(
        store, artifact_revision(workspace, "model", 3)
    ) == artifact_revision(workspace, "model", 1)
    store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:measurements",
        json_files={"metadata.json": metadata_for_model(edited).model_dump(mode="json")},
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
    pins: dict[ArtifactId, GitOid] = {
        "model": artifact_revision(workspace, "model", 3),
        "panel": artifact_revision(workspace, "panel", 1),
    }
    effects = run_async(_run_posterior(workspace, store, pins, FitSettingsSpec()))
    assert effects.produced[0].derived_from == pins
    assert effects.diagnostics["input_pins"] == pins
    assert read_model(store, artifact_revision(workspace, "model", 4)).question == edited.question
    assert inference_input_revision(
        store, artifact_revision(workspace, "model", 4)
    ) == artifact_revision(workspace, "model", 1)

    changed_science = edited.revised(measurement_clock="2d")
    store.write_artifact(
        "model",
        derived_from={"model": artifact_revision(workspace, "model", 4)},
        produced_by=None,
        json_files={"model.json": changed_science.model_dump(mode="json")},
    )
    assert inference_input_revision(
        store, artifact_revision(workspace, "model", 5)
    ) == artifact_revision(workspace, "model", 5)
