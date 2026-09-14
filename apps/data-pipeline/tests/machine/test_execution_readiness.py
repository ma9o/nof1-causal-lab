"""Execution follows the selected scientific revision without a persisted receipt."""

from unittest.mock import AsyncMock, Mock

import polars as pl
import pytest

from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.moves import ExecOptions, WriteArtifact
from nof1_causal_lab.machine.runners import execute_transition
from nof1_causal_lab.machine.snapshots import ModelReader
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, TransitionRecord
from nof1_causal_lab.machine.writes import execute_write
from tests.helpers import complete_test_model, make_model, run_async


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils import storage

    monkeypatch.setenv("DEPLOYMENT_ENV", "development")
    monkeypatch.setattr(storage, "is_remote", lambda: False)
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    return "READINESS"


def test_readiness_is_derived_for_each_selected_revision(workspace):
    partial = make_model(["X", "Y"], [("X", "Y")])
    complete = complete_test_model(partial)
    missing_law = complete.revised(
        parameters=tuple(p.model_copy(update={"distribution": None}) for p in complete.parameters)
    )
    journal = EpisodeJournal(workspace)
    state = EpisodeState()
    snapshots = []
    for version, model, ready in (
        (1, partial, False),
        (2, complete, True),
        (3, missing_law, False),
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
                trace_ids=[],
                resume=None,
            )
        )
        snapshot = ModelReader(workspace).snapshot()
        assert snapshot.model is not None
        finding = snapshot.findings.execution
        assert finding is not None
        assert finding.source == snapshot.model.source
        assert finding.value.ready is ready
        assert bool(finding.value.anchor_certificates) is ready
        assert bool(finding.value.unmet_requirements) is not ready
        assert "execution_readiness" not in snapshot.model.value.model_dump()
        snapshots.append(snapshot)
    assert "compiled_ssm" not in ARTIFACT_IDS
    for snapshot in snapshots:
        assert ModelReader(workspace, at_seq=snapshot.context.seq).snapshot() == snapshot


@pytest.mark.parametrize("deployment", ["development", "production"])
def test_incomplete_model_is_rejected_before_local_or_remote_inference(
    workspace, monkeypatch, deployment
):
    from nof1_causal_lab.flows import modal_runners
    from nof1_causal_lab.flows.transitions.inference import flow

    monkeypatch.setenv("DEPLOYMENT_ENV", deployment)
    local, remote = Mock(), AsyncMock()
    monkeypatch.setattr(flow, "run_inference_with_data", local)
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
