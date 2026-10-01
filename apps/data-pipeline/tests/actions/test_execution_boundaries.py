"""Partial models stay inspectable; numerical operations validate their own inputs."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock

import polars as pl
import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest, FitRequest
from nof1_causal_lab.actions.runners import run_action
from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import AttemptRecord
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import edit_and_check
from tests.data_fixtures import metadata_for_model
from tests.git_fixtures import artifact_revision
from tests.helpers import make_model, run_async

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
    complete = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'execution_boundaries/partial_model_revisions_remain_readable_with_capability_findings_complete_test_model.json').read_text())
    missing_law = complete.revised(
        distributions={},
        parameters=tuple(
            type(p).model_validate({**p.model_dump(), "distribution": None})
            for p in complete.parameters
        ),
    )
    journal = StudyRepository(workspace)
    state = StudyState()
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
        state = state.with_artifacts(effects.produced)
        state = type(state).model_validate({**state.model_dump(), "checks": effects.checks})
        journal.append(
            AttemptRecord(
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
    from nof1_causal_lab.actions import modal_runners

    monkeypatch.setenv("DEPLOYMENT_ENV", deployment)
    local, remote = Mock(), AsyncMock()
    monkeypatch.setattr(flow, "fit", local)
    monkeypatch.setattr(modal_runners, "run_fit_on_modal", remote)
    store = ArtifactStore(workspace)
    model = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
    )
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        parquet_files={"panel.parquet": pl.DataFrame({"value": [1.0]})},
    )
    state = StudyState().with_artifacts([model, panel])
    with pytest.raises(IncompleteModelError):
        run_async(
            run_action(
                workspace,
                FitRequest(model_revision=model.revision, panel_revision=panel.revision),
                state,
            )
        )
    local.assert_not_called()
    remote.assert_not_called()
    assert store.list_revisions("model") == [model.revision]


def test_refit_after_question_edit_uses_selected_model_and_preserves_current_question(
    workspace, monkeypatch
):
    from nof1_causal_lab.actions import fit as flow
    from nof1_causal_lab.actions.runners import _run_fit
    from nof1_causal_lab.study.store import read_model

    prior = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'execution_boundaries/refit_after_question_edit_uses_selected_model_and_preserves_current_question_complete_test_model.json').read_text()).revised(
        question="Does X change Y?"
    )
    fitted = prior.revised(time_points=(0.0, 1.0))
    edited = fitted.revised(question="How does X change Y?")
    store = ArtifactStore(workspace)
    for revision, (model, producer) in enumerate(
        ((prior, "edit_model"), (fitted, "fit"), (edited, None)), 1
    ):
        store.write_artifact(
            "model",
            derived_from={"model": artifact_revision(workspace, "model", revision - 1)}
            if revision > 1
            else {},
            produced_by=producer,
            json_files={"model.json": model.model_dump(mode="json")},
        )
    store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={"metadata.json": metadata_for_model(edited).model_dump(mode="json")},
        parquet_files={"panel.parquet": pl.DataFrame({"value": [1.0]})},
    )

    def fit(**kwargs):
        assert kwargs["model_spec"] == edited
        return {
            "_model": kwargs["model_spec"].revised(time_points=(0.0, 1.0)),
            "engine_evidence": {},
            "time_origin": kwargs["time_origin"],
            "inference_metadata": {"method": "mock", "n_samples": 1, "duration_seconds": 0.0},
        }

    monkeypatch.setattr(flow, "fit", fit)
    pins: dict[ArtifactId, GitOid] = {
        "model": artifact_revision(workspace, "model", 3),
        "panel": artifact_revision(workspace, "panel", 1),
    }
    effects = run_async(_run_fit(store, pins, FitSettingsSpec()))
    assert effects.produced[0].derived_from == pins
    assert effects.diagnostics["input_pins"] == pins
    assert read_model(store, artifact_revision(workspace, "model", 4)).question == edited.question
