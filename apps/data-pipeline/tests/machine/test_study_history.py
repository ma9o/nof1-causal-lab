"""Git branch isolation, publication and commit-local evidence across process restarts."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.execution import RetractedArtifact
from nof1_causal_lab.machine.history import BranchConflict, StudyRepository
from nof1_causal_lab.machine.snapshots import ModelReader, SnapshotRevisionNotFound
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from nof1_causal_lab.utils import data as data_module

pytestmark = pytest.mark.contract


@pytest.fixture
def study(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    return StudyRepository("STUDY"), ArtifactStore("STUDY")


def _record(seq, *, branch="main", status="applied", **effects):
    return TransitionRecord(
        seq=seq,
        branch=branch,
        ts="2026-09-23T12:00:00+00:00",
        action="prepare_data",
        operation_id="raw_data",
        inputs={},
        status=status,
        trace_ids=[],
        resume=None,
        **effects,
    )


def _model(store, question, parent=0):
    return store.write_artifact(
        "model",
        produced_by=None,
        derived_from={"model": parent} if parent else {},
        json_files={"model.json": {"question": question}},
    )


def test_branches_share_ancestry_but_isolate_state_and_logs(study):
    repository, store = study
    first = _model(store, "Does X change Y?")
    root = repository.append(_record(1, produced=[first]), logs={"events.json": b"[1]"})
    repository.create_branch("alternative", at=root)
    second = _model(store, "Does X change Y at night?", first.revision)
    main = repository.append(_record(2, produced=[second]), expected_head=root)
    fork = repository.append(
        _record(
            3,
            branch="alternative",
            retracted=[RetractedArtifact(artifact_id="model", reason_ref="finding:revisit")],
            diagnostics={"finding": "alternative only"},
        ),
        expected_head=root,
        logs={"events.json": b"[3]"},
    )
    reopened = StudyRepository("STUDY")
    assert reopened.record(main).parent_ids == [root]
    assert reopened.record(fork).parent_ids == [root]
    main_model = ModelReader("STUDY", branch="main").model
    assert main_model is not None
    assert main_model.question == "Does X change Y at night?"
    assert ModelReader("STUDY", branch="alternative").model is None
    initial_model = ModelReader("STUDY", at=root).model
    assert initial_model is not None
    assert initial_model.question == "Does X change Y?"
    assert [record.seq for record in ModelReader("STUDY", branch="main").records] == [1, 2]
    assert [record.seq for record in ModelReader("STUDY", branch="alternative").records] == [1, 3]
    assert reopened.read_file(root, "logs/events.json") == b"[1]"
    assert reopened.read_file(main, "logs/events.json") == b"[]"
    assert reopened.read_file(fork, "logs/events.json") == b"[3]"
    assert not (repository.path.parent / "journal").exists()


def test_failed_attempt_and_stale_publication_do_not_advance_branch(study):
    repository, _ = study
    base = repository.head()
    attempt_id = uuid4()
    accepted = _record(1, attempt_id=attempt_id)
    head = repository.append(accepted, expected_head=base)
    assert repository.append(accepted, expected_head=base) == head
    persisted = StudyRepository("STUDY").dispatched_attempt(attempt_id)
    assert persisted is not None
    assert persisted.commit_id == head
    with pytest.raises(FileExistsError):
        repository.append(_record(2, attempt_id=attempt_id), expected_head=head)
    with pytest.raises(BranchConflict):
        repository.append(_record(2), expected_head=base)
    assert repository.read_attempt(2) is None
    failed = repository.append(_record(2, status="raised"), expected_head=base)
    assert repository.head() == head
    assert repository.record(failed).status == "raised"
    with pytest.raises(SnapshotRevisionNotFound):
        ModelReader("STUDY", at=failed)
    assert [record.seq for record in StudyRepository("STUDY").attempts()] == [1, 2]
    assert len(repository.branches()) == 1
    with pytest.raises(ValueError, match="Invalid branch"):
        repository.create_branch("../escape", at=head)


def test_action_captures_selected_branch_before_validation(study, monkeypatch):
    from nof1_causal_lab.machine.temporal import workflow as episode_workflow
    from nof1_causal_lab.machine.temporal.activities import journal_activity, read_branch_activity
    from nof1_causal_lab.machine.temporal.messages import ActionRequest, EpisodeInit

    repository, store = study
    first = _model(store, "Shared question")
    root = repository.append(_record(1, produced=[first]))
    repository.create_branch("alternative", at=root)
    second = _model(store, "Main question", first.revision)
    repository.append(_record(2, produced=[second]))

    async def execute(name, input, **kwargs):
        if name == "read_branch_activity":
            return await read_branch_activity(input)
        if name == "journal_activity":
            return await journal_activity(input)
        if name in {"collect_completed_runs_activity", "emit_action_message_activity"}:
            return None
        raise AssertionError(f"Rejected action must not execute: {name}")

    monkeypatch.setattr(episode_workflow.workflow, "execute_activity", execute)
    monkeypatch.setattr(episode_workflow.workflow, "now", lambda: datetime(2026, 9, 28, tzinfo=UTC))
    worker = episode_workflow.EpisodeWorkflow(EpisodeInit(workspace_id="STUDY", initial_seq=2))
    from tests.helpers import run_async

    result = run_async(
        episode_workflow.EpisodeWorkflow.execute_action(
            worker,
            ActionRequest(
                branch="alternative",
                request=EditModelRequest(
                    expected_revision=second.revision, model=ModelSpec(question="Stale edit")
                ),
            ),
        )
    )
    assert result.status == "rejected"
    assert result.state.current["model"].revision == first.revision
    assert result.branch == "alternative"
    assert repository.head("alternative") == root
