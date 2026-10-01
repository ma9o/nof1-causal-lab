"""Git branch isolation, publication and commit-local evidence across process restarts."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.study.history import BranchConflict, StudyRepository
from nof1_causal_lab.study.records import AttemptRecord
from nof1_causal_lab.study.snapshots import ModelReader, SnapshotRevisionNotFound
from nof1_causal_lab.study.state import RetractedArtifact
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils import data as data_module

pytestmark = pytest.mark.contract


@pytest.fixture
def study(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    return StudyRepository("STUDY"), ArtifactStore("STUDY")


def _record(seq, *, branch="main", status="applied", **effects):
    return AttemptRecord(
        seq=seq,
        branch=branch,
        ts="2026-09-23T12:00:00+00:00",
        action="prepare_data",
        inputs={},
        status=status,
        trace_ids=[],
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
    root = repository.append(_record(1, produced=[first]), logs={"notes.json": b"[1]"})
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
        logs={"notes.json": b"[3]"},
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
    assert reopened.read_file(root, "logs/notes.json") == b"[1]"
    with pytest.raises(KeyError):
        reopened.read_file(main, "logs/notes.json")
    assert reopened.read_file(fork, "logs/notes.json") == b"[3]"
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
    from nof1_causal_lab.actions.temporal import workflow as study_workflow
    from nof1_causal_lab.actions.temporal.activities import journal_activity, read_branch_activity
    from nof1_causal_lab.actions.temporal.messages import ActionRequest, StudyInit

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
        if name == "collect_completed_runs_activity":
            return None
        raise AssertionError(f"Rejected action must not execute: {name}")

    monkeypatch.setattr(study_workflow.workflow, "execute_activity", execute)
    monkeypatch.setattr(study_workflow.workflow, "now", lambda: datetime(2026, 9, 28, tzinfo=UTC))
    monkeypatch.setattr(study_workflow.workflow, "upsert_memo", lambda _: None)
    worker = study_workflow.StudyWorkflow(StudyInit(workspace_id="STUDY", initial_seq=2))
    from tests.helpers import run_async

    run_async(
        study_workflow.StudyWorkflow.execute_action(
            worker,
            ActionRequest(
                branch="alternative",
                request=EditModelRequest(
                    expected_revision=second.revision, model=ModelSpec(question="Stale edit")
                ),
            ),
        )
    )
    rejected = repository.read_attempt(3)
    assert rejected is not None
    assert (rejected.status, rejected.branch) == ("rejected", "alternative")
    assert rejected.parent_ids == [root]
    assert repository.state(root).current["model"].revision == first.revision
    assert repository.head("alternative") == root


@pytest.mark.parametrize("fails", [False, True])
def test_data_comparison_is_a_saved_leaf_at_dispatch_head(study, monkeypatch, fails):
    from temporalio.exceptions import ActivityError, ApplicationError

    from nof1_causal_lab.actions import data_diff
    from nof1_causal_lab.actions.reads import read_action_body
    from nof1_causal_lab.actions.results import DataComparisonResult
    from nof1_causal_lab.actions.temporal import workflow as study_workflow
    from nof1_causal_lab.actions.temporal.activities import journal_activity, read_branch_activity
    from nof1_causal_lab.actions.temporal.messages import ActionRequest, StudyInit
    from tests.helpers import run_async

    repository, store = study
    root = repository.append(_record(1, produced=[_model(store, "Original")]))
    head = repository.append(_record(2, produced=[_model(store, "Changed while queued")]))
    left = data_diff.DataRef(kind="panel", revision=root)
    right = data_diff.DataRef(kind="panel", revision=head)
    request = data_diff.DataDiffRequest(left=left, right=right)
    report = data_diff.DataDiffReport(left=(left,), right=(right,), variables=())
    reads = []

    def compare(workspace, selection):
        reads.append((workspace, selection))
        if fails:
            raise ValueError("The selected panel has no saved observations")
        return report

    async def execute(name, input, **kwargs):
        if name == "read_branch_activity":
            return await read_branch_activity(input)
        if name == "journal_activity":
            try:
                # A journal activity retry reads the saved report rather than comparing again.
                commit = await journal_activity(input)
                assert await journal_activity(input) == commit
                return commit
            except ApplicationError as exc:
                raise ActivityError(
                    "comparison failed",
                    scheduled_event_id=1,
                    started_event_id=2,
                    identity="test",
                    activity_type=name,
                    activity_id="test",
                    retry_state=None,
                ) from exc
        if name == "collect_completed_runs_activity":
            return None
        raise AssertionError(f"A comparison must not execute scientific actions: {name}")

    monkeypatch.setattr(data_diff, "read_data_diff", compare)
    monkeypatch.setattr(study_workflow.workflow, "execute_activity", execute)
    monkeypatch.setattr(study_workflow.workflow, "now", lambda: datetime(2026, 9, 30, tzinfo=UTC))
    monkeypatch.setattr(study_workflow.workflow, "upsert_memo", lambda _: None)
    worker = study_workflow.StudyWorkflow(StudyInit(workspace_id="STUDY", initial_seq=2))
    action = ActionRequest(request=request, expected_head=root)
    run_async(study_workflow.StudyWorkflow.execute_action(worker, action))
    leaf = repository.read_attempt(3)
    assert leaf is not None
    assert leaf.parent_ids == [root]
    assert leaf.action == "data_diff"
    assert leaf.status == ("raised" if fails else "applied")
    assert leaf.diagnostics == {}
    assert leaf.produced == leaf.retracted == []
    assert leaf.checks is None
    assert repository.state(leaf.commit_id) == repository.state(root)
    assert repository.head() == head
    assert len(reads) == 1
    if fails:
        assert "no saved observations" in leaf.error_message
    else:
        body = read_action_body("STUDY", leaf)
        assert isinstance(body, DataComparisonResult)
        assert body.report == report
        with pytest.raises(ValueError, match="read-only leaf"):
            repository.create_branch("from-comparison", at=leaf.commit_id)


def test_timeline_links_named_arguments_and_check_reads():
    from nof1_causal_lab.study.records import StudyRevision, record_dependencies
    from nof1_causal_lab.study.state import ArtifactRecord

    def oid(n):
        return f"{n:040x}"

    def revision(seq, action, inputs, *produced, status="applied"):
        return StudyRevision(
            seq=seq,
            ts="2026-09-30T12:00:00+00:00",
            action=action,
            inputs=inputs,
            status=status,
            produced=[
                ArtifactRecord(
                    artifact_id=artifact_id,
                    revision=oid(n),
                    derived_from={key: oid(value) for key, value in derived.items()},
                )
                for artifact_id, n, derived in produced
            ],
            trace_ids=[],
            commit_id=oid(100 + seq),
            parent_ids=[oid(99 + seq)],
        )

    records = [
        revision(1, "edit_model", {"expected_revision": None}, ("model", 1, {})),
        revision(2, "simulate", {"model_revision": oid(1), "end": 10}),
        revision(
            3,
            "prepare_data",
            {"input": {"revision": oid(102), "replicate": 0}},
            ("raw_data", 3, {}),
            ("panel", 4, {"raw_data": 3}),
        ),
        revision(
            4,
            "fit",
            {"model_revision": oid(1), "panel_revision": oid(4)},
            ("model", 5, {"model": 1, "panel": 4}),
        ),
        revision(
            5,
            "edit_model",
            {"expected_revision": oid(5)},
            ("model", 6, {"model": 5}),
            ("validation_report", 7, {"model": 6, "panel": 4}),
            # Carried forward unchanged: its first producer keeps it.
            ("panel", 4, {"raw_data": 3}),
        ),
        revision(6, "fit", {"model_revision": oid(6), "panel_revision": oid(4)}, status="raised"),
        # An older request that named no base still revised the previous model.
        revision(8, "edit_model", {}, ("model", 8, {"model": 6})),
        revision(
            7,
            "data_diff",
            {
                "left": {"kind": "panel", "revision": oid(4)},
                "right": {"kind": "simulation", "revision": oid(102)},
            },
        ),
    ]
    assert [
        (item.seq, item.source_seq, item.argument, item.check)
        for item in record_dependencies(records)
    ] == [
        (2, 1, "model", False),
        (3, 2, "simulation", False),
        (4, 1, "model", False),
        (4, 3, "panel", False),
        (5, 4, "model", False),
        (5, 3, "panel", True),
        (6, 5, "model", False),
        (6, 3, "panel", False),
        (8, 5, "model", False),
        (7, 3, "left", False),
        (7, 2, "right", False),
    ]
