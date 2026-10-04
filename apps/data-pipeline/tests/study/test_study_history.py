"""Git publication and commit-local evidence across process restarts."""


from nof1_causal_lab.artifacts.simulation import SimulationEvidence

from nof1_causal_lab.artifacts.predictive_provenance import AuthoredLawProvenance
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from nof1_causal_lab.actions.contracts import EditModelRequest, PrepareDataRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.availability import NotApplicable
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import (
    Applied,
    AttemptRecord,
    DataPreparationResult,
    PrepareAttempt,
    Raised,
    Rejected,
)
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.study.view_models import PanelRef
from nof1_causal_lab.utils import data as data_module
from tests.action_fixtures import applied_record
from tests.model_fixtures import x_y_model

pytestmark = pytest.mark.contract


@pytest.fixture
def study(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    return StudyRepository("STUDY"), ArtifactStore("STUDY")


def _record(seq, *, outcome=None, attempt_id=None, **effects):
    if outcome is not None:
        return AttemptRecord(
            seq=seq,
            attempt_id=attempt_id,
            ts="2026-09-23T12:00:00+00:00",
            attempt=PrepareAttempt(request=None, outcome=outcome, action="prepare_data"),
        )
    return applied_record(
        Applied(result=DataPreparationResult(), effects=ActionEffects(**effects)),
        seq=seq,
        attempt_id=attempt_id,
        ts="2026-09-23T12:00:00+00:00",
    )


def _model(store, clock, parent=0):
    return store.write_artifact(
        "model",
        produced_by=None,
        derived_from={"model": parent} if parent else {},
        json_files={"model.json": {"measurement_clock": clock}},
    )


def test_failed_attempt_and_invalid_publication_do_not_advance_scientific_state(study):
    repository, _ = study
    base = repository.head()
    attempt_id = uuid4()
    accepted = _record(1, attempt_id=attempt_id)
    head = repository.append(accepted, parent_id=base).commit_id
    assert repository.append(accepted, parent_id=base).commit_id == head
    persisted = StudyRepository("STUDY")
    assert str(persisted.repo.references[f"refs/actions/{attempt_id}"].target) == head
    with pytest.raises(FileExistsError):
        repository.append(_record(2, attempt_id=attempt_id), parent_id=head)
    with pytest.raises(RuntimeError, match="journal parent"):
        repository.append(_record(2), parent_id=base)
    assert repository.read_attempt(2) is None
    failed = repository.append(
        _record(2, outcome=Raised(error_type="SavedError", error_message="failure")),
        parent_id=base,
    ).commit_id
    assert repository.head() == head
    assert repository.record(failed).record.attempt.outcome.status == "raised"
    with pytest.raises(StudyLookupError):
        ModelReader("STUDY", at=failed)
    assert [record.record.seq for record in StudyRepository("STUDY").attempts()] == [1, 2]


@pytest.mark.parametrize("fails", [False, True])
def test_data_comparison_is_a_saved_leaf_of_the_journal(study, monkeypatch, fails):
    from temporalio.exceptions import ActivityError, ApplicationError

    from nof1_causal_lab.actions import data_diff
    from nof1_causal_lab.actions.temporal import workflow as study_workflow
    from nof1_causal_lab.actions.temporal.activities import (
        journal_activity,
        read_inputs_activity,
        run_action_activity,
    )
    from nof1_causal_lab.actions.temporal.messages import ActionRequest, StudyInit
    from tests.helpers import run_async

    repository, store = study
    from tests.action_fixtures import question_root

    root = question_root("STUDY").commit_id
    head = repository.append(_record(2, produced=[_model(store, "2d")])).commit_id
    left = PanelRef(revision=root)
    right = PanelRef(revision=head)
    request = data_diff.DataDiffRequest(left=left, right=right)
    report = data_diff.DataDiffReport(left=(left,), right=(right,), variables=())
    reads = []

    def compare(workspace, selection):
        reads.append((workspace, selection))
        if fails:
            raise StudyLookupError("The selected panel has no saved observations")
        return report

    async def execute(name, payload, **kwargs):
        name = name if isinstance(name, str) else name.__name__
        if name == "read_inputs_activity":
            return await read_inputs_activity(payload)
        if name == "run_action_activity":
            return await run_action_activity(payload)
        if name == "journal_activity":
            try:
                # A journal activity retry reads the saved report rather than comparing again.
                commit = await journal_activity(payload)
                assert await journal_activity(payload) == commit
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
    action = ActionRequest(request=request)
    run_async(study_workflow.StudyWorkflow.execute_action(worker, action))
    leaf = repository.read_attempt(3)
    assert leaf is not None
    assert leaf.parent_ids == (head,)
    assert leaf.record.attempt.action == "data_diff"
    outcome = leaf.record.attempt.outcome
    assert outcome.status == ("rejected" if fails else "applied")
    assert repository.state(leaf.commit_id) == repository.state(head)
    assert repository.head() == head
    assert len(reads) == 1
    if fails:
        assert isinstance(outcome, Rejected)
        assert "no saved observations" in outcome.detail
    else:
        assert isinstance(outcome, Applied)
        assert outcome.result is None
        assert outcome.effects.produced == outcome.effects.retracted == ()
        assert leaf.record.attempt.request == request
        assert "report" not in leaf.record.model_dump_json(round_trip=True)


def test_timeline_links_named_arguments_and_check_reads():
    from nof1_causal_lab.study.records import StudyRevision, record_dependencies
    from nof1_causal_lab.study.state import ArtifactRecord

    def oid(n):
        return f"{n:040x}"

    from datetime import date

    from nof1_causal_lab.actions.contracts import FitRequest, SimulateRequest
    from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.artifacts.simulation import SimulationReport, SimulationSpec
    from nof1_causal_lab.study.records import (
        FitAttempt,
        ModelSimulationResult,
    )
    from nof1_causal_lab.study.view_models import DataDiffReport, DataDiffRequest
    from tests.inference_fixtures import inference_log

    def revision(seq, action, inputs, *produced, status="applied"):
        artifacts = tuple(
            ArtifactRecord(
                artifact_id=artifact_id,
                revision=oid(n),
                derived_from={key: oid(value) for key, value in derived.items()},
            )
            for artifact_id, n, derived in produced
        )
        if action == "edit_model":
            result = Applied(
                result=None,
                effects=ActionEffects(produced=artifacts),
            )
            record = applied_record(result, seq=seq, request=EditModelRequest(expected_revision=inputs["expected_revision"], model=ModelSpec()))
        elif action == "prepare_data":
            record = applied_record(
                Applied(
                    result=DataPreparationResult(),
                    effects=ActionEffects(produced=artifacts),
                ),
                seq=seq,
                request=PrepareDataRequest(input=inputs["input"]),
            )
        elif action == "simulate":
            ref = GitRef(workspace_id="STUDY", revision=inputs["model_revision"], path="model.json")
            report = SimulationReport(causal=NotApplicable(reason="No intervention was requested."), fit_reliability="not_fitted", law=AuthoredLawProvenance(), evidence=SimulationEvidence(model=ref, design=SimulationSpec(start=date(2026, 1, 1), horizon="10d"), time_origin=datetime(2026, 1, 1, tzinfo=UTC), times=(0, 10), draws=1, seed=0, state_ids=(), parameter_draws={}, latent_paths="paths", observations="observations", observation_layout={
                    "variables": [],
                    "support_start_times": "starts",
                    "support_end_times": "ends",
                    "mask": "mask",
                }))
            record = applied_record(
                Applied(result=ModelSimulationResult(evidence=(report).evidence), effects=ActionEffects()),
                seq=seq,
                request=SimulateRequest(
                    model_revision=ref.revision, start=date(2026, 1, 1), horizon="10d"
                ),
            )
        elif action == "fit":
            request = FitRequest(**inputs)
            if status == "raised":
                record = AttemptRecord(
                    seq=seq,
                    ts="2026-10-01T00:00:00Z",
                    attempt=FitAttempt(
                        action="fit",
                        request=request,
                        outcome=Raised(error_type="WorkerError", error_message="failed"),
                    ),
                )
            else:
                from tests.model_fixtures import load_model_fixture
                model = load_model_fixture("causal_proofs/conditioned_treatment_outcome.json")
                result = inference_log(
                    model, prior_revision=inputs["model_revision"], seq=seq
                ).record.attempt.outcome.result
                result = result.revised(
                    panel=GitRef(
                        workspace_id="STUDY",
                        revision=inputs["panel_revision"],
                        path="panel.parquet",
                    )
                )
                record = applied_record(
                    Applied(result=result, effects=ActionEffects(produced=artifacts)),
                    seq=seq,
                    request=request,
                )
        else:
            request = DataDiffRequest(**inputs)
            record = applied_record(
                Applied(
                    result=None,
                    effects=ActionEffects(),
                ),
                seq=seq,
                request=request,
            )
        return StudyRevision(record=record, commit_id=oid(100 + seq), parent_ids=(oid(99 + seq),))

    records = [
        revision(1, "edit_model", {"expected_revision": None}, ("model", 1, {})),
        revision(2, "simulate", {"model_revision": oid(1)}),
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
        revision(8, "edit_model", {"expected_revision": oid(6)}, ("model", 8, {"model": 6})),
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
