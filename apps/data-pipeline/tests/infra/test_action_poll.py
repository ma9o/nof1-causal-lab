"""One contract covers saved calls, in-flight deduplication and retryable failures."""

import asyncio
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from nof1_causal_lab import tool_server
from nof1_causal_lab.actions.contracts import (
    EditModelRequest,
    FitRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SetQuestionRequest,
    SimulateRequest,
    call_identity,
)
from nof1_causal_lab.actions.results import ActionPoll, CompletedPoll, RunningPoll
from nof1_causal_lab.actions.temporal import workflow as study_workflow
from nof1_causal_lab.actions.temporal.activities import (
    journal_activity,
    read_inputs_activity,
    set_question_activity,
)
from nof1_causal_lab.actions.temporal.messages import ActionRequest, StudyInit
from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, Rejected
from nof1_causal_lab.study.view_models import DataDiffRequest, ModelDiffRequest, PanelRef
from nof1_causal_lab.utils import data
from tests.git_fixtures import git_oid

if TYPE_CHECKING:
    from collections.abc import Awaitable

pytestmark = pytest.mark.contract


async def _execute_action(
    machine: study_workflow.StudyWorkflow, request: ActionRequest
) -> ActionPoll:
    return await cast(
        "Awaitable[ActionPoll]", study_workflow.StudyWorkflow.execute_action(machine, request)
    )


@pytest.mark.parametrize(
    "action_request",
    [
        SetQuestionRequest(question=QuestionSpec(text="Does exercise improve sleep?")),
        EditModelRequest(expected_revision=None, model=ModelSpec()),
        PrepareDataRequest(input=SimulationReplicateRef(revision=git_oid(1), replicate=0)),
        FitRequest(model_revision=git_oid(1), panel_revision=git_oid(2)),
        SimulateRequest(model_revision=git_oid(1), start=date(2026, 1, 1), horizon="1d"),
        DataDiffRequest(left=PanelRef(revision=git_oid(1)), right=PanelRef(revision=git_oid(2))),
        ModelDiffRequest(before=git_oid(1), after=git_oid(2)),
    ],
    ids=lambda action_request: action_request.action,
)
def test_action_reasoning_round_trips_without_changing_call_identity(action_request):
    intent = "Check the revised assumptions.\nGoal: decide which model to fit."
    explained = action_request.revised(reasoning=intent)
    adapter = TypeAdapter(ScientificActionRequest | DataDiffRequest | ModelDiffRequest)
    assert action_request.reasoning is None
    assert adapter.validate_json(explained.model_dump_json(round_trip=True)) == explained
    assert explained.reasoning == intent
    assert call_identity(action_request) == call_identity(explained)
    assert call_identity(explained.revised(reasoning="A new explanation")) == call_identity(
        action_request
    )


def test_call_contract_reuses_applied_and_running_calls_but_retries_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    monkeypatch.setattr(study_workflow.workflow, "now", lambda: datetime(2026, 10, 4, tzinfo=UTC))
    monkeypatch.setattr(study_workflow.workflow, "upsert_memo", lambda _: None)
    request = SetQuestionRequest(
        question=QuestionSpec(text="Does exercise improve sleep?"),
        reasoning="Define the outcome to guide model construction.",
    )
    repository = StudyRepository("CALLS")

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        executions = 0

        async def execute(name, payload, **_options):
            nonlocal executions
            name = name if isinstance(name, str) else name.__name__
            if name == "read_inputs_activity":
                return await read_inputs_activity(payload)
            if name == "set_question_activity":
                executions += 1
                entered.set()
                await release.wait()
                return await set_question_activity(payload)
            if name == "journal_activity":
                return await journal_activity(payload)
            if name == "collect_completed_runs_activity":
                return None
            raise AssertionError(name)

        monkeypatch.setattr(study_workflow.workflow, "execute_activity", execute)
        machine = study_workflow.StudyWorkflow(StudyInit(workspace_id="CALLS"))
        first = asyncio.create_task(
            _execute_action(machine, ActionRequest(request=request, attempt_id=UUID(int=1)))
        )
        await entered.wait()
        duplicate = await _execute_action(
            machine,
            ActionRequest(
                request=request.revised(reasoning="Polling progress"), attempt_id=UUID(int=2)
            ),
        )
        assert isinstance(duplicate, RunningPoll)
        assert duplicate.attempt_id == UUID(int=1)
        assert duplicate.request.reasoning == request.reasoning
        assert executions == 1
        assert repository.attempts() == []
        release.set()
        applied = await first
        assert isinstance(applied, CompletedPoll)
        assert isinstance(applied.attempt.outcome, Applied)
        assert (
            await _execute_action(machine, ActionRequest(request=request, attempt_id=UUID(int=3)))
            == applied
        )
        assert executions == 1
        assert len(repository.attempts()) == 1
        restarted = study_workflow.StudyWorkflow(StudyInit(workspace_id="CALLS", initial_seq=1))
        assert (
            await _execute_action(
                restarted,
                ActionRequest(request=request.revised(reasoning=None), attempt_id=UUID(int=4)),
            )
            == applied
        )
        assert executions == 1
        assert len(repository.attempts()) == 1
        failure = SetQuestionRequest(
            question=QuestionSpec(text="A different question"), reasoning="Reframe the study goal."
        )
        for attempt_id in (5, 6):
            rejected = await _execute_action(
                restarted, ActionRequest(request=failure, attempt_id=UUID(int=attempt_id))
            )
            assert isinstance(rejected, CompletedPoll)
            assert isinstance(rejected.attempt.outcome, Rejected)
            assert rejected.attempt.request == failure
        assert len(repository.attempts()) == 3
        assert [entry.record.attempt.outcome.status for entry in repository.attempts()] == [
            "applied",
            "rejected",
            "rejected",
        ]

    asyncio.run(scenario())
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    client = TestClient(tool_server.app)
    first = client.post("/api/studies/CALLS/set_question", json=request.model_dump(mode="json"))
    second = client.post(
        "/api/studies/CALLS/set_question",
        json=request.revised(reasoning=None).model_dump(mode="json"),
    )
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert first.json()["attempt"]["request"]["reasoning"] == request.reasoning
    timeline = client.get("/api/studies/CALLS/timeline").json()
    assert timeline["attempts"][0]["record"]["attempt"]["request"]["reasoning"] == request.reasoning
    assert len(repository.attempts()) == 3
    assert (
        client.post(
            "/api/studies/CALLS/set_question", json={"question": {"text": "Unsaved"}}
        ).status_code
        == 403
    )
