"""One contract covers saved calls, in-flight deduplication and retryable failures."""

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from nof1_causal_lab import tool_server
from nof1_causal_lab.actions.contracts import SetQuestionRequest
from nof1_causal_lab.actions.results import CompletedPoll, RunningPoll
from nof1_causal_lab.actions.temporal import workflow as study_workflow
from nof1_causal_lab.actions.temporal.activities import (
    journal_activity,
    read_inputs_activity,
    set_question_activity,
)
from nof1_causal_lab.actions.temporal.messages import ActionRequest, StudyInit
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, Rejected
from nof1_causal_lab.utils import data

pytestmark = pytest.mark.contract


def test_call_contract_reuses_applied_and_running_calls_but_retries_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    monkeypatch.setattr(study_workflow.workflow, "now", lambda: datetime(2026, 10, 4, tzinfo=UTC))
    monkeypatch.setattr(study_workflow.workflow, "upsert_memo", lambda _: None)
    request = SetQuestionRequest(question=QuestionSpec(text="Does exercise improve sleep?"))
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
        first = asyncio.create_task(machine.execute_action(ActionRequest(request=request, attempt_id=UUID(int=1))))
        await entered.wait()
        duplicate = await machine.execute_action(ActionRequest(request=request, attempt_id=UUID(int=2)))
        assert isinstance(duplicate, RunningPoll)
        assert duplicate.attempt_id == UUID(int=1)
        assert executions == 1 and repository.attempts() == []
        release.set()
        applied = await first
        assert isinstance(applied, CompletedPoll) and isinstance(applied.attempt.outcome, Applied)
        assert await machine.execute_action(ActionRequest(request=request, attempt_id=UUID(int=3))) == applied
        assert executions == 1 and len(repository.attempts()) == 1
        restarted = study_workflow.StudyWorkflow(StudyInit(workspace_id="CALLS", initial_seq=1))
        assert await restarted.execute_action(ActionRequest(request=request, attempt_id=UUID(int=4))) == applied
        assert executions == 1 and len(repository.attempts()) == 1
        failure = SetQuestionRequest(question=QuestionSpec(text="A different question"))
        for attempt_id in (5, 6):
            rejected = await restarted.execute_action(ActionRequest(request=failure, attempt_id=UUID(int=attempt_id)))
            assert isinstance(rejected, CompletedPoll) and isinstance(rejected.attempt.outcome, Rejected)
        assert len(repository.attempts()) == 3
        assert [entry.record.attempt.outcome.status for entry in repository.attempts()] == ["applied", "rejected", "rejected"]

    asyncio.run(scenario())
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    client = TestClient(tool_server.app)
    first = client.post("/api/studies/CALLS/set_question", json=request.model_dump(mode="json"))
    second = client.post("/api/studies/CALLS/set_question", json=request.model_dump(mode="json"))
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert len(repository.attempts()) == 3
    assert client.post("/api/studies/CALLS/set_question", json={"question": {"text": "Unsaved"}}).status_code == 403
