"""Completed applied/rejected/raised attempts have typed, cached HTTP/tool results."""

from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab import study_api, tool_server
from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.results import ActionPoll, CompletedPoll, RunningPoll
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import (
    Applied,
    AttemptRecord,
    EditAttempt,
    ModelEditResult,
    Raised,
    Rejected,
)
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.utils import data
from tests.action_fixtures import edit_and_check
from tests.helpers import run_async

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("status", ["applied", "rejected", "raised"])
def test_completed_poll_is_typed_for_tools_and_cached_for_http(tmp_path, monkeypatch, status):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    attempt_id = UUID(int=1)
    request = EditModelRequest(expected_revision=None, model=ModelSpec(question="Does X affect Y?"))
    outcomes = {
        "applied": Applied(result=edit_and_check("POLL", request, StudyState())),
        "rejected": Rejected(reason="revision_conflict", detail="Selected base changed"),
        "raised": Raised(error_type="WorkerError", error_message="failed"),
    }
    publication = StudyRepository("POLL").append(
        AttemptRecord(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            attempt_id=attempt_id,
            attempt=EditAttempt(request=request, outcome=outcomes[status]),
        )
    )
    client = TestClient(tool_server.app)
    first = client.get(f"/api/studies/POLL/actions/{attempt_id}")
    assert first.status_code == 200
    assert first.json()["kind"] == "completed"
    assert first.json()["commit_id"] == publication.commit_id
    assert first.json()["attempt"]["outcome"]["status"] == status

    def unexpected_render(*_args, **_kwargs):
        raise AssertionError("A completed attempt should use its cached bytes")

    monkeypatch.setattr(study_api, "CompletedPoll", unexpected_render)
    second = client.get(f"/api/studies/POLL/actions/{attempt_id}")
    assert second.content == first.content
    typed = run_async(
        study_api.read_action_poll("POLL", attempt_id, study_api.TemporalClientProvider())
    )
    assert isinstance(typed, CompletedPoll)
    assert typed.model_dump(mode="json") == first.json()
    tool = client.post(
        "/api/tools/scientific/poll_action",
        json={"workspace_id": "POLL", "input": {"attempt_id": str(attempt_id)}},
    )
    assert tool.status_code == 200
    assert tool.json() == {"result": first.json()}


def test_poll_and_action_schema_reject_unrelated_payloads():
    adapter = TypeAdapter(ActionPoll)
    assert adapter.validate_python({"kind": "running"}) == RunningPoll()
    with pytest.raises(ValidationError):
        adapter.validate_python({"kind": "completed", "commit_id": None})
    # A successful edit cannot carry a preparation payload; nor can failure carry a result.
    payload = EditAttempt(request=None, outcome=Applied(result=ModelEditResult())).model_dump(
        mode="json"
    )
    payload["outcome"]["result"]["action"] = "prepare_data"
    with pytest.raises(ValidationError):
        EditAttempt.model_validate(payload)
    payload["outcome"] = {
        "status": "raised",
        "error_type": "Error",
        "error_message": "failed",
        "result": {},
    }
    with pytest.raises(ValidationError):
        EditAttempt.model_validate(payload)
