"""Finished action reads retain HTTP caching and typed in-process results."""

from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from nof1_causal_lab import episode_api, tool_server
from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import TransitionRecord
from nof1_causal_lab.utils import data
from tests.action_fixtures import edit_and_check
from tests.helpers import run_async

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("status", ["applied", "raised"])
def test_completed_poll_is_typed_for_tools_and_cached_for_http(tmp_path, monkeypatch, status):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    attempt_id = UUID(int=1)
    effects = edit_and_check(
        "POLL",
        EditModelRequest(expected_revision=None, model=ModelSpec(question="Does X affect Y?")),
        EpisodeState(),
    )
    StudyRepository("POLL").append(
        TransitionRecord(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            attempt_id=attempt_id,
            action="edit_model",
            status=status,
            produced=effects.produced if status == "applied" else [],
            checks=effects.checks if status == "applied" else None,
            error_type="ValueError" if status == "raised" else None,
            error_message="failed" if status == "raised" else None,
            trace_ids=[],
            resume=None,
        )
    )
    client = TestClient(tool_server.app)
    first = client.get(f"/api/episodes/POLL/actions/{attempt_id}")
    assert first.status_code == 200
    assert first.json()["done"] is True
    assert (first.json()["body"] is not None) == (status == "applied")

    from nof1_causal_lab.actions import reads

    def unexpected_render(*_args):
        raise AssertionError("A finished attempt should use its cached bytes")

    monkeypatch.setattr(reads, "read_action_body", unexpected_render)
    second = client.get(f"/api/episodes/POLL/actions/{attempt_id}")
    assert second.content == first.content
    typed = run_async(episode_api.read_action_poll("POLL", attempt_id))
    assert typed.done
    assert typed.model_dump(mode="json") == first.json()
    tool = client.post(
        "/api/tools/scientific/poll_action",
        json={"workspace_id": "POLL", "input": {"attempt_id": str(attempt_id)}},
    )
    assert tool.status_code == 200
    assert tool.json() == {"result": first.json()}
