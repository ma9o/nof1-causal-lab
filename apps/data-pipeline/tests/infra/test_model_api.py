"""Scientific edits are the sole model mutation contract; history remains inspectable."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from temporalio.api.common.v1 import Payloads
from temporalio.api.failure.v1 import Failure
from temporalio.api.update.v1 import Outcome
from temporalio.client import WorkflowUpdateHandle, WorkflowUpdateStage

from nof1_causal_lab import episode_api
from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.actions.results import ActionMessage, ActionPoll
from nof1_causal_lab.machine.errors import ArtifactWriteRejected
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord, read_current_state
from nof1_causal_lab.read_facade import create_read_facade_app
from nof1_causal_lab.utils import data as data_module
from tests.action_fixtures import edit_and_check
from tests.git_fixtures import artifact_revision, commit_id
from tests.helpers import make_model

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import OperationId


@pytest.fixture
def model_api(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "0")
    calls = []
    pending = {}

    class Handle:
        def __init__(self, workspace):
            self.workspace = workspace

        async def start_update(self, method, envelope, *, id, wait_for_stage):
            assert id == str(envelope.attempt_id)
            assert wait_for_stage == WorkflowUpdateStage.ACCEPTED
            calls.append(envelope)
            pending[envelope.attempt_id] = (self.workspace, envelope)
            return WorkflowUpdateHandle(Mock(), id, f"episode-{self.workspace}")

        async def query(self, method, attempt_id):
            return ActionPoll(done=False) if attempt_id in pending else None

        def complete(self, attempt_id):
            workspace, envelope = pending.pop(UUID(attempt_id))
            assert workspace == self.workspace
            request = envelope.request
            assert isinstance(request, EditModelRequest)
            state = read_current_state(self.workspace, branch=envelope.branch)
            journal = StudyRepository(self.workspace)
            seq = journal.latest_seq() + 1
            try:
                effects = edit_and_check(self.workspace, request, state)
                record = TransitionRecord(
                    seq=seq,
                    attempt_id=envelope.attempt_id,
                    branch=envelope.branch,
                    ts="2026-01-01T00:00:00Z",
                    action=request.action,
                    inputs=request.model_dump(mode="json", exclude={"action"}),
                    status="applied",
                    produced=effects.produced,
                    retracted=effects.retracted,
                    diagnostics=effects.diagnostics,
                    checks=effects.checks,
                    trace_ids=[],
                    resume=None,
                )
            except ArtifactWriteRejected as exc:
                record = TransitionRecord(
                    seq=seq,
                    attempt_id=envelope.attempt_id,
                    branch=envelope.branch,
                    ts="2026-01-01T00:00:00Z",
                    action=request.action,
                    status="rejected",
                    reason=str(exc),
                    trace_ids=[],
                    resume=None,
                )
            timestamp = datetime.now(UTC)
            record = record.model_copy(
                update={
                    "messages": (
                        ActionMessage(
                            timestamp=timestamp, level="info", label="EDIT_MODEL_STARTED"
                        ),
                        *(
                            completion_messages(
                                self.workspace,
                                record.action,
                                record.produced,
                                record.diagnostics,
                                timestamp,
                                checks=record.checks,
                            )
                            if record.status == "applied"
                            else ()
                        ),
                        ActionMessage(
                            timestamp=timestamp,
                            level="info" if record.status == "applied" else "error",
                            label="ACTION_COMPLETED"
                            if record.status == "applied"
                            else "REVISION_CONFLICT",
                        ),
                    )
                }
            )
            journal.append(record)

    async def handle(workspace):
        return Handle(workspace)

    monkeypatch.setattr(episode_api, "_episode_handle", handle)

    class Client:
        def get_workflow_handle(self, workflow_id):
            return Handle(workflow_id.removeprefix("episode-"))

    async def get_client():
        return Client()

    monkeypatch.setattr(episode_api, "_get_client", get_client)
    return TestClient(create_read_facade_app()), calls, Handle("API").complete


@pytest.mark.parametrize(
    "outcome",
    [
        None,
        Outcome(success=Payloads()),
        Outcome(
            failure=Failure(
                message="Failed decoding arguments",
                cause=Failure(message="Cannot access os.environ.setdefault from inside a workflow"),
            )
        ),
    ],
)
def test_action_submission_surfaces_known_rejection_without_waiting(
    model_api, monkeypatch, outcome
):
    client, _, _ = model_api
    repository = StudyRepository("API")
    head = repository.head()
    update = WorkflowUpdateHandle(Mock(), str(uuid4()), "episode-API", known_outcome=outcome)
    update.result = AsyncMock(side_effect=AssertionError("Submission must not wait for completion"))
    handle = Mock(start_update=AsyncMock(return_value=update))
    monkeypatch.setattr(episode_api, "_episode_handle", AsyncMock(return_value=handle))

    response = client.post(
        "/api/episodes/API/actions",
        json={"action": "edit_model", "expected_revision": None, "model": {"question": "Why?"}},
    )

    if outcome is not None and outcome.HasField("failure"):
        assert response.status_code == 500
        assert response.json() == {
            "detail": "Workflow update rejected: Failed decoding arguments: "
            "Cannot access os.environ.setdefault from inside a workflow"
        }
    else:
        assert response.status_code == 202
        assert set(response.json()) == {"attempt_id"}
    update.result.assert_not_awaited()
    assert repository.head() == head
    assert repository.latest_seq() == 0


def test_edit_action_validates_identity_and_base_before_publication(model_api, monkeypatch):
    client, calls, complete = model_api
    url = "/api/episodes/API/actions"
    first = client.post(
        url,
        json={
            "action": "edit_model",
            "expected_revision": None,
            "model": {"question": "  Does X change Y?  "},
        },
    )
    assert first.status_code == 202
    assert set(first.json()) == {"attempt_id"}
    attempt_url = f"{url}/{first.json()['attempt_id']}"
    assert client.get(attempt_url).json() == {"done": False, "body": None, "messages": []}
    assert not read_current_state("API").has("model")
    complete(first.json()["attempt_id"])
    result = client.get(attempt_url).json()
    assert result["done"] is True
    assert result["body"]["action"] == "edit_model"
    assert result["body"]["model"]["question"] == "Does X change Y?"
    assert "MODEL_INCOMPLETE" in {message["label"] for message in result["messages"]}
    assert all(message["timestamp"] for message in result["messages"])
    assert client.get(f"{url}/{uuid4()}").status_code == 404
    initial = client.get("/api/episodes/API/model").json()
    assert initial["context"]["seq"] == 1
    assert initial["model"]["value"]["question"] == "Does X change Y?"
    revision = artifact_revision("API", "model", 1)
    stale = client.post(
        url,
        json={
            "action": "edit_model",
            "expected_revision": None,
            "model": {"question": "A conflicting edit"},
        },
    )
    assert stale.status_code == 202
    complete(stale.json()["attempt_id"])
    failed = client.get(f"{url}/{stale.json()['attempt_id']}").json()
    assert failed["done"] is True
    assert failed["body"] is None
    assert failed["messages"][-1]["level"] == "error"
    assert (
        client.get("/api/episodes/API/model").json()["context"]["commit_id"]
        == initial["context"]["commit_id"]
    )
    invalid = client.post(
        url,
        json={
            "action": "edit_model",
            "expected_revision": revision,
            "model": {"default_outcome": "construct:absent"},
        },
    )
    assert invalid.status_code == 422
    assert len(calls) == 2
    assert ArtifactStore("API").list_revisions("model") == [revision]
    model = make_model(["X", "Y"], [("X", "Y")]).revised(question="Does X change Y?")
    second = client.post(
        url,
        json={
            "action": "edit_model",
            "expected_revision": revision,
            "model": model.model_dump(mode="json"),
        },
    )
    assert second.status_code == 202
    complete(second.json()["attempt_id"])
    assert client.get("/api/episodes/API/model").json()["model"]["value"] == model.model_dump(
        mode="json"
    )
    assert (
        client.get(
            "/api/episodes/API/model", params={"at": initial["context"]["commit_id"]}
        ).json()["model"]["value"]
        == initial["model"]["value"]
    )
    records = client.get("/api/episodes/API/timeline").json()["transitions"]
    assert [entry["action"] for entry in records] == ["edit_model"] * 3
    assert all("move" not in entry for entry in records)
    assert all("provenance" not in item for entry in records for item in entry["produced"])
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "1")
    assert client.get(attempt_url).json() == result
    assert client.get(f"{url}/{uuid4()}").status_code == 404


def test_only_four_scientific_actions_are_exposed(model_api):
    client, calls, _ = model_api
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/episodes/{workspace_id}/moves" not in paths
    assert "/api/episodes/{workspace_id}/recipes/observational-study" not in paths
    assert "put" not in paths["/api/episodes/{workspace_id}/model"]
    assert "/api/episodes" not in paths
    body = {"action": "edit_model", "expected_revision": None, "model": {}}
    assert (
        client.post("/api/episodes/API/actions", json={**body, "provenance": "human"}).status_code
        == 422
    )
    assert (
        client.post("/api/episodes/API/actions", json={"action": "latent_structure"}).status_code
        == 422
    )
    assert not calls
    assert "Provenance" not in schema["components"]["schemas"]
    assert "Move" not in schema["components"]["schemas"]


def test_operation_trace_index_retains_recorded_authoring_jobs(model_api):
    client, _, _ = model_api
    store, journal = ArtifactStore("TRACES"), StudyRepository("TRACES")
    operations: list[OperationId | None] = ["latent_structure", "measurement_structure", None]
    for seq, operation in enumerate(operations, 1):
        info = store.write_artifact(
            "model",
            derived_from={},
            produced_by="edit_model",
            json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
        )
        journal.append(
            TransitionRecord(
                seq=seq,
                ts="2026-01-01T00:00:00Z",
                action="edit_model",
                operation_id=operation,
                status="applied",
                produced=[info],
                trace_ids=[f"trace-{seq}"],
                resume=None,
            )
        )
    for seq, operation in enumerate(operations[:2], 1):
        response = client.get(f"/api/episodes/TRACES/operations/{operation}/traces")
        assert response.status_code == 200
        assert response.json()["commit_id"] == commit_id("TRACES", seq)
    latest = client.get("/api/episodes/TRACES/artifacts/model/traces")
    assert latest.json()["commit_id"] == commit_id("TRACES", 3)
