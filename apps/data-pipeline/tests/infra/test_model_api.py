"""Scientific edits are the sole model mutation contract; history remains inspectable."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from temporalio.api.common.v1 import Payloads
from temporalio.api.failure.v1 import Failure
from temporalio.api.update.v1 import Outcome
from temporalio.client import WorkflowUpdateHandle, WorkflowUpdateStage

from nof1_causal_lab import study_api
from nof1_causal_lab.actions.contracts import EditModelRequest, SetQuestionRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.actions.results import RunningPoll
from nof1_causal_lab.actions.set_question import set_question
from nof1_causal_lab.read_facade import create_read_facade_app
from nof1_causal_lab.study.errors import ArtifactWriteRejected
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import (
    ActionMessage,
    Applied,
    AttemptRecord,
    DataComparisonResult,
    EditAttempt,
    Rejected,
)
from nof1_causal_lab.study.store import ArtifactStore, read_current_state
from nof1_causal_lab.study.view_models import PanelRef
from nof1_causal_lab.utils import data as data_module
from tests.action_fixtures import applied_record, edit_and_check
from tests.git_fixtures import artifact_revision, commit_id
from tests.helpers import make_model

pytestmark = pytest.mark.contract


def test_facade_clients_share_connections_only_within_their_application(monkeypatch):
    from nof1_causal_lab.actions.temporal import client as temporal_client

    first_connection, second_connection = object(), object()
    connect = AsyncMock(side_effect=[RuntimeError("offline"), first_connection, second_connection])
    monkeypatch.setattr(temporal_client, "connect_client", connect)
    first_app, second_app = create_read_facade_app(), create_read_facade_app()

    async def scenario():
        with pytest.raises(RuntimeError, match="offline"):
            await first_app.state.study_clients.get()
        connections = await asyncio.gather(
            first_app.state.study_clients.get(), first_app.state.study_clients.get()
        )
        assert all(connection is first_connection for connection in connections)
        assert await second_app.state.study_clients.get() is second_connection

    asyncio.run(scenario())
    assert connect.await_count == 3


@pytest.fixture
def model_api(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("READ_ONLY_FACADE", "0")
    calls = []
    pending = {}

    class Handle:
        def __init__(self, workspace):
            self.workspace = workspace

        async def start_update(self, method, envelope, *, id, wait_for_stage, result_type):  # noqa: A002 -- Temporal's SDK names this keyword id.
            assert id == str(envelope.attempt_id)
            assert wait_for_stage == WorkflowUpdateStage.ACCEPTED
            calls.append(envelope)
            pending[envelope.attempt_id] = (self.workspace, envelope)
            return WorkflowUpdateHandle(Mock(), id, f"study-{self.workspace}")

        async def query(self, method, attempt_id):
            return RunningPoll() if attempt_id in pending else None

        def complete(self, attempt_id):
            workspace, envelope = pending.pop(UUID(attempt_id))
            assert workspace == self.workspace
            request = envelope.request
            assert isinstance(request, (SetQuestionRequest, EditModelRequest))
            state = read_current_state(self.workspace, branch=envelope.branch)
            journal = StudyRepository(self.workspace)
            seq = journal.latest_seq() + 1
            try:
                effects = (
                    set_question(self.workspace, request)
                    if isinstance(request, SetQuestionRequest)
                    else edit_and_check(self.workspace, request, state)
                )
                record = applied_record(
                    effects,
                    request=request,
                    seq=seq,
                    attempt_id=envelope.attempt_id,
                    branch=envelope.branch,
                    ts="2026-01-01T00:00:00Z",
                )
            except ArtifactWriteRejected as exc:
                record = AttemptRecord(
                    seq=seq,
                    attempt_id=envelope.attempt_id,
                    branch=envelope.branch,
                    ts="2026-01-01T00:00:00Z",
                    attempt=EditAttempt(
                        action="edit_model",
                        request=request,
                        outcome=Rejected(reason="revision_conflict", detail=str(exc)),
                    ),
                )
            timestamp = datetime.now(UTC)
            outcome = record.attempt.outcome
            record = record.with_logs(
                trace_ids=(),
                messages=(
                    ActionMessage(timestamp=timestamp, level="info", label="EDIT_MODEL_STARTED"),
                    *(
                        completion_messages(
                            outcome,
                            timestamp,
                            ArtifactStore(self.workspace).completion_reports(
                                outcome.effects.produced
                            ),
                        )
                        if isinstance(outcome, Applied)
                        else ()
                    ),
                    ActionMessage(
                        timestamp=timestamp,
                        level="info" if isinstance(outcome, Applied) else "error",
                        label="ACTION_COMPLETED"
                        if isinstance(outcome, Applied)
                        else "REVISION_CONFLICT",
                    ),
                ),
            )
            journal.append(record)

    async def handle(workspace, clients):
        return Handle(workspace)

    monkeypatch.setattr(study_api, "_study_handle", handle)

    class Client:
        def get_workflow_handle(self, workflow_id):
            return Handle(workflow_id.removeprefix("study-"))

    async def get_client(self):
        return Client()

    monkeypatch.setattr(study_api.TemporalClientProvider, "get", get_client)
    return TestClient(create_read_facade_app()), calls, Handle("API").complete


def test_data_diff_dispatch_captures_head_and_saved_report_is_a_read(model_api, monkeypatch):
    from nof1_causal_lab.study.view_models import DataDiffReport

    client, calls, _ = model_api
    repository = StudyRepository("API")
    head = repository.head()
    source = PanelRef(revision=head)
    response = client.post(
        "/api/studies/API/data-diff",
        json={"left": source.model_dump(), "right": source.model_dump()},
    )
    assert response.status_code == 202
    assert calls[-1].request.action == "data_diff"
    assert calls[-1].expected_head == head
    assert repository.head() == head
    report = DataDiffReport(left=(source,), right=(source,), variables=())
    leaf = repository.append(
        applied_record(
            Applied(result=DataComparisonResult(report=report), effects=ActionEffects()),
            seq=1,
            attempt_id=UUID(response.json()["attempt_id"]),
            ts="2026-09-30T00:00:00Z",
        ),
        expected_head=head,
    ).commit_id
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    assert client.get(f"/api/studies/API/data-diff/{leaf}").json() == report.model_dump(mode="json")
    poll = client.get(f"/api/studies/API/actions/{response.json()['attempt_id']}").json()
    assert poll["commit_id"] == leaf
    assert poll["attempt"]["outcome"]["result"]["report"] == report.model_dump(mode="json")
    assert (
        client.post(
            "/api/studies/API/data-diff",
            json={"left": source.model_dump(), "right": source.model_dump()},
        ).status_code
        == 403
    )


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
    update = WorkflowUpdateHandle(Mock(), str(uuid4()), "study-API", known_outcome=outcome)
    update.result = AsyncMock(side_effect=AssertionError("Submission must not wait for completion"))
    handle = Mock(start_update=AsyncMock(return_value=update))
    monkeypatch.setattr(study_api, "_study_handle", AsyncMock(return_value=handle))

    response = client.post(
        "/api/studies/API/actions",
        json={"action": "set_question", "question": {"text": "Why?"}},
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
    url = "/api/studies/API/actions"
    question = client.post(
        url, json={"action": "set_question", "question": {"text": "  Does X change Y?  "}}
    )
    assert question.status_code == 202
    complete(question.json()["attempt_id"])
    first = client.post(url, json={"action": "edit_model", "expected_revision": None, "model": {}})
    assert first.status_code == 202
    assert set(first.json()) == {"attempt_id"}
    attempt_url = f"{url}/{first.json()['attempt_id']}"
    assert client.get(attempt_url).json() == {"kind": "running", "messages": []}
    assert not read_current_state("API").has("model")
    complete(first.json()["attempt_id"])
    result = client.get(attempt_url).json()
    assert result["kind"] == "completed"
    assert result["attempt"]["action"] == "edit_model"
    assert result["attempt"]["outcome"]["effects"]["produced"][0]["artifact_id"] == "model"
    assert "MODEL_INCOMPLETE" in {message["label"] for message in result["messages"]}
    assert all(message["timestamp"] for message in result["messages"])
    assert client.get(f"{url}/{uuid4()}").status_code == 404
    initial = client.get("/api/studies/API/model").json()
    assert initial["selected_seq"] == 2
    assert initial["question"]["value"]["text"] == "Does X change Y?"
    revision = artifact_revision("API", "model", 1)
    stale = client.post(
        url,
        json={
            "action": "edit_model",
            "expected_revision": None,
            "model": {"measurement_clock": "2d"},
        },
    )
    assert stale.status_code == 202
    complete(stale.json()["attempt_id"])
    failed = client.get(f"{url}/{stale.json()['attempt_id']}").json()
    assert failed["kind"] == "completed"
    assert failed["attempt"]["outcome"]["status"] == "rejected"
    assert failed["attempt"]["outcome"]["reason"] == "revision_conflict"
    assert failed["messages"][-1]["level"] == "error"
    assert client.get("/api/studies/API/model").json()["commit_id"] == initial["commit_id"]
    invalid = client.post(
        url,
        json={
            "action": "edit_model",
            "expected_revision": revision,
            "model": {"question": "The question belongs to set_question"},
        },
    )
    assert invalid.status_code == 422
    assert len(calls) == 3
    assert ArtifactStore("API").list_revisions("model") == [revision]
    model = make_model(["X", "Y"], [("X", "Y")])
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
    assert client.get("/api/studies/API/model").json()["model"]["value"] == model.model_dump(
        mode="json"
    )
    assert (
        client.get("/api/studies/API/model", params={"at": initial["commit_id"]}).json()["model"][
            "value"
        ]
        == initial["model"]["value"]
    )
    records = client.get("/api/studies/API/timeline").json()["attempts"]
    assert [entry["record"]["attempt"]["action"] for entry in records] == [
        "set_question",
        *["edit_model"] * 3,
    ]
    assert all("move" not in entry for entry in records)
    assert all(
        "provenance" not in item
        for entry in records
        for item in (entry["record"]["attempt"]["outcome"].get("effects", {}).get("produced", []))
    )
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    assert client.get(attempt_url).json() == result
    assert client.get(f"{url}/{uuid4()}").status_code == 404


def test_only_five_scientific_actions_are_exposed(model_api):
    client, calls, _ = model_api
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert "/api/studies/{workspace_id}/moves" not in paths
    assert "/api/studies/{workspace_id}/recipes/observational-study" not in paths
    assert "put" not in paths["/api/studies/{workspace_id}/model"]
    assert "/api/studies" not in paths
    body = {"action": "edit_model", "expected_revision": None, "model": {}}
    assert (
        client.post("/api/studies/API/actions", json={**body, "provenance": "human"}).status_code
        == 422
    )
    assert (
        client.post("/api/studies/API/actions", json={"action": "latent_structure"}).status_code
        == 422
    )
    assert not calls
    assert "Provenance" not in schema["components"]["schemas"]
    assert "Move" not in schema["components"]["schemas"]


def test_artifact_trace_index_follows_the_current_producer(model_api):
    client, _, _ = model_api
    store, journal = ArtifactStore("TRACES"), StudyRepository("TRACES")
    for seq in (1, 2, 3):
        info = store.write_artifact(
            "model",
            derived_from={},
            produced_by="edit_model",
            json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
        )
        journal.append(
            applied_record(
                Applied(result=None, effects=ActionEffects(produced=[info])),
                seq=seq,
                ts="2026-01-01T00:00:00Z",
                trace_ids=[f"trace-{seq}"],
            )
        )
    latest = client.get("/api/studies/TRACES/artifacts/model/traces")
    assert latest.json()["commit_id"] == commit_id("TRACES", 3)
