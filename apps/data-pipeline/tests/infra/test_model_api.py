"""HTTP model writes share machine validation; operation traces survive later authorship."""

import pytest
from fastapi.testclient import TestClient

from nof1_causal_lab import episode_api
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.moves import RunOperation, WriteArtifact, validate_move
from nof1_causal_lab.machine.store import (
    ArtifactStore,
    EpisodeJournal,
    TransitionRecord,
    derive_current_state,
)
from nof1_causal_lab.machine.writes import execute_write
from nof1_causal_lab.read_facade import create_read_facade_app
from nof1_causal_lab.utils import data as data_module
from tests.helpers import make_model


@pytest.fixture
def model_api(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "0")
    calls = []

    async def local_propose(workspace, body):
        calls.append(body)
        state = derive_current_state(workspace)
        reason = validate_move(state, body.move)
        if reason is not None:
            return {"status": "rejected", "reason": reason}
        effects = execute_write(
            workspace,
            "model",
            body.payload,
            body.move.provenance,
            state,
            expected_model_version=body.move.expected_model_version,
        )
        journal = EpisodeJournal(workspace)
        seq = journal.latest_seq() + 1
        journal.append(
            TransitionRecord(
                seq=seq,
                ts="2026-01-01T00:00:00+00:00",
                move=body.move,
                status="applied",
                produced=effects.produced,
                retracted=effects.retracted,
                trace_ids=[],
                resume=None,
            )
        )
        return {
            "status": "applied",
            "seq": seq,
            "state": derive_current_state(workspace).model_dump(mode="json"),
        }

    monkeypatch.setattr(episode_api, "_propose", local_propose)
    return TestClient(create_read_facade_app()), calls


def test_model_put_validates_identity_and_expected_revision_before_publication(model_api):
    client, calls = model_api
    url = "/api/episodes/API/model"
    first = client.put(
        url,
        json={
            "expected_version": 0,
            "model": {"question": "  Does X change Y?  "},
        },
    )
    assert first.status_code == 200
    assert first.json()["model"]["source"]["ref"]["version"] == 1
    assert first.json()["context"]["seq"] == 1
    initial = first.json()["model"]["value"]
    assert initial["question"] == "Does X change Y?"
    assert initial["edges"] == []
    stale = client.put(
        url,
        json={
            "expected_version": 0,
            "model": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json"),
        },
    )
    assert stale.status_code == 409
    invalid = client.put(
        url,
        json={
            "expected_version": 1,
            "model": {
                **make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json"),
                "default_outcome": "construct:absent",
            },
        },
    )
    assert invalid.status_code == 422
    assert len(calls) == 2
    assert ArtifactStore("API").list_versions("model") == [1]
    enriched = make_model(["X", "Y"], [("X", "Y")]).revised(question=initial["question"])
    second = client.put(
        url,
        json={
            "expected_version": 1,
            "model": enriched.model_dump(mode="json"),
        },
    )
    assert second.status_code == 200
    assert second.json()["model"]["source"]["ref"]["version"] == 2
    assert client.get(url, params={"at_seq": 1}).json()["model"]["source"]["ref"]["version"] == 1
    assert ModelSpec.model_validate(second.json()["model"]["value"]) == enriched
    assert client.get(url, params={"at_seq": 1}).json()["model"]["value"] == initial


def test_operation_trace_index_retains_each_authoring_operation(model_api):
    client, _ = model_api
    store, journal = ArtifactStore("TRACES"), EpisodeJournal("TRACES")
    moves = [
        RunOperation(operation_id="latent_structure"),
        RunOperation(operation_id="measurement_structure"),
        WriteArtifact(artifact_id="model", expected_model_version=2),
    ]
    for seq, move in enumerate(moves, 1):
        info = store.write_version(
            "model",
            provenance="computed" if seq < 3 else "human",
            derived_from={},
            produced_by="test",
            json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
        )
        journal.append(
            TransitionRecord(
                seq=seq,
                ts="2026-01-01T00:00:00+00:00",
                move=move,
                status="applied",
                produced=[info],
                trace_ids=[f"trace-{seq}"],
                resume=None,
            )
        )
    for seq, operation in enumerate(("latent_structure", "measurement_structure"), 1):
        response = client.get(f"/api/episodes/TRACES/operations/{operation}/traces")
        assert response.status_code == 200
        assert response.json()["seq"] == seq
        assert response.json()["trace_ids"] == [f"trace-{seq}"]
    latest = client.get("/api/episodes/TRACES/artifacts/model/traces")
    assert latest.status_code == 200
    assert latest.json()["seq"] == 3

    journal.append(
        TransitionRecord(
            seq=4,
            ts="2026-01-01T00:00:00Z",
            move=RunOperation(operation_id="measurements"),
            status="applied",
            produced=[],
            trace_ids=["empty-extraction"],
            resume=None,
        )
    )
    empty = client.get("/api/episodes/TRACES/operations/measurements/traces")
    assert empty.status_code == 200
    assert empty.json() == {"workspace_id": "TRACES", "seq": 4, "trace_ids": ["empty-extraction"]}


def test_episode_question_creates_and_revises_the_model(model_api, monkeypatch):
    from unittest.mock import AsyncMock

    client, calls = model_api
    monkeypatch.setattr(episode_api, "_episode_handle", AsyncMock())
    response = client.post(
        "/api/episodes", json={"workspace_id": "QUESTION", "question": "Does X change Y?"}
    )
    assert response.status_code == 200
    assert calls[-1].move == WriteArtifact(artifact_id="model", expected_model_version=0)
    initial = client.get("/api/episodes/QUESTION/model").json()
    assert initial["model"]["value"]["question"] == "Does X change Y?"
    assert initial["model"]["value"]["edges"] == []
    graph = make_model(["X", "Y"], [("X", "Y")]).revised(question="Does X change Y?")
    assert (
        client.put(
            "/api/episodes/QUESTION/model",
            json={"expected_version": 1, "model": graph.model_dump(mode="json")},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/episodes", json={"workspace_id": "QUESTION", "question": "How does X change Y?"}
        ).status_code
        == 200
    )
    revised = client.get("/api/episodes/QUESTION/model").json()["model"]
    assert calls[-1].move.expected_model_version == 2
    assert revised["source"]["ref"]["version"] == 3
    assert revised["value"]["question"] == "How does X change Y?"
    assert revised["value"]["edges"] == graph.model_dump(mode="json")["edges"]
    assert client.get("/api/episodes/QUESTION/model?at_seq=1").json() == initial
