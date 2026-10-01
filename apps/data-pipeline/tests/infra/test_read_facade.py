"""Read-only facade: same read endpoints as the full facade, scientific actions return 403."""

import pytest
from fastapi.testclient import TestClient

from nof1_causal_lab.read_facade import create_read_facade_app
from nof1_causal_lab.utils import data as data_module
from tests.git_fixtures import artifact_revision, commit_id, git_oid

pytestmark = pytest.mark.contract


def test_read_facade_serves_reads_and_rejects_actions(monkeypatch, tmp_path):
    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    client = TestClient(create_read_facade_app())

    assert client.get("/api/capabilities").json() == {"actions_enabled": False}

    status = client.get("/api/studies/WS-READONLY")
    assert status.status_code == 200
    body = status.json()
    assert body["seq"] == 0
    assert set(body["actions"]) == {"edit_model", "prepare_data", "fit", "simulate"}

    action = client.post(
        "/api/studies/WS-READONLY/actions",
        json={"action": "prepare_data", "source": "files"},
    )
    assert action.status_code == 403
    upload = client.post(
        "/api/upload",
        data={"workspaceId": "WS-READONLY"},
        files={"file": ("data.csv", b"x,y\n1,2\n", "text/csv")},
    )
    assert upload.status_code == 403


def test_artifact_endpoint_serves_pinned_versions(monkeypatch, tmp_path):
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import AttemptRecord
    from nof1_causal_lab.study.store import ArtifactStore

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    store = ArtifactStore("WS-ART")
    question = store.write_artifact(
        "model",
        derived_from={},
        produced_by=None,
        json_files={"model.json": {"question": "does X cause Y?"}},
    )
    client = TestClient(create_read_facade_app())

    pinned = client.get(
        "/api/studies/WS-ART/artifacts/model",
        params={"revision": artifact_revision("WS-ART", "model", 1)},
    )
    assert pinned.status_code == 200
    body = pinned.json()
    assert body["payload"]["model.json"] == {"question": "does X cause Y?"}
    assert "provenance" not in body["meta"]
    assert body["binary_files"] == []

    # Explicit revision reads can inspect an uncommitted artifact, but it does
    # not become the current revision until an applied action records the effect.
    assert client.get("/api/studies/WS-ART/artifacts/model").status_code == 404
    StudyRepository("WS-ART").append(
        AttemptRecord(
            seq=1,
            ts="2026-07-09T00:00:00+00:00",
            action="edit_model",
            inputs={"expected_revision": None},
            status="applied",
            produced=[question],
            trace_ids=[],
        )
    )
    current = client.get("/api/studies/WS-ART/artifacts/model")
    assert current.status_code == 200
    assert current.json()["payload"]["model.json"] == {"question": "does X cause Y?"}
    missing = client.get("/api/studies/WS-ART/artifacts/model", params={"revision": git_oid(7)})
    assert missing.status_code == 404


def test_trace_endpoints_join_artifact_version_to_promoted_trace(monkeypatch, tmp_path):
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import AttemptRecord
    from nof1_causal_lab.study.store import ArtifactStore, collect_run_traces
    from nof1_causal_lab.utils import storage
    from nof1_causal_lab.utils.llm import LLMTrace, TraceMessage

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("READ_ONLY_FACADE", "1")
    raw_data = ArtifactStore("WS-TRACE").write_artifact(
        "raw_data",
        derived_from={},
        produced_by="prepare_data",
    )
    source = str(tmp_path / "data/WS-TRACE/scratch/runs/seq-000001/llm/raw-data/trace.json")
    storage.write_text(
        source,
        LLMTrace(
            messages=[TraceMessage(role="assistant", content="profiled")],
            model="test-model",
        ).model_dump_json(),
    )
    logs = collect_run_traces("WS-TRACE", 1)
    StudyRepository("WS-TRACE").append(
        AttemptRecord(
            seq=1,
            ts="2026-07-09T00:00:00+00:00",
            action="prepare_data",
            inputs={},
            status="applied",
            produced=[raw_data],
            trace_ids=["raw-data"],
        ),
        logs=logs,
    )
    client = TestClient(create_read_facade_app())

    trace_list = client.get("/api/studies/WS-TRACE/artifacts/raw_data/traces")
    assert trace_list.status_code == 200
    assert trace_list.json()["trace_ids"] == ["raw-data"]
    trace = client.get(f"/api/studies/WS-TRACE/traces/{commit_id('WS-TRACE', 1)}/raw-data")
    assert trace.status_code == 200
    assert trace.json()["messages"][0]["content"] == "profiled"


def test_workspaces_endpoint_lists_study_questions(monkeypatch, tmp_path):
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import AttemptRecord
    from nof1_causal_lab.study.store import ArtifactStore

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.setenv("READ_ONLY_FACADE", "1")

    store = ArtifactStore("WS-LIST")
    question = store.write_artifact(
        "model",
        derived_from={},
        produced_by=None,
        json_files={"model.json": {"question": "does X cause Y?"}},
    )
    StudyRepository("WS-LIST").append(
        AttemptRecord(
            seq=1,
            ts="2026-07-09T00:00:00+00:00",
            action="edit_model",
            inputs={"expected_revision": None},
            status="applied",
            produced=[question],
            trace_ids=[],
        )
    )

    client = TestClient(create_read_facade_app())
    response = client.get("/api/workspaces")

    assert response.status_code == 200
    assert response.json() == {
        "workspaces": [
            {
                "href": "/v2/WS-LIST",
                "question": "does X cause Y?",
                "workspaceId": "WS-LIST",
            }
        ]
    }


def test_upload_endpoint_stages_input_file(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import storage

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    monkeypatch.delenv("READ_ONLY_FACADE", raising=False)
    client = TestClient(create_read_facade_app())

    response = client.post(
        "/api/upload",
        data={"workspaceId": "WS-UPLOAD"},
        files={"file": ("data.csv", b"x,y\n1,2\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json() == {"path": "WS-UPLOAD/input/data.csv"}
    assert storage.read_text(str(tmp_path / "data" / "WS-UPLOAD" / "input" / "data.csv")) == (
        "x,y\n1,2\n"
    )


def test_full_facade_advertises_actions(monkeypatch):
    monkeypatch.delenv("READ_ONLY_FACADE", raising=False)
    from nof1_causal_lab import tool_server

    client = TestClient(tool_server.app)
    assert client.get("/api/capabilities").json() == {"actions_enabled": True}
