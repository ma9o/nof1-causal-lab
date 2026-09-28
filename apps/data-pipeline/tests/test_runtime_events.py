from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from nof1_causal_lab.actions.results import ActionMessage
from nof1_causal_lab.flows.runtime_events import (
    ActionMessageEvent,
    ExtractionPlanEvent,
    ExtractionSnapshotEvent,
    ExtractionWorkerEvent,
    ModelSpecAdmissionEvent,
    TransitionRuntimeEvent,
    emit_action_message,
    emit_event,
    emit_extraction_plan_event,
    emit_extraction_snapshot_event,
    emit_extraction_worker_event,
    emit_transition_event,
    read_events,
)

pytestmark = pytest.mark.contract


def test_runtime_events_roundtrip_as_discriminated_models(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    workspace_id = "runtime-event-contracts"

    emit_transition_event(
        workspace_id,
        "statistical_model_spec",
        "failed",
        error={"type": "ValidationError", "message": "invalid prior"},
    )
    emit_event(
        workspace_id,
        ModelSpecAdmissionEvent(
            event="nof1-causal-lab.model-spec.admission.barrier_report",
            payload={"context_id": "statistical-model-spec", "round": 2, "status": "ready"},
        ),
    )

    emit_extraction_plan_event(workspace_id, total_workers=2, max_concurrent_workers=1, max_rpm=30)
    emit_extraction_worker_event(
        workspace_id, worker_id=0, state="completed", n_windows=3, n_extractions=4
    )
    emit_extraction_snapshot_event(
        workspace_id,
        snapshot={
            "total_workers": 2,
            "pending_workers": 1,
            "running_workers": 0,
            "completed_workers": 1,
            "failed_workers": 0,
            "llm_requests_last_60s": 1,
        },
    )
    events = read_events(workspace_id)
    assert len(events) == 5
    assert isinstance(events[0], TransitionRuntimeEvent)
    assert events[0].error is not None
    assert events[0].error.type == "ValidationError"
    assert isinstance(events[1], ModelSpecAdmissionEvent)
    assert events[1].payload["context_id"] == "statistical-model-spec"
    assert events[1].event == "nof1-causal-lab.model-spec.admission.barrier_report"

    for event in (events[0], *events[2:]):
        payload = event.model_dump()
        assert not {"payload", "type", "context_id", "status"} & payload.keys()
    assert isinstance(events[2], ExtractionPlanEvent)
    assert isinstance(events[3], ExtractionWorkerEvent)
    assert isinstance(events[4], ExtractionSnapshotEvent)
    assert events[2].total_workers == 2
    assert events[3].n_extractions == 4
    assert events[4].completed_workers == 1

    label = ActionMessageEvent(
        attempt_id=uuid4(),
        action="edit_model",
        index=0,
        message=ActionMessage(timestamp=datetime.now(UTC), level="warn", label="MODEL_INCOMPLETE"),
    )
    emit_action_message(label, workspace_id)
    emit_action_message(label, workspace_id)
    emitted = read_events(workspace_id, after=events[-1].cursor)
    assert len(emitted) == 1  # Retrying an activity does not duplicate a label.
    assert isinstance(emitted[0], ActionMessageEvent)
    assert emitted[0].message == label.message
    assert emitted[0].attempt_id == label.attempt_id
