"""Transition and delegated-context telemetry events, persisted for UI polling.

Worker fan-out progress and action messages are live telemetry
the web UI renders while a transition runs. Events land as one JSON file each under
``data/{workspace_id}/scratch/events/`` (neither local fs nor R2 supports
atomic append); consumers list the directory and sort by filename, which is
time-ordered by construction. Events are transport, not a read model: nothing
may reconstruct state from them, and the sweep truncates the stream freely.

``workspace_id`` is the stream key — one episode per workspace.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from nof1_causal_lab.actions.results import ActionMessage  # noqa: TC001
from nof1_causal_lab.artifacts.identity import ScientificActionId  # noqa: TC001
from nof1_causal_lab.json_types import JsonObject  # noqa: TC001
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

EXTRACTION_EVENT_PREFIX = "nof1-causal-lab.extraction"


class RuntimeEventModel(BaseModel):
    """Immutable JSON event record, with a cursor added only on reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cursor: str = ""


class RuntimeEventError(BaseModel):
    """Serialized transition failure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    message: str


class ActionMessageEvent(RuntimeEventModel):
    """A label emitted by one dispatched action, with replay ordering outside the message."""

    event: Literal["nof1-causal-lab.action.message"] = "nof1-causal-lab.action.message"
    attempt_id: uuid.UUID
    action: ScientificActionId
    index: int
    message: ActionMessage


class TransitionRuntimeEvent(RuntimeEventModel):
    """One transition lifecycle event, identified by its event name."""

    event: Literal[
        "nof1-causal-lab.transition.running",
        "nof1-causal-lab.transition.completed",
        "nof1-causal-lab.transition.failed",
    ]
    transition_id: str
    error: RuntimeEventError | None = None


class ExtractionPlanEvent(RuntimeEventModel):
    """Static extraction fan-out plan."""

    event: Literal["nof1-causal-lab.extraction.plan"]
    total_workers: int = Field(ge=0)
    max_concurrent_workers: int | None = Field(default=None, gt=0)
    max_rpm: int | None = Field(default=None, gt=0)


class ExtractionWorkerEvent(RuntimeEventModel):
    """One extraction worker state transition."""

    event: Literal["nof1-causal-lab.extraction.worker"]
    worker_id: int = Field(ge=0)
    state: Literal["pending", "running", "completed", "failed"]
    n_windows: int = Field(ge=0)
    n_extractions: int | None = Field(default=None, ge=0)
    n_llm_calls: int | None = Field(default=None, ge=0)
    error: str | None = None


class ExtractionSnapshotEvent(RuntimeEventModel):
    """Aggregate extraction progress snapshot."""

    event: Literal["nof1-causal-lab.extraction.snapshot"]
    total_workers: int = Field(ge=0)
    pending_workers: int = Field(ge=0)
    running_workers: int = Field(ge=0)
    completed_workers: int = Field(ge=0)
    failed_workers: int = Field(ge=0)
    llm_requests_last_60s: int = Field(ge=0)


class ModelSpecAdmissionEvent(RuntimeEventModel):
    """Recorded construct-admission event from an earlier study history."""

    event: Literal[
        "nof1-causal-lab.model-spec.admission.plan",
        "nof1-causal-lab.model-spec.admission.resumed",
        "nof1-causal-lab.model-spec.admission.construct_started",
        "nof1-causal-lab.model-spec.admission.construct_checking",
        "nof1-causal-lab.model-spec.admission.construct_report",
        "nof1-causal-lab.model-spec.admission.barrier_report",
        "nof1-causal-lab.model-spec.admission.done",
        "nof1-causal-lab.model-spec.admission.failed",
    ]
    payload: JsonObject

    @field_validator("payload")
    @classmethod
    def validate_context(cls, payload: JsonObject) -> JsonObject:
        if payload.get("context_id") != "statistical-model-spec":
            raise ValueError("model-spec admission events require context_id")
        return payload


type RuntimeEvent = Annotated[
    ActionMessageEvent
    | TransitionRuntimeEvent
    | ExtractionPlanEvent
    | ExtractionWorkerEvent
    | ExtractionSnapshotEvent
    | ModelSpecAdmissionEvent,
    Field(discriminator="event"),
]

_RUNTIME_EVENT_ADAPTER = TypeAdapter(RuntimeEvent)


def emit_action_message(event: ActionMessageEvent, workspace_id: str) -> None:
    """Activity retries overwrite the same event, preserving message identity and order."""
    directory = events_dir(workspace_id)
    storage.makedirs(directory)
    nanos = int(event.message.timestamp.timestamp() * 1_000_000_000)
    name = f"{nanos:020d}-{event.attempt_id.hex}-{event.index:06d}.json"
    storage.write_text(
        storage.join(directory, name),
        event.model_dump_json(exclude={"cursor"}),
    )


def events_dir(workspace_id: str) -> str:
    return data_module.scratch_events_dir(workspace_id)


def emit_event(workspace_id: str, event: RuntimeEvent) -> None:
    directory = events_dir(workspace_id)
    storage.makedirs(directory)
    name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.json"
    storage.write_text(
        storage.join(directory, name),
        json.dumps(event.model_dump(mode="json", exclude_none=True, exclude={"cursor"})),
    )


def read_events(workspace_id: str, *, after: str | None = None) -> list[RuntimeEvent]:
    """Events in emission order; ``after`` is the last seen filename cursor."""
    directory = events_dir(workspace_id)
    if not storage.exists(directory):
        return []
    entries = sorted(e for e in storage.listdir(directory) if e.endswith(".json"))
    events: list[RuntimeEvent] = []
    for entry in entries:
        cursor = entry.rsplit("/", 1)[-1]
        if after is not None and cursor <= after:
            continue
        record = storage.read_json(entry)
        events.append(_RUNTIME_EVENT_ADAPTER.validate_python({**record, "cursor": cursor}))
    return events


def emit_transition_event(
    workspace_id: str,
    transition_id: str,
    status: Literal["running", "completed", "failed"],
    *,
    error: JsonObject | None = None,
) -> None:
    event_type: Literal[
        "nof1-causal-lab.transition.running",
        "nof1-causal-lab.transition.completed",
        "nof1-causal-lab.transition.failed",
    ]
    if status == "running":
        event_type = "nof1-causal-lab.transition.running"
    elif status == "completed":
        event_type = "nof1-causal-lab.transition.completed"
    else:
        event_type = "nof1-causal-lab.transition.failed"
    emit_event(
        workspace_id,
        TransitionRuntimeEvent(
            event=event_type,
            transition_id=transition_id,
            error=RuntimeEventError.model_validate(error) if error is not None else None,
        ),
    )


def emit_extraction_plan_event(
    workspace_id: str,
    *,
    total_workers: int,
    max_concurrent_workers: int | None,
    max_rpm: int | None,
) -> None:
    """Emit the static extraction execution plan for replay/bootstrap."""
    emit_event(
        workspace_id,
        ExtractionPlanEvent(
            event=f"{EXTRACTION_EVENT_PREFIX}.plan",
            total_workers=total_workers,
            max_concurrent_workers=max_concurrent_workers,
            max_rpm=max_rpm,
        ),
    )


def emit_extraction_worker_event(
    workspace_id: str,
    *,
    worker_id: int,
    state: Literal["pending", "running", "completed", "failed"],
    n_windows: int,
    n_extractions: int | None = None,
    n_llm_calls: int | None = None,
    error: str | None = None,
) -> None:
    """Emit an extraction worker state transition."""
    emit_event(
        workspace_id,
        ExtractionWorkerEvent(
            event=f"{EXTRACTION_EVENT_PREFIX}.worker",
            worker_id=worker_id,
            state=state,
            n_windows=n_windows,
            n_extractions=n_extractions,
            n_llm_calls=n_llm_calls,
            error=error,
        ),
    )


def emit_extraction_snapshot_event(workspace_id: str, *, snapshot: dict[str, int]) -> None:
    """Emit an extraction runtime snapshot."""
    emit_event(
        workspace_id,
        ExtractionSnapshotEvent.model_validate(
            {"event": f"{EXTRACTION_EVENT_PREFIX}.snapshot", **snapshot}
        ),
    )


__all__ = [
    "ActionMessageEvent",
    "EXTRACTION_EVENT_PREFIX",
    "emit_action_message",
    "emit_event",
    "emit_extraction_plan_event",
    "emit_extraction_snapshot_event",
    "emit_extraction_worker_event",
    "emit_transition_event",
    "events_dir",
    "read_events",
]
