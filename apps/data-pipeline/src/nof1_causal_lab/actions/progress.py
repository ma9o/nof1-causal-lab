"""Live progress of a running attempt: ingestion and extraction step events.

Events land as one JSON file each under ``data/{workspace_id}/scratch/events/`` (neither
local fs nor R2 supports atomic append); readers list the directory and sort by
filename, which is time-ordered by construction. Progress is disposable: the attempt
record and its traces are authoritative, and the sweep truncates the stream freely.
Every event names the attempt that emitted it, and each attempt reads only its own.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

type ProgressStep = Literal["ingestion", "extraction"]
type StepStatus = Literal["running", "completed", "failed"]


class ProgressEventModel(Value):
    """One attempt's immutable event record, with a cursor added only on reads."""

    attempt_id: uuid.UUID
    cursor: str = ""


class StepError(Value):
    """The error type and message of a failed step."""

    type: str
    message: str


class StepEvent(ProgressEventModel):
    """A data-preparation step changed status."""

    event: Literal["nof1-causal-lab.step"] = "nof1-causal-lab.step"
    step: ProgressStep
    status: StepStatus
    error: StepError | None = None


class ExtractionPlanEvent(ProgressEventModel):
    """The extraction fan-out plan."""

    event: Literal["nof1-causal-lab.extraction.plan"] = "nof1-causal-lab.extraction.plan"
    total_workers: int = Field(ge=0)
    max_concurrent_workers: int | None = Field(default=None, gt=0)


class ExtractionWorkerEvent(ProgressEventModel):
    """One extraction worker's state; a worker reports its LLM calls when it finishes."""

    event: Literal["nof1-causal-lab.extraction.worker"] = "nof1-causal-lab.extraction.worker"
    worker_id: int = Field(ge=0)
    state: Literal["pending", "running", "completed", "failed"]
    n_windows: int = Field(ge=0)
    n_extractions: int | None = Field(default=None, ge=0)
    n_llm_calls: int | None = Field(default=None, ge=0)
    error: str | None = None


class ExtractionSnapshotEvent(ProgressEventModel):
    """Aggregate extraction worker counts."""

    event: Literal["nof1-causal-lab.extraction.snapshot"] = "nof1-causal-lab.extraction.snapshot"
    total_workers: int = Field(ge=0)
    pending_workers: int = Field(ge=0)
    running_workers: int = Field(ge=0)
    completed_workers: int = Field(ge=0)
    failed_workers: int = Field(ge=0)


type ProgressEvent = Annotated[
    StepEvent | ExtractionPlanEvent | ExtractionWorkerEvent | ExtractionSnapshotEvent,
    Field(discriminator="event"),
]

_PROGRESS_EVENT_ADAPTER = TypeAdapter(ProgressEvent)


def events_dir(workspace_id: str) -> str:
    return data_module.scratch_events_dir(workspace_id)


def emit_event(workspace_id: str, event: ProgressEvent) -> None:
    directory = events_dir(workspace_id)
    storage.makedirs(directory)
    name = f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}.json"
    storage.write_text(
        storage.join(directory, name),
        json.dumps(event.model_dump(mode="json", exclude_none=True, exclude={"cursor"})),
    )


def read_events(
    workspace_id: str, attempt_id: uuid.UUID, *, after: str | None = None
) -> list[ProgressEvent]:
    """One attempt's retained events in emission order; ``after`` is the last seen cursor."""
    directory = events_dir(workspace_id)
    if not storage.exists(directory):
        return []
    events: list[ProgressEvent] = []
    for entry in sorted(e for e in storage.listdir(directory) if e.endswith(".json")):
        cursor = entry.rsplit("/", 1)[-1]
        if after is not None and cursor <= after:
            continue
        event = _PROGRESS_EVENT_ADAPTER.validate_python(
            {**storage.read_json(entry), "cursor": cursor}
        )
        if event.attempt_id == attempt_id:
            events.append(event)
    return events
