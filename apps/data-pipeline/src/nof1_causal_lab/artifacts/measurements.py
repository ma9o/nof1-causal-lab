"""Persisted extraction-worker results."""

from typing import Literal

from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from .base import ArtifactPayload
from .identity import IndicatorId


class ObservationRecord(TypedDict):
    """Canonical serialized extraction observation row."""

    indicator_id: IndicatorId
    value: str | int | float | bool | None
    anchor_time: str | None
    support_kind: str | None
    summary_operator: str | None
    anchor_policy: str | None
    observation_window: str | None
    support_start: str | None
    support_end: str | None


class WorkerStatus(BaseModel):
    """A worker status reports extraction progress, produced measurements, and any failure for
    one worker.
    """

    model_config = ConfigDict(extra="forbid")

    worker_id: int
    status: Literal["pending", "running", "completed", "failed"]
    n_extractions: int
    n_windows: int
    error: str | None = None


class MeasurementsArtifact(ArtifactPayload):
    """This artifact records extraction-worker progress and output counts for a measurement
    run.
    """

    workers: list[WorkerStatus]
