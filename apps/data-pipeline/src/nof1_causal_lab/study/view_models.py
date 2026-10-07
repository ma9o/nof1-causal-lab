"""Parsed observation histories supplied to the scientific comparison engine."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import AwareDatetime, Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_comparison import DataPoint
from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.artifacts.identity import (
    GitOid,
    IndicatorId,
)
from nof1_causal_lab.artifacts.observations import ResolvedObservationSpec


class DataSeries(Value):
    """One variable's recorded measurements in one history; no pooling across replicas."""

    variable: ResolvedObservationSpec | None
    time_origin: AwareDatetime | None = Field(
        description="Recorded calendar binding; null means the point dates are serialization coordinates, not real dates."
    )
    points: tuple[DataPoint, ...]


class Dataset(Value):
    """One parsed observation history, identified by its immutable source."""

    source: DataRef[GitOid, int]
    series: Mapping[IndicatorId, DataSeries]
