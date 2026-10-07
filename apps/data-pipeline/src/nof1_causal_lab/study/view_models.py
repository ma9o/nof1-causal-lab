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

    variable: ResolvedObservationSpec
    points: tuple[DataPoint, ...]


class Dataset(Value):
    """One parsed observation history, identified by its immutable source."""

    source: DataRef[GitOid, int]
    time_origin: AwareDatetime = Field(description="Calendar instant of model day zero.")
    series: Mapping[IndicatorId, DataSeries]
