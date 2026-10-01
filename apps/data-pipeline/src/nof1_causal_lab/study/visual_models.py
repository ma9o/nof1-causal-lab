"""Revision-pinned plot coordinates; scientific reductions belong to the server."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId, ParameterRef
from nof1_causal_lab.artifacts.posterior_diagnostics import PPCOverlay
from nof1_causal_lab.study.view_models import ViewValue


class RecordedPath(ViewValue):
    draw: int
    values: tuple[FiniteFloat | None, ...]


class PathSeries(ViewValue):
    label: str
    action: tuple[RecordedPath, ...]
    reference: tuple[RecordedPath, ...] = ()
    levels: tuple[str, ...] | None = None


class SimulationPaths(ViewValue):
    """Contiguous pages of original draws, with every recorded time point intact."""

    times: tuple[FiniteFloat, ...]
    time_origin: AwareDatetime | None
    total_draws: int
    start: int
    count: int
    states: dict[ConstructId, PathSeries]
    indicators: dict[IndicatorId, PathSeries]
    effect: PathSeries | None = None


class EmpiricalPoint(ViewValue):
    value: FiniteFloat
    probability: FiniteFloat
    count: int


class ObservationHistory(ViewValue):
    """All prepared observations, their true anchors and their measurement support."""

    indicator_id: IndicatorId
    label: str
    times: tuple[FiniteFloat, ...]
    values: tuple[FiniteFloat | None, ...]
    support_start: tuple[FiniteFloat | None, ...]
    support_end: tuple[FiniteFloat | None, ...]
    time_origin: AwareDatetime | None
    levels: tuple[str, ...] | None
    empirical: tuple[EmpiricalPoint, ...]


class PredictiveHistory(ViewValue):
    """A saved check on the exact schedule and scale used to evaluate it."""

    times: tuple[FiniteFloat, ...]
    time_origin: AwareDatetime | None
    standardized: bool
    overlay: PPCOverlay


class ParameterDrawColumn(ViewValue):
    label: str
    subject: ParameterRef
    values: tuple[FiniteFloat, ...]
    empirical: tuple[EmpiricalPoint, ...]


class ParameterDraws(ViewValue):
    """Every retained parameter coordinate, without thinning or pair selection."""

    columns: tuple[ParameterDrawColumn, ...]
    unavailable_reason: str | None = None


class MechanismViewRequest(ViewValue):
    owner_id: str
    axis: ConstructId | None = None
    lower: FiniteFloat = -3
    upper: FiniteFloat = 3
    held: dict[ConstructId, FiniteFloat] = Field(default_factory=dict)
    moderator: ConstructId | None = None
    levels: tuple[FiniteFloat, ...] = Field(default=(-1, 0, 1), min_length=1, max_length=5)
    start: int = Field(default=0, ge=0)
    count: int = Field(default=24, ge=1, le=128)
    points: int = Field(default=201, ge=21, le=1001)

    @model_validator(mode="after")
    def range_order(self):
        if self.upper <= self.lower:
            raise ValueError("Response range must increase")
        return self


class ResponseCurve(RecordedPath):
    level: FiniteFloat | None = None


class MechanismCurves(ViewValue):
    """Exact conditional drift contributions, not marginal or total causal effects."""

    axis: ConstructId
    axis_label: str
    target_label: str
    states: dict[ConstructId, str]
    held: dict[ConstructId, FiniteFloat]
    moderator: ConstructId | None
    x: tuple[FiniteFloat, ...]
    curves: tuple[ResponseCurve, ...]
    law: Literal["retained", "sampled", "fixed"]
    total_draws: int
    start: int
    count: int
    nonfinite: int
