"""Revision-pinned plot coordinates; scientific reductions belong to the server."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.effects import EffectSummary
from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId, ParameterRef
from nof1_causal_lab.artifacts.simulation import CategoryProbabilitySummary


class RecordedPath(Value):
    draw: int
    values: tuple[FiniteFloat | None, ...]


class PathSeries(Value):
    label: str
    action: tuple[RecordedPath, ...]
    reference: tuple[RecordedPath, ...] = ()
    levels: tuple[str, ...] | None = None


class SimulationPaths(Value):
    """Contiguous pages of original draws, with every recorded time point intact."""

    times: tuple[FiniteFloat, ...]
    time_origin: AwareDatetime | None
    total_draws: int
    start: int
    count: int
    states: Mapping[ConstructId, PathSeries]
    indicators: Mapping[IndicatorId, PathSeries]
    effect: PathSeries | None = None
    effect_summary: EffectSummary | None = None
    reference_mean: FiniteFloat | None = None
    manifest_effects: Mapping[IndicatorId, FiniteFloat] = Field(default_factory=dict)
    action_category_probabilities: Mapping[IndicatorId, CategoryProbabilitySummary] = Field(
        default_factory=dict
    )
    reference_category_probabilities: Mapping[IndicatorId, CategoryProbabilitySummary] = Field(
        default_factory=dict
    )


class EmpiricalPoint(Value):
    value: FiniteFloat
    probability: FiniteFloat
    count: int


class ObservationHistory(Value):
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


class ParameterDrawColumn(Value):
    label: str
    subject: ParameterRef
    values: tuple[FiniteFloat, ...]
    empirical: tuple[EmpiricalPoint, ...]


class ParameterDraws(Value):
    """Every retained parameter coordinate, without thinning or pair selection."""

    columns: tuple[ParameterDrawColumn, ...]
    unavailable_reason: str | None = None


class MechanismViewRequest(Value):
    owner_id: str
    axis: ConstructId | None = None
    lower: FiniteFloat = -3
    upper: FiniteFloat = 3
    held: Mapping[ConstructId, FiniteFloat] = Field(default_factory=dict)
    moderator: ConstructId | None = None
    levels: tuple[FiniteFloat, ...] = Field(default=(-1, 0, 1), min_length=1, max_length=5)
    start: int = Field(default=0, ge=0)
    count: int = Field(default=24, ge=1, le=128)
    points: int = Field(default=201, ge=21, le=1001)

    @model_validator(mode="after")
    def range_order(self) -> MechanismViewRequest:
        if self.upper <= self.lower:
            raise ValueError("Response range must increase")
        return self


class ResponseCurve(Value):
    draw: int
    values: tuple[FiniteFloat | None, ...]
    level: FiniteFloat | None = None


class MechanismCurves(Value):
    """Exact conditional drift contributions, not marginal or total causal effects."""

    axis: ConstructId
    axis_label: str
    target_label: str
    states: Mapping[ConstructId, str]
    held: Mapping[ConstructId, FiniteFloat]
    moderator: ConstructId | None
    x: tuple[FiniteFloat, ...]
    curves: tuple[ResponseCurve, ...]
    law: Literal["retained", "sampled", "fixed"]
    total_draws: int
    start: int
    count: int
    nonfinite: int
