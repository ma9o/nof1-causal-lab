"""Revision-pinned plot coordinates; scientific reductions belong to the server."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import AwareDatetime, Field, FiniteFloat

from nof1_causal_lab.artifacts.availability import Availability
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


class ObservationHistory(Value):
    """All prepared observations, their true anchors and their measurement support."""

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


type ParameterDraws = Availability[tuple[ParameterDrawColumn, ...]]
