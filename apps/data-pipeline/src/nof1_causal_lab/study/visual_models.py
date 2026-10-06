"""Revision-pinned plot coordinates; scientific reductions belong to the server."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import AwareDatetime, Field, FiniteFloat, computed_field

from nof1_causal_lab.artifacts.availability import Availability
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.display_frames import central_frame
from nof1_causal_lab.artifacts.effects import EffectSummary
from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId, ParameterRef
from nof1_causal_lab.artifacts.simulation import CategoryProbabilitySummary


class RecordedPath(Value):
    """One original trajectory identified by its retained draw index."""

    draw: int = Field(description="Index of this draw in the retained simulation evidence.")
    values: tuple[FiniteFloat | None, ...] = Field(
        description=(
            "Values aligned with the enclosing time grid; non-finite entries are represented by"
            " null."
        )
    )


class PathSeries(Value):
    """Action and reference trajectories for one plotted scientific quantity."""

    label: str = Field(description="Display name of the state, indicator, or effect.")
    action: tuple[RecordedPath, ...] = Field(
        description="Recorded draws from the requested simulation arm."
    )
    reference: tuple[RecordedPath, ...] = Field(
        default=(), description="Matched natural-course draws when a reference arm exists."
    )
    levels: tuple[str, ...] | None = Field(
        default=None,
        description="Labels for discrete category codes, or null for numeric quantities.",
    )

    @computed_field
    @property
    def frame(self) -> tuple[float, float] | None:
        """Value range charts show: the widest per-time central 95% of both arms' draws."""
        return central_frame(path.values for path in (*self.action, *self.reference))


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
    """A step of an empirical cumulative distribution."""

    value: FiniteFloat = Field(description="Distinct finite observed or sampled value.")
    probability: FiniteFloat = Field(
        description="Fraction of finite values less than or equal to `value`."
    )


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


type ObservationData = Mapping[IndicatorId, ObservationHistory]


class ParameterDrawColumn(Value):
    """Posterior draws for one scientifically identified parameter coordinate."""

    label: str = Field(description="Display name of the parameter coordinate.")
    subject: ParameterRef = Field(
        description="Parameter and element identity to which the draws belong."
    )
    values: tuple[FiniteFloat, ...] = Field(description="Retained finite draws in sampling order.")
    empirical: tuple[EmpiricalPoint, ...] = Field(
        description="Empirical cumulative distribution of those values."
    )


type ParameterDraws = Availability[tuple[ParameterDrawColumn, ...]]
