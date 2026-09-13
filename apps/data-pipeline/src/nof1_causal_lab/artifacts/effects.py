"""Reported causal-effect summaries and their plotting data."""

from collections.abc import Sequence
from itertools import pairwise
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EffectSummary(BaseModel):
    """An effect summary reports posterior location, uncertainty, and sign probability."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    mean: float
    median: float
    lower_95: float
    upper_95: float
    prob_positive: float = Field(ge=0, le=1)


class HistogramBin(BaseModel):
    """A histogram bin gives its interval, center, and number of posterior draws."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    bin_center: float
    bin_start: float
    bin_end: float
    count: int = Field(ge=0)


class EffectTrajectoryPoint(BaseModel):
    """An effect trajectory point records a causal delta at one elapsed rollout time."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    day: float = Field(ge=0)
    effect: float


def validate_effect_horizons(days: Sequence[float]) -> None:
    """Require finite, nonnegative, strictly increasing elapsed-day horizons."""
    if not days or any(not isfinite(day) or day < 0 for day in days):
        raise ValueError("Effect horizons must be nonempty, finite, and nonnegative")
    if any(right <= left for left, right in pairwise(days)):
        raise ValueError("Effect horizons must be unique and increasing")


class TemporalEffect(BaseModel):
    """A temporal effect summarizes a trajectory at requested horizons and its absolute peak."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    horizons: list[EffectTrajectoryPoint] = Field(min_length=1)
    peak_effect: float
    time_to_peak_days: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_horizons(self) -> "TemporalEffect":
        validate_effect_horizons([point.day for point in self.horizons])
        return self
