"""Reported causal-effect summaries and their plotting data."""

from pydantic import BaseModel, ConfigDict, Field


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
    """A certified paired contrast and 95% interval at one absolute model time."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    day: float
    effect: float
    lower_95: float
    upper_95: float
