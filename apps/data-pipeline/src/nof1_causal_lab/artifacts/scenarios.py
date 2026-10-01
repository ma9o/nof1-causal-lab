"""Timestamped interventions and certified readouts of generated histories."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .effects import EffectSummary, EffectTrajectoryPoint
from .identity import ConstructId


class InterventionSpec(BaseModel):
    """Set a latent state at one model time, then let its dynamics resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: ConstructId
    time: FiniteFloat = Field(description="Absolute time in model days.")
    value: FiniteFloat


class CausalEffectResult(BaseModel):
    """Causal effects and realized trajectories under the enclosing report's design."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    outcome: ConstructId
    labels: dict[ConstructId, str]
    summary: EffectSummary
    effect_trajectory: list[EffectTrajectoryPoint]
    trajectory_peak: EffectTrajectoryPoint | None = None
    manifest_effects: dict[str, float] | None = None
    reference_mean: float
    warnings: list[str] = Field(default_factory=list)
