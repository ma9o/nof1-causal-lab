"""Timestamped interventions and certified readouts of generated histories."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ConfigDict, Field, FiniteFloat

from nof1_causal_lab.artifacts.base import Value

from .effects import EffectSummary, EffectTrajectoryPoint
from .identity import ConstructId


class InterventionSpec(Value):
    """Set a latent state at one model time, then let its dynamics resume."""

    target: ConstructId
    time: FiniteFloat = Field(description="Absolute time in model days.")
    value: FiniteFloat


class CausalEffectResult(Value):
    """Causal effects and realized trajectories under the enclosing report's design."""

    model_config = ConfigDict(allow_inf_nan=False)

    outcome: ConstructId
    labels: Mapping[ConstructId, str]
    summary: EffectSummary
    effect_trajectory: tuple[EffectTrajectoryPoint, ...]
    trajectory_peak: EffectTrajectoryPoint | None = None
    manifest_effects: Mapping[str, float] | None = None
    reference_mean: float
    warnings: tuple[str, ...] = Field(default_factory=tuple)
