"""Timestamped interventions and certified readouts of generated histories."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .effects import EffectSummary, EffectTrajectoryPoint  # noqa: TC001
from .identity import ConstructId  # noqa: TC001


class InterventionSpec(BaseModel):
    """Set a latent state at one model time, then let its dynamics resume."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: ConstructId
    time: FiniteFloat = Field(description="Absolute time in model days.")
    value: FiniteFloat


class SimulationTrajectory(BaseModel):
    """One construct's mean reference and intervention paths across simulated draws."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    reference_mean: list[float] = Field(
        min_length=2,
        description="Mean natural latent path on the result's time_grid_days, including day zero.",
    )
    action_mean: list[float] = Field(
        min_length=2,
        description="Mean latent path under the dated interventions on the same full time grid.",
    )


class CausalEffectResult(BaseModel):
    """Causal effects and realized trajectories under the enclosing report's design."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    time_grid_days: list[float] = Field(
        min_length=2,
        description="Shared elapsed-day coordinates for all trajectories, including day zero.",
    )
    outcome: ConstructId
    labels: dict[ConstructId, str]
    summary: EffectSummary
    effect_trajectory: list[EffectTrajectoryPoint] | None = None
    trajectory_peak: EffectTrajectoryPoint | None = None
    trajectories: dict[ConstructId, SimulationTrajectory] = Field(
        description="Reference and action means for each simulated construct on time_grid_days.",
    )
    manifest_effects: dict[str, float] | None = None
    reference_mean: float
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_trajectories(self) -> Self:
        from .effects import validate_effect_horizons

        validate_effect_horizons(self.time_grid_days)
        if self.time_grid_days[0] != 0:
            raise ValueError("A simulation time grid starts at zero")
        if not self.trajectories.keys() <= self.labels.keys():
            raise ValueError("Simulation trajectories must have construct labels")
        n_times = len(self.time_grid_days)
        for identity, trajectory in self.trajectories.items():
            if len(trajectory.reference_mean) != n_times or len(trajectory.action_mean) != n_times:
                raise ValueError(
                    f"Simulation trajectories for {identity} must align with time_grid_days"
                )
        return self
