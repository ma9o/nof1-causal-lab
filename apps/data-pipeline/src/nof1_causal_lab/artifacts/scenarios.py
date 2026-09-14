"""Runtime-only simulation requests and responses for a fitted model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .effects import EffectSummary, EffectTrajectoryPoint  # noqa: TC001
from .identity import (  # noqa: TC001
    ConstructId,
    ModelRevision,
)


class ScenarioStartInput(BaseModel):
    """Where the forward rollout begins (replaces the rung-2/rung-3 split)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["baseline", "abducted"] = Field(
        default="baseline",
        description=(
            "'baseline' starts from the deterministic drift equilibrium (an interventional, "
            "rung-2 query). 'abducted' conditions on the individual's observed evidence and starts "
            "from the recovered fitted latent state (a counterfactual, rung-3 query)."
        ),
    )
    time_index: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Abducted start only: observed fitted-state index to begin from. "
            "Defaults to the final retained fitted latent state."
        ),
    )
    time: str | None = Field(
        default=None,
        description=(
            "Abducted start only: ISO-8601 observed timestamp matching a retained fitted latent "
            "state. Use either time_index or time, not both."
        ),
    )

    @model_validator(mode="after")
    def validate_payload(self) -> ScenarioStartInput:
        if self.time_index is not None and self.time is not None:
            raise ValueError("Use either start.time_index or start.time, not both")
        if self.kind == "baseline" and (self.time_index is not None or self.time is not None):
            raise ValueError("start.kind='baseline' takes no time_index/time")
        return self


class ScenarioClamp(BaseModel):
    """A do-operator on one latent variable over a time window.

    The window is ``[from_day, to_day)`` in days relative to the rollout start; outside
    the window the variable evolves under its natural dynamics. ``set`` pins to an absolute
    value, ``shift`` adds an amount to the variable's start-state value, ``ramp`` linearly
    interpolates across the window, and ``trajectory`` tracks a list of values across it.
    """

    model_config = ConfigDict(extra="forbid")

    target: ConstructId = Field(description="Persistent ID of the construct to clamp.")
    mode: Literal["set", "shift", "ramp", "trajectory"] = Field(
        description="How the clamped value is specified over the window."
    )
    value: float | None = Field(
        default=None, description="Required when mode='set'. Absolute latent-space value."
    )
    amount: float | None = Field(
        default=None,
        description="Required when mode='shift'. Additive delta from the start-state value.",
    )
    value_start: float | None = Field(
        default=None, description="Required when mode='ramp'. Value at from_day."
    )
    value_end: float | None = Field(
        default=None, description="Required when mode='ramp'. Value at to_day."
    )
    values: list[float] | None = Field(
        default=None,
        description="Required when mode='trajectory'. Values sampled evenly across the window.",
    )
    from_day: float = Field(
        default=0.0, ge=0.0, description="Window onset in days from the rollout start."
    )
    to_day: float | None = Field(
        default=None,
        description="Window end in days from the rollout start. Null runs through the horizon.",
    )

    @model_validator(mode="after")
    def validate_payload(self) -> ScenarioClamp:
        if self.to_day is not None and self.to_day <= self.from_day:
            raise ValueError("clamp to_day must be greater than from_day")
        if self.mode == "set" and self.value is None:
            raise ValueError("mode='set' requires value")
        if self.mode == "shift" and self.amount is None:
            raise ValueError("mode='shift' requires amount")
        if self.mode == "ramp" and (self.value_start is None or self.value_end is None):
            raise ValueError("mode='ramp' requires value_start and value_end")
        if self.mode == "trajectory" and (self.values is None or len(self.values) < 2):
            raise ValueError("mode='trajectory' requires values with at least two points")
        return self


class ScenarioQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    estimand: Literal["end_state", "trajectory"] = Field(
        default="trajectory",
        description="Report the final-horizon outcome effect or the full effect trajectory.",
    )
    horizon_days: int = Field(
        default=30, ge=1, le=365, description="Forward horizon in days from the rollout start."
    )
    projection: Literal["latent", "manifest", "both"] = Field(
        default="latent",
        description="Report latent outcome effects, manifest projections, or both.",
    )


class ScenarioRequest(BaseModel):
    """One reusable request for an on-demand simulation of a fitted model."""

    model_config = ConfigDict(extra="forbid")

    start: ScenarioStartInput = Field(default_factory=ScenarioStartInput)
    clamps: list[ScenarioClamp] = Field(
        min_length=1, description="One or more timed latent clamps composing the scenario."
    )
    outcome: ConstructId = Field(description="Persistent ID of the requested outcome.")
    readout: ScenarioQueryInput = Field(default_factory=ScenarioQueryInput)


class SimulationTrajectory(BaseModel):
    """One construct's mean reference and intervention paths across simulated draws."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    reference_mean: list[float] = Field(
        min_length=2,
        description="Mean no-clamp latent path on the result's time_grid_days, including day zero.",
    )
    action_mean: list[float] = Field(
        min_length=2,
        description="Mean latent path under the requested clamps on the same full time grid.",
    )


class SimulationResult(BaseModel):
    """Ephemeral response to a runtime simulation request.

    This engine integrates the true nonlinear drift for each posterior draw.
    It does not include future process noise or claim the mean of the SDE.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    request: ScenarioRequest
    model: ModelRevision = Field(description="Exact fitted model revision used by this response.")
    time_grid_days: list[float] = Field(
        min_length=2,
        description="Shared elapsed-day coordinates for all trajectories, including day zero.",
    )
    start_time_index: int | None = Field(
        default=None,
        ge=0,
        description="Resolved fitted-state index for an abducted start; null for a baseline start.",
    )
    start_time: str | None = Field(
        default=None,
        description="Observed timestamp of the resolved abducted start, when available.",
    )
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
    def validate_request_basis(self) -> SimulationResult:
        from .effects import validate_effect_horizons

        validate_effect_horizons(self.time_grid_days)
        if self.time_grid_days[0] != 0:
            raise ValueError("A simulation time grid starts at zero")
        targets = {self.request.outcome, *(clamp.target for clamp in self.request.clamps)}
        if not targets <= self.labels.keys():
            raise ValueError("Simulation labels must describe the requested constructs")
        if not targets <= self.trajectories.keys():
            raise ValueError("Simulation trajectories must include the requested constructs")
        if not self.trajectories.keys() <= self.labels.keys():
            raise ValueError("Simulation trajectories must have construct labels")
        n_times = len(self.time_grid_days)
        for identity, trajectory in self.trajectories.items():
            if len(trajectory.reference_mean) != n_times or len(trajectory.action_mean) != n_times:
                raise ValueError(
                    f"Simulation trajectories for {identity} must align with time_grid_days"
                )
        if self.request.start.kind == "baseline":
            if self.start_time_index is not None or self.start_time is not None:
                raise ValueError("A baseline simulation has no observed start time")
        elif self.start_time_index is None:
            raise ValueError("An abducted simulation must record the resolved observed start")
        return self
