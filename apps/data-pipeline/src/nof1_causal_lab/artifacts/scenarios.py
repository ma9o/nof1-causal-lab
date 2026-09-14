"""Live simulation requests and responses, optionally retained together in a report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .effects import EffectSummary, EffectTrajectoryPoint  # noqa: TC001
from .identity import (  # noqa: TC001
    ConstructId,
    ConstructRef,
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

    target: ConstructRef = Field(description="Persistent identity of the construct to clamp.")
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
    outcome: ConstructRef = Field(description="Persistent identity of the requested outcome.")
    readout: ScenarioQueryInput = Field(default_factory=ScenarioQueryInput)


class BaselineReportVisualization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_node_trajectories: dict[ConstructId, list[float]] | None = Field(
        default=None,
        description=(
            "Per-construct latent trajectories for the reference (no-clamp) path aligned to "
            "effect_trajectory days."
        ),
    )
    action_node_trajectories: dict[ConstructId, list[float]] | None = Field(
        default=None,
        description=(
            "Per-construct latent trajectories under the composed clamps aligned to "
            "effect_trajectory days."
        ),
    )
    node_effect_trajectories: dict[ConstructId, list[float]] | None = Field(
        default=None,
        description=(
            "Per-construct latent effect trajectories aligned to effect_trajectory days. "
            "Values are causal deltas relative to the reference path."
        ),
    )
    start_state: dict[ConstructId, float] | None = Field(
        default=None,
        description="Posterior mean latent state the rollout started from.",
    )


class SimulationProvenance(BaseModel):
    """The retained fit and actual numerical settings used by this response."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model: ModelRevision
    engine: Literal["nonlinear_drift_v1", "illustrative_fixture"] = "nonlinear_drift_v1"
    solver: Literal["Tsit5"] = "Tsit5"
    rtol: float = Field(gt=0)
    atol: float = Field(gt=0)
    max_steps: int = Field(ge=1)
    draw_count: int = Field(ge=1)
    time_grid_days: list[float] = Field(min_length=2)
    start_time_index: int | None = Field(default=None, ge=0)
    start_time: str | None = None

    @model_validator(mode="after")
    def validate_basis(self) -> SimulationProvenance:
        from .effects import validate_effect_horizons

        validate_effect_horizons(self.time_grid_days)
        if self.time_grid_days[0] != 0:
            raise ValueError("A simulation time grid starts at zero")
        return self


class SimulationResult(BaseModel):
    """Ephemeral response, retained only when explicitly included in a report.

    This engine integrates the true nonlinear drift for each posterior draw.
    It does not include future process noise or claim the mean of the SDE.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    request: ScenarioRequest
    provenance: SimulationProvenance
    labels: dict[ConstructId, str]
    summary: EffectSummary
    effect_trajectory: list[EffectTrajectoryPoint] | None = None
    trajectory_peak: EffectTrajectoryPoint | None = None
    visualization: BaselineReportVisualization | None = None
    manifest_effects: dict[str, float] | None = None
    reference_mean: float
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_request_basis(self) -> SimulationResult:
        targets = {self.request.outcome.id, *(clamp.target.id for clamp in self.request.clamps)}
        if not targets <= self.labels.keys():
            raise ValueError("Simulation labels must describe the requested constructs")
        if self.request.start.kind == "baseline":
            if (
                self.provenance.start_time_index is not None
                or self.provenance.start_time is not None
            ):
                raise ValueError("A baseline simulation has no observed start time")
        elif self.provenance.start_time_index is None:
            raise ValueError("An abducted simulation must record the resolved observed start")
        return self
