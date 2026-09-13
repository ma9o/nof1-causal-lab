"""Causal scenario requests, resolved query identities, and reported results."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .effects import EffectSummary, EffectTrajectoryPoint  # noqa: TC001
from .identity import (  # noqa: TC001
    ArtifactRef,
    ConstructId,
    ConstructRef,
    ModelRef,
    ScenarioEvaluationId,
    ScenarioQueryId,
)


class ScenarioStartInput(BaseModel):
    """Where the forward rollout begins (replaces the rung-2/rung-3 split)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["baseline", "abducted"] = Field(
        default="baseline",
        description=(
            "'baseline' starts from the population baseline steady state (an interventional, "
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


class LatentClampInput(BaseModel):
    """A do-operator on one latent variable over a time window.

    The window is ``[from_day, to_day)`` in days relative to the rollout start; outside
    the window the variable evolves under its natural dynamics. ``set`` pins to an absolute
    value, ``shift`` adds an amount to the variable's start-state value, ``ramp`` linearly
    interpolates across the window, and ``trajectory`` tracks a list of values across it.
    """

    model_config = ConfigDict(extra="forbid")

    variable: str = Field(description="Latent construct to clamp.")
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
    def validate_payload(self) -> LatentClampInput:
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


class SimulateScenarioInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: ScenarioStartInput = Field(default_factory=ScenarioStartInput)
    clamps: list[LatentClampInput] = Field(
        min_length=1, description="One or more timed latent clamps composing the scenario."
    )
    outcome: str | None = Field(
        default=None,
        description="Outcome construct. Defaults to the workspace’s selected query outcome.",
    )
    query: ScenarioQueryInput = Field(default_factory=ScenarioQueryInput)


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


class ScenarioClamp(LatentClampInput):
    """A resolved clamp binds its transport label to a persistent construct identity."""

    target: ConstructRef


class ScenarioDefinition(BaseModel):
    """A scientific scenario fixes its targets, start rule, and requested readout across fits."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: ScenarioStartInput
    clamps: list[ScenarioClamp] = Field(min_length=1)
    outcome: ConstructRef
    readout: ScenarioQueryInput

    def identity_payload(self) -> str:
        # Transport labels aid presentation; only persistent IDs identify targets.
        payload = self.model_dump(mode="json", exclude={"id"})
        for clamp in payload["clamps"]:
            del clamp["variable"]
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ScenarioQuery(ScenarioDefinition):
    """A scientific query keeps its identity across model and posterior revisions."""

    id: ScenarioQueryId

    @classmethod
    def from_definition(cls, definition: ScenarioDefinition) -> ScenarioQuery:
        identity = "query:" + hashlib.sha256(definition.identity_payload().encode()).hexdigest()
        return cls(id=identity, **definition.model_dump(mode="json"))

    @model_validator(mode="after")
    def validate_identity(self) -> ScenarioQuery:
        expected = "query:" + hashlib.sha256(self.identity_payload().encode()).hexdigest()
        if self.id != expected:
            raise ValueError("Scenario query identity does not match its definition")
        return self


class ScenarioEvaluation(BaseModel):
    """An evaluation binds a scientific query to one model and exact posterior version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: ScenarioEvaluationId
    query_id: ScenarioQueryId
    model: ModelRef
    posterior: ArtifactRef

    @staticmethod
    def identity_for(query_id: ScenarioQueryId, model: ModelRef, posterior: ArtifactRef) -> str:
        payload = json.dumps(
            {
                "query_id": query_id,
                "model": model.model_dump(),
                "posterior": posterior.model_dump(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return "evaluation:" + hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def for_query(
        cls, query: ScenarioQuery, *, model: ModelRef, posterior: ArtifactRef
    ) -> ScenarioEvaluation:
        return cls(
            id=cls.identity_for(query.id, model, posterior),
            query_id=query.id,
            model=model,
            posterior=posterior,
        )

    @model_validator(mode="after")
    def validate_identity(self) -> ScenarioEvaluation:
        if self.posterior.artifact_id != "posterior":
            raise ValueError("A scenario evaluation must pin a posterior artifact")
        if self.id != self.identity_for(self.query_id, self.model, self.posterior):
            raise ValueError("Scenario evaluation identity does not match its execution basis")
        return self


class ScenarioStartResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["baseline", "abducted"]
    time_index: int | None = None
    time: str | None = None
    state_source: Literal["baseline_steady_state", "fitted_latent_paths"]


class ScenarioResult(BaseModel):
    """Computed outputs reference the evaluation that fixes their query and posterior."""

    model_config = ConfigDict(extra="forbid")

    evaluation_id: ScenarioEvaluationId
    start: ScenarioStartResult
    outcome_label: str = Field(description="Display name of the query's outcome at execution time.")
    summary: EffectSummary
    effect_trajectory: list[EffectTrajectoryPoint] | None = None
    trajectory_peak: EffectTrajectoryPoint | None = None
    visualization: BaselineReportVisualization | None = None
    manifest_effects: dict[str, float] | None = None
    reference_mean: float = Field(
        description="Mean reference outcome (baseline steady state or factual forecast)."
    )
    warnings: list[str] = Field(default_factory=list)


class ScenarioEvaluationResult(BaseModel):
    """One posterior-specific evaluation and its matching computed outputs."""

    model_config = ConfigDict(extra="forbid")

    evaluation: ScenarioEvaluation
    result: ScenarioResult

    @model_validator(mode="after")
    def validate_result_evaluation(self) -> ScenarioEvaluationResult:
        if self.result.evaluation_id != self.evaluation.id:
            raise ValueError("Scenario result belongs to a different evaluation")
        return self


class SimulateScenarioResult(ScenarioEvaluationResult):
    """A simulation response includes its reusable scientific query and pinned evaluation."""

    query: ScenarioQuery

    @model_validator(mode="after")
    def validate_evaluation_query(self) -> SimulateScenarioResult:
        if self.evaluation.query_id != self.query.id:
            raise ValueError("Scenario evaluation belongs to a different query")
        return self
