"""Forward generation from a model's current laws and its recorded evidence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from itertools import pairwise
from typing import Literal, Self

from pydantic import AwareDatetime, Field, FiniteFloat, computed_field, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_ref import DataRef

from .availability import Available, Evaluation, NotApplicable, Unavailable
from .checks import PredictiveAssessment
from .duration import Duration
from .identity import ConstructId, GitOid, GitRef, IndicatorId
from .observations import ResolvedObservationSpec
from .predictive_provenance import PredictiveLawProvenance
from .scenarios import CausalEffectResult, InterventionSpec, StateAssignment


class SimulationSpec(Value):
    """Generate from a calendar day over a horizon, with interventions placed after the start.

    The start is the only absolute time. A record's origin places it in model days;
    without a record, the start is model day zero.
    """

    start: date = Field(description="Calendar day the window starts, at 00:00 UTC.")
    horizon: Duration = Field(description="How long the window lasts, such as 9w or 61d.")
    interventions: tuple[InterventionSpec, ...] = ()

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        """Require interventions within the horizon and at most one assignment per state and time."""
        if any(
            event.after is not None and event.after.seconds >= self.horizon.seconds
            for event in self.interventions
        ):
            raise ValueError("Interventions must occur before the end of the horizon")
        offsets = {
            (event.target, event.after.seconds if event.after is not None else 0)
            for event in self.interventions
        }
        if len(offsets) != len(self.interventions):
            raise ValueError("A state can have only one intervention at each time")
        return self

    @property
    def start_instant(self) -> datetime:
        """Requested simulation start at midnight UTC."""
        return datetime.combine(self.start, time(), tzinfo=UTC)

    def start_day(self, origin: datetime) -> float:
        """The start in model days after a record's origin."""
        from nof1_causal_lab.utils.time_coordinates import ObservationInstant

        return ObservationInstant(self.start_instant).relative_to(ObservationInstant(origin)).days

    def end_day(self, origin: datetime) -> float:
        """Express the simulation horizon's endpoint as model days after the supplied origin."""
        return self.start_day(origin) + self.horizon.days

    def assignments(self, origin: datetime) -> tuple[StateAssignment, ...]:
        """Place every intervention in model days after a record's origin."""
        start = self.start_day(origin)
        return tuple(
            StateAssignment(
                target=event.target,
                time=start + event.after.days if event.after is not None else start,
                value=event.value,
            )
            for event in self.interventions
        )


class SimulationObservationLayout(Value):
    """Saved observation semantics and coordinates; generation truths remain separate."""

    variables: tuple[ResolvedObservationSpec, ...]
    support_start_times: str
    support_end_times: str
    mask: str

    @property
    def indicator_ids(self) -> tuple[IndicatorId, ...]:
        """The observation axis is owned by the ordered variable definitions."""
        return tuple(item.id for item in self.variables)

    @model_validator(mode="after")
    def resolved_variables(self) -> Self:
        """Reject simulation layouts with repeated observation identities."""
        if len({item.id for item in self.variables}) != len(self.variables):
            raise ValueError("Simulation variables must have unique IDs")
        return self


class CategoryProbabilitySummary(Value):
    """Predictive probabilities for each declared level; unobserved anchors are null."""

    probabilities: Mapping[str, tuple[FiniteFloat | None, ...]]
    n_draws: tuple[int, ...]


type FitReliability = Literal["not_fitted", "converged", "unconverged", "unknown"]


class SimulationEvidence(Value):
    """Exact generated histories with their production coordinates and input provenance."""

    model: GitRef
    design: SimulationSpec
    time_origin: AwareDatetime = Field(description="Calendar instant of model day zero.")
    times: tuple[FiniteFloat, ...] = Field(min_length=2)
    draws: int = Field(ge=1)
    seed: int = Field(ge=0)
    origin_data: DataRef[GitOid, int] | None = Field(
        default=None,
        description="Panel that supplied the time origin: the fit's panel for fitted laws, otherwise the explicitly named panel when present.",
    )
    state_ids: tuple[ConstructId, ...]
    parameter_draws: Mapping[str, str]
    latent_paths: str
    observations: str
    observation_layout: SimulationObservationLayout
    reference_latent_paths: str | None = None
    reference_observations: str | None = None

    @computed_field
    @property
    def assignments(self) -> tuple[StateAssignment, ...]:
        """Intervention assignments positioned on the evidence's retained model-time origin."""
        return self.design.assignments(self.time_origin)

    @model_validator(mode="after")
    def validate_histories(self) -> Self:
        """Require the requested time span and paired reference histories exactly for interventions."""
        if (
            any(right <= left for left, right in pairwise(self.times))
            or self.times[0] != self.design.start_day(self.time_origin)
            or self.times[-1] != self.design.end_day(self.time_origin)
        ):
            raise ValueError("Simulation times must increase from the requested start to its end")
        paired = self.reference_latent_paths is not None and self.reference_observations is not None
        if bool(self.design.interventions) != paired or (
            (self.reference_latent_paths is None) != (self.reference_observations is None)
        ):
            raise ValueError("Interventions require paired reference histories")
        return self


class ModelSimulationResult(Value):
    """The exact generated histories and provenance retained by one simulation."""

    evidence: SimulationEvidence


class SimulationReport(Value):
    """Findings evaluated during simulation and retained with its exact evidence."""

    evidence: SimulationEvidence
    law: PredictiveLawProvenance
    findings: tuple[PredictiveAssessment, ...] = ()
    fit_reliability: FitReliability
    causal: Evaluation[CausalEffectResult]

    def with_causal_result(self, result: CausalEffectResult) -> Self:
        """Return a report containing the available causal-effect result."""
        return self.revised(causal=Available(value=result))

    def without_causal_result(self, reason: str) -> Self:
        """Return a report explaining why the requested causal-effect result is unavailable."""
        return self.revised(causal=Unavailable(reason=reason))

    @model_validator(mode="after")
    def own_causal_scope(self) -> Self:
        """Require causal findings to match intervention scope and retain all involved state trajectories."""
        if isinstance(self.causal, NotApplicable) != (not self.evidence.design.interventions):
            raise ValueError("Causal evaluation applies exactly when interventions are requested")
        if isinstance(self.causal, Available) and not {
            self.causal.value.outcome,
            *(event.target for event in self.evidence.design.interventions),
        } <= set(self.evidence.state_ids):
            raise ValueError("Causal trajectories must include the outcome and interventions")
        return self
