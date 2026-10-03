"""Forward generation from a model's current laws and its recorded evidence."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from itertools import pairwise
from typing import Literal, Self

from pydantic import AwareDatetime, Field, FiniteFloat, model_validator

from nof1_causal_lab.artifacts.base import Value

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
        return datetime.combine(self.start, time(), tzinfo=UTC)

    def start_day(self, origin: datetime) -> float:
        """The start in model days after a record's origin."""
        from nof1_causal_lab.utils.time_coordinates import ObservationInstant

        return ObservationInstant(self.start_instant).relative_to(ObservationInstant(origin)).days

    def end_day(self, origin: datetime) -> float:
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
        if len({item.id for item in self.variables}) != len(self.variables):
            raise ValueError("Simulation variables must have unique IDs")
        return self


class CategoryProbabilitySummary(Value):
    """Predictive probabilities for each declared level; unobserved anchors are null."""

    kind: Literal["categorical"] = "categorical"
    probabilities: Mapping[str, tuple[FiniteFloat | None, ...]]
    n_draws: tuple[int, ...]


type FitReliability = Literal["not_fitted", "converged", "unconverged", "unknown"]


class SimulationReport(Value):
    """Generated histories and derived findings with their resolved execution coordinates."""

    model: GitRef
    design: SimulationSpec
    time_origin: AwareDatetime = Field(description="Calendar instant of model day zero.")
    assignments: tuple[StateAssignment, ...] = Field(
        description="The design's interventions in model days."
    )
    times: tuple[FiniteFloat, ...] = Field(min_length=2)
    draws: int = Field(ge=1)
    seed: int = Field(ge=0)
    origin_panel_revision: GitOid | None = Field(
        default=None,
        description="Panel that supplied the time origin: the fit's panel for fitted laws, otherwise the current panel when present.",
    )
    state_ids: tuple[ConstructId, ...]
    parameter_draws: Mapping[str, str]
    latent_paths: str
    observations: str
    observation_layout: SimulationObservationLayout
    law: PredictiveLawProvenance | None = None
    reference_latent_paths: str | None = None
    reference_observations: str | None = None
    findings: tuple[PredictiveAssessment, ...] = ()
    fit_reliability: FitReliability
    causal: Evaluation[CausalEffectResult]

    def with_provenance(
        self, *, law: PredictiveLawProvenance, origin_panel_revision: GitOid | None
    ) -> Self:
        return self.revised(law=law, origin_panel_revision=origin_panel_revision)

    def with_causal_result(self, result: CausalEffectResult) -> Self:
        return self.revised(causal=Available(value=result))

    def without_causal_result(self, reason: str) -> Self:
        return self.revised(causal=Unavailable(reason=reason))

    @model_validator(mode="after")
    def validate_histories(self) -> Self:
        if (
            any(b <= a for a, b in pairwise(self.times))
            or self.times[0] != self.design.start_day(self.time_origin)
            or self.times[-1] != self.design.end_day(self.time_origin)
        ):
            raise ValueError("Simulation times must increase from the requested start to its end")
        if self.assignments != self.design.assignments(self.time_origin):
            raise ValueError("Assignments must place the design's interventions in model days")
        paired = self.reference_latent_paths is not None and self.reference_observations is not None
        if bool(self.design.interventions) != paired or (
            (self.reference_latent_paths is None) != (self.reference_observations is None)
        ):
            raise ValueError("Interventions require paired reference histories")
        if isinstance(self.causal, NotApplicable) != (not self.design.interventions):
            raise ValueError("Causal evaluation applies exactly when interventions are requested")
        if isinstance(self.causal, Available):
            if not paired:
                raise ValueError("Certified effects require paired histories")
            targets = {
                self.causal.value.outcome,
                *(event.target for event in self.design.interventions),
            }
            if not targets <= set(self.state_ids):
                raise ValueError("Causal trajectories must include the outcome and interventions")
        return self
