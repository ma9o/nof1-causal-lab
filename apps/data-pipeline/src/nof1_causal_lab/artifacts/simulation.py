"""Forward generation from a model's current laws and its recorded evidence."""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .checks import PredictiveCheckFinding
from .identity import ConstructId, GitOid, GitRef, IndicatorId
from .observations import ObservationSpec
from .predictive_provenance import PredictiveLawProvenance
from .scenarios import CausalEffectResult, InterventionSpec


class SimulationSpec(BaseModel):
    """Generate through end, optionally starting earlier and applying dated interventions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    end: FiniteFloat = Field(description="Absolute end time in model days.")
    start: FiniteFloat | None = Field(
        default=None,
        description="Absolute start time in model days; omitted uses the model's latest state time, or zero for its initial-state law.",
    )
    interventions: tuple[InterventionSpec, ...] = ()

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.start is not None and self.end <= self.start:
            raise ValueError("Simulation end must be after start")
        for event in self.interventions:
            if event.time > self.end or (self.start is not None and event.time < self.start):
                raise ValueError("Interventions must occur within the simulation window")
        if len({(event.target, event.time) for event in self.interventions}) != len(
            self.interventions
        ):
            raise ValueError("A state can have only one intervention at each time")
        return self


class SimulationObservationLayout(BaseModel):
    """Saved observation semantics and coordinates; generation truths remain separate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variables: tuple[ObservationSpec, ...]
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
        if any(item.observation_window is None for item in self.variables):
            raise ValueError("Simulation variables must retain resolved observation windows")
        return self


class TrajectorySummary(BaseModel):
    """Pointwise means and fixed 95% quantiles across generated numeric draws."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["numeric"] = "numeric"
    mean: tuple[FiniteFloat | None, ...]
    lower: tuple[FiniteFloat | None, ...]
    upper: tuple[FiniteFloat | None, ...]
    n_draws: tuple[int, ...]


class CategoryProbabilitySummary(BaseModel):
    """Predictive probabilities for each declared level; unobserved anchors are null."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["categorical"] = "categorical"
    probabilities: dict[str, tuple[FiniteFloat | None, ...]]
    n_draws: tuple[int, ...]


type PredictiveSummary = Annotated[
    TrajectorySummary | CategoryProbabilitySummary, Field(discriminator="kind")
]


class SimulationSeriesSummary(BaseModel):
    """One state's or indicator's generated distribution in each simulated arm."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    action: PredictiveSummary
    reference: PredictiveSummary | None = None


type FitReliability = Literal["not_fitted", "converged", "unconverged", "unknown"]


class SimulationPredictiveReport(BaseModel):
    """Model implications, independently of whether a causal contrast is certified."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    states: dict[ConstructId, SimulationSeriesSummary]
    indicators: dict[IndicatorId, SimulationSeriesSummary]
    fit_reliability: FitReliability


class SimulationReport(BaseModel):
    """Generated histories and derived findings with their resolved execution coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: GitRef
    design: SimulationSpec
    times: tuple[FiniteFloat, ...] = Field(min_length=2)
    draws: int = Field(ge=1)
    seed: int = Field(ge=0)
    time_origin: AwareDatetime | None = Field(
        description="Known calendar instant of model day zero."
    )
    origin_panel_revision: GitOid | None = Field(
        default=None,
        description="Panel that supplied the time origin: the fit's panel for fitted laws, otherwise the current panel when present.",
    )
    state_ids: tuple[ConstructId, ...]
    parameter_draws: dict[str, str]
    latent_paths: str
    observations: str
    observation_layout: SimulationObservationLayout
    law: PredictiveLawProvenance | None = None
    reference_latent_paths: str | None = None
    reference_observations: str | None = None
    findings: tuple[PredictiveCheckFinding, ...] = ()
    predictive: SimulationPredictiveReport
    causal_result: CausalEffectResult | None = None
    causal_unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_histories(self) -> Self:
        if any(b <= a for a, b in pairwise(self.times)) or self.times[-1] != self.design.end:
            raise ValueError("Simulation times must increase through the requested end")
        if self.design.start is not None and self.times[0] != self.design.start:
            raise ValueError("Simulation times must begin at the requested start")
        paired = self.reference_latent_paths is not None and self.reference_observations is not None
        if bool(self.design.interventions) != paired or (
            (self.reference_latent_paths is None) != (self.reference_observations is None)
        ):
            raise ValueError("Interventions require paired reference histories")
        if self.causal_result is not None:
            if not paired or self.causal_unavailable_reason is not None:
                raise ValueError(
                    "Certified effects require paired histories and no rejection reason"
                )
            targets = {
                self.causal_result.outcome,
                *(event.target for event in self.design.interventions),
            }
            if not targets <= self.predictive.states.keys():
                raise ValueError("Causal trajectories must include the outcome and interventions")
            if tuple(point.day for point in self.causal_result.effect_trajectory) != self.times:
                raise ValueError("Causal trajectories must align with the generated histories")
        if set(self.predictive.states) != set(self.state_ids) or set(
            self.predictive.indicators
        ) != set(self.observation_layout.indicator_ids):
            raise ValueError("Predictive summaries must cover the recorded simulation layout")
        for series in (*self.predictive.states.values(), *self.predictive.indicators.values()):
            if (series.reference is not None) != paired:
                raise ValueError("Predictive summaries must retain each simulated arm")
            for summary in (series.action, series.reference):
                if summary is None:
                    continue
                columns = (
                    (summary.mean, summary.lower, summary.upper)
                    if isinstance(summary, TrajectorySummary)
                    else tuple(summary.probabilities.values())
                )
                if any(len(column) != len(self.times) for column in (*columns, summary.n_draws)):
                    raise ValueError("Predictive summaries must align with simulation times")
        return self
