"""Forward generation from a model's current laws and its recorded evidence."""

from __future__ import annotations

from itertools import pairwise
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .checks import PredictiveCheckFinding  # noqa: TC001
from .identity import ConstructId, GitRef, IndicatorId  # noqa: TC001
from .observations import ObservationSpec  # noqa: TC001
from .predictive_provenance import PredictiveLawProvenance  # noqa: TC001
from .scenarios import CausalEffectResult, InterventionSpec  # noqa: TC001


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

    @model_validator(mode="after")
    def resolved_variables(self) -> Self:
        if len({item.id for item in self.variables}) != len(self.variables):
            raise ValueError("Simulation variables must have unique IDs")
        if any(item.observation_window is None for item in self.variables):
            raise ValueError("Simulation variables must retain resolved observation windows")
        return self


class SimulationReport(BaseModel):
    """Generated histories and derived findings with their resolved execution coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: GitRef
    design: SimulationSpec
    times: tuple[FiniteFloat, ...] = Field(min_length=2)
    draws: int = Field(ge=1)
    seed: int = Field(ge=0)
    state_ids: tuple[ConstructId, ...]
    indicator_ids: tuple[IndicatorId, ...]
    parameter_draws: dict[str, str]
    latent_paths: str
    observations: str
    observation_layout: SimulationObservationLayout
    law: PredictiveLawProvenance | None = None
    reference_latent_paths: str | None = None
    reference_observations: str | None = None
    findings: tuple[PredictiveCheckFinding, ...] = ()
    causal_result: CausalEffectResult | None = None
    causal_unavailable_reason: str | None = None

    @model_validator(mode="after")
    def validate_histories(self) -> Self:
        if self.indicator_ids != tuple(item.id for item in self.observation_layout.variables):
            raise ValueError("Simulation observation schema must match its indicator order")
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
            if not targets <= self.causal_result.trajectories.keys():
                raise ValueError("Causal trajectories must include the outcome and interventions")
            if len(self.causal_result.time_grid_days) != len(self.times):
                raise ValueError("Causal trajectories must align with the generated histories")
        return self
