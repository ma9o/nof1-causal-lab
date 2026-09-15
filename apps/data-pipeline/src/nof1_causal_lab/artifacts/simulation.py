"""Explicit simulation designs and their independently recorded results."""

from __future__ import annotations

from itertools import pairwise
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from .identity import ConstructId, IndicatorId, ModelRevision  # noqa: TC001
from .posterior_diagnostics import PosteriorPredictiveChecks  # noqa: TC001
from .scenarios import ScenarioClamp, ScenarioRequest, SimulationResult  # noqa: TC001


class SimulationSpec(BaseModel):
    """A replicated study generated from the selected model's current laws."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["trajectory"] = "trajectory"
    times: tuple[FiniteFloat, ...] = Field(min_length=2)
    draws: int = Field(default=100, ge=1, le=10000)
    seed: int = Field(default=0, ge=0)
    edge_contrasts: bool = False
    initial_state: Literal["new_study", "retained", "fixed", "equilibrium"] = "new_study"
    state_time: FiniteFloat | None = None
    state_values: dict[ConstructId, FiniteFloat] = Field(default_factory=dict)
    process_noise: bool = True
    observation_noise: bool = True
    interventions: tuple[ScenarioClamp, ...] = ()
    context: Literal["exploration", "calibration", "prediction"] = "exploration"
    checks: tuple[Literal["dynamics", "measurement", "data_comparison"], ...] = (
        "dynamics",
        "measurement",
        "data_comparison",
    )
    confinement_growth_ratio: FiniteFloat = Field(default=5.0, gt=1)
    confinement_failure_fraction: FiniteFloat = Field(default=0.01, gt=0, le=1)
    comparison_time_offset: FiniteFloat = 0.0

    @model_validator(mode="after")
    def conditioning_contract(self) -> Self:
        if self.initial_state == "retained":
            if self.state_time is None or self.times[0] != self.state_time:
                raise ValueError(
                    "A retained start requires state_time equal to the first design time"
                )
        elif self.state_time is not None:
            raise ValueError("state_time belongs to a retained-state start")
        if (self.initial_state == "fixed") != bool(self.state_values):
            raise ValueError("A fixed start requires state_values; other starts do not accept them")
        if self.edge_contrasts and (
            self.initial_state != "new_study" or not self.process_noise or self.interventions
        ):
            raise ValueError(
                "Edge-share calibration requires an unintervened new study with process noise"
            )
        for clamp in self.interventions:
            for boundary in (clamp.from_day, clamp.to_day):
                if boundary is not None and self.times[0] + boundary not in self.times:
                    raise ValueError("Include intervention boundaries in the simulation time grid")
        return self

    @field_validator("times")
    @classmethod
    def increasing_times(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(right <= left for left, right in pairwise(value)):
            raise ValueError("Simulation times must be strictly increasing")
        return value


class CausalSimulationSpec(BaseModel):
    """An identified paired scenario, certified against the selected production fit."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["causal"] = "causal"
    query: ScenarioRequest
    draws: int = Field(default=100, ge=1, le=10000)
    seed: int = Field(default=0, ge=0)
    process_noise: bool = False
    observation_noise: bool = False


type SimulationDesign = Annotated[
    SimulationSpec | CausalSimulationSpec, Field(discriminator="kind")
]


class SimulationFinding(BaseModel):
    """A measured quantity with the criterion used to interpret it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check: str
    target: str
    value: str
    criterion: str
    passed: bool | None
    explanation: str


class SimulationReport(BaseModel):
    """Evidence from one explicit simulation, separate from an inference report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: ModelRevision
    design: SimulationDesign
    comparison_panel_version: int | None = None
    state_ids: tuple[ConstructId, ...]
    indicator_ids: tuple[IndicatorId, ...]
    parameter_draws: dict[str, str]
    latent_paths: str
    observations: str
    findings: tuple[SimulationFinding, ...] = ()
    predictive_checks: PosteriorPredictiveChecks | None = None
    reference_latent_paths: str | None = None
    reference_observations: str | None = None
    causal_result: SimulationResult | None = None
