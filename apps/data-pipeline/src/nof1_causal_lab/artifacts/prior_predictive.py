"""Results of simulating and checking a model under its declared priors."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .identity import ConstructId, IndicatorId  # noqa: TC001


class PriorPredictiveDiagnostic(BaseModel):
    """A measured prior-predictive check and its evaluation criteria."""

    model_config = ConfigDict(extra="forbid")

    check: str
    construct_id: ConstructId
    value: str
    band: str
    passed: bool
    note: str
    diagnosis: list[str] = Field(default_factory=list)
    mode: str


class PriorPredictiveResult(BaseModel):
    """Simulated observations and checks recorded by a model-authoring operation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    samples: dict[IndicatorId, list[float]]
    diagnostics: list[PriorPredictiveDiagnostic] = Field(default_factory=list)
