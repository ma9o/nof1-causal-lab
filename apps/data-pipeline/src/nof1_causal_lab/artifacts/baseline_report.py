"""Persisted baseline treatment-effect results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .base import ArtifactPayload
from .effects import EffectSummary, HistogramBin, TemporalEffect  # noqa: TC001
from .identity import ConstructId  # noqa: TC001
from .scenarios import SimulationResult  # noqa: TC001


class TreatmentEffect(BaseModel):
    """A treatment effect stores posterior effect draws and optional temporal or observed-scale
    summaries.
    """

    model_config = ConfigDict(extra="forbid")

    treatment: str
    treatment_id: ConstructId
    summary: EffectSummary | None
    histogram: list[HistogramBin] = Field(default_factory=list)
    posterior_draws: list[float] | None = None
    temporal: TemporalEffect | None = None
    manifest_effects: dict[str, float] | None = None


class BaselineReportArtifact(ArtifactPayload):
    """A baseline report collects treatment effects and explicitly retained simulations from the fitted model."""

    intervention_results: list[TreatmentEffect]
    simulation_results: list[SimulationResult] = Field(default_factory=list)
    final_summary: str | None = None
