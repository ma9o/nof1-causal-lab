"""Persisted baseline treatment-effect results."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import ArtifactPayload
from .effects import EffectSummary, HistogramBin, TemporalEffect  # noqa: TC001
from .identity import ConstructId  # noqa: TC001
from .scenarios import ScenarioEvaluationResult, ScenarioQuery  # noqa: TC001


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


class SavedScenario(BaseModel):
    """A saved scenario preserves a labeled causal query and its optional narrative summary."""

    model_config = ConfigDict(extra="forbid")

    label: str
    query: ScenarioQuery
    evaluations: list[ScenarioEvaluationResult] = Field(default_factory=list)
    summary: str | None = None

    @model_validator(mode="after")
    def validate_evaluations(self) -> SavedScenario:
        identities = [item.evaluation.id for item in self.evaluations]
        if len(identities) != len(set(identities)):
            raise ValueError("Saved scenario contains duplicate evaluations")
        for item in self.evaluations:
            if item.evaluation.query_id != self.query.id:
                raise ValueError("Saved scenario evaluation belongs to a different query")
        return self


class BaselineReportArtifact(ArtifactPayload):
    """A baseline report collects treatment effects and saved scenarios from the fitted model."""

    intervention_results: list[TreatmentEffect]
    saved_scenarios: list[SavedScenario] | None = None
    final_summary: str | None = None


class SavedScenariosArtifact(ArtifactPayload):
    """Saved scenarios preserve the selected queries for a fitted model."""

    scenarios: list[SavedScenario]

    @model_validator(mode="after")
    def validate_unique_queries(self) -> SavedScenariosArtifact:
        identities = [item.query.id for item in self.scenarios]
        if len(identities) != len(set(identities)):
            raise ValueError("Save each scientific query once, with all its evaluations")
        return self
