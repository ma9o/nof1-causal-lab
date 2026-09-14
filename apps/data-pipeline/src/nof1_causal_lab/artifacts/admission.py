"""Findings from admitting a scientific model and its declared priors."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .base import ArtifactPayload
from .identity import ConstructId, IndicatorId  # noqa: TC001


class PriorPredictiveDiagnostic(BaseModel):
    """A prior predictive diagnostic records the result of one exact model-admission check."""

    model_config = ConfigDict(extra="forbid")

    check: str
    construct_id: ConstructId
    value: str
    band: str
    passed: bool
    note: str
    diagnosis: list[str] = Field(default_factory=list)
    mode: str


class AdmissionReport(ArtifactPayload):
    """Prior research and admission findings pinned to the model that was checked."""

    search_queries: dict[str, str] | None = None
    validation_warnings: list[str] | None = None
    prior_predictive_samples: dict[IndicatorId, list[float]] | None = None
    prior_predictive_diagnostics: list[PriorPredictiveDiagnostic] = Field(default_factory=list)
