"""Persisted posterior results and sampling metadata."""

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.json_types import JsonObject

from .identity import ConstructId
from .posterior_diagnostics import (
    LOODiagnostics,
    PosteriorMarginal,
    PosteriorPair,
    PosteriorPredictiveChecks,
)


class InferenceMetadata(BaseModel):
    """Inference metadata records the sampling method, sample count, and run duration."""

    model_config = ConfigDict(extra="forbid")

    method: str
    n_samples: int
    duration_seconds: float


class PosteriorDrawsInfo(BaseModel):
    """Axes of aligned joint draws stored in the posterior's fitted payload."""

    model_config = ConfigDict(extra="forbid")

    n_draws: int = Field(ge=1)
    parameter_shapes: dict[str, list[int]]
    state_ids: tuple[ConstructId, ...] = ()
    latent_shape: tuple[int, int] | None = Field(
        default=None, description="Time and state axis lengths per retained latent draw."
    )


class PosteriorAssessment(BaseModel):
    """Predictive assessments of a fitted posterior."""

    model_config = ConfigDict(extra="forbid")

    ppc: PosteriorPredictiveChecks
    loo_diagnostics: LOODiagnostics | None = None


class InferenceReport(BaseModel):
    """Display findings recorded by an inference transition, separate from ModelSpec."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    inference_metadata: InferenceMetadata
    inference_diagnostics: JsonObject = Field(
        default_factory=dict,
        description="Engine-reported telemetry for this fit; keys and values are engine-defined.",
    )
    assessment: PosteriorAssessment
    posterior_marginals: list[PosteriorMarginal] | None = None
    posterior_pairs: list[PosteriorPair] | None = None
