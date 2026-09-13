"""Persisted posterior results and sampling metadata."""

from pydantic import BaseModel, ConfigDict, Field

from .base import ArtifactPayload
from .identity import CausalDesignRef
from .posterior_diagnostics import (
    LOODiagnostics,
    MCMCDiagnostics,
    PosteriorMarginal,
    PosteriorPair,
    PosteriorPredictiveChecks,
    SMCDiagnostics,
)


class InferenceMetadata(BaseModel):
    """Inference metadata records the sampling method, sample count, and run duration."""

    model_config = ConfigDict(extra="forbid")

    method: str
    n_samples: int
    duration_seconds: float


class PosteriorProvenance(BaseModel):
    """Exact model and observation versions defining the posterior distribution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    causal_design: CausalDesignRef
    compiled_ssm_version: int = Field(ge=1)
    panel_version: int = Field(ge=1)


class PosteriorDrawsInfo(BaseModel):
    """Axes of aligned joint draws stored in the posterior's fitted payload."""

    model_config = ConfigDict(extra="forbid")

    n_draws: int = Field(ge=1)
    parameter_shapes: dict[str, list[int]]
    latent_shape: tuple[int, int] | None = Field(
        default=None, description="Time and state axis lengths per retained latent draw."
    )


class PosteriorAssessment(BaseModel):
    """Sampling and predictive assessments of a fitted posterior."""

    model_config = ConfigDict(extra="forbid")

    ppc: PosteriorPredictiveChecks
    mcmc_diagnostics: MCMCDiagnostics | None = None
    smc_diagnostics: SMCDiagnostics | None = None
    loo_diagnostics: LOODiagnostics | None = None


class PosteriorArtifact(ArtifactPayload):
    """A joint posterior's draw axes, exact provenance, summaries, and separate assessment."""

    draws: PosteriorDrawsInfo
    provenance: PosteriorProvenance
    inference_metadata: InferenceMetadata
    assessment: PosteriorAssessment
    posterior_marginals: list[PosteriorMarginal] | None = None
    posterior_pairs: list[PosteriorPair] | None = None
