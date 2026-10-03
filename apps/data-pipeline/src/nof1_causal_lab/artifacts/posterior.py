"""Persisted posterior results and sampling metadata."""

from collections.abc import Mapping

from pydantic import AwareDatetime, Field

from nof1_causal_lab.artifacts.base import Value

from .checks import Assessment
from .identity import ConstructId, ParameterRef
from .posterior_diagnostics import (
    ChainDiagnostics,
    LOODiagnostics,
    LOOPITPoint,
    ParameterConvergenceReport,
    ParetoKPoint,
    ParticleMCMCEvidence,
    ParticleSamplerDiagnostics,
    PosteriorMarginal,
    RankHistogram,
    TemperingDiagnostics,
    TraceSeries,
)


class FitSettingsSpec(Value):
    """Optional numerical controls applied to the configured particle sampler."""

    num_samples: int | None = Field(default=None, ge=1)
    num_warmup: int | None = Field(default=None, ge=0)
    num_chains: int | None = Field(default=None, ge=1)
    n_particles: int | None = Field(default=None, ge=2)
    seed: int | None = Field(default=None, ge=0)


class InferenceMetadata(Value):
    """Inference metadata records the sampling method, sample count, and run duration."""

    method: str
    n_samples: int
    duration_seconds: float


class PosteriorDrawsInfo(Value):
    """Axes of aligned joint draws stored in the posterior's fitted payload."""

    n_draws: int = Field(ge=1)
    parameter_shapes: Mapping[str, tuple[int, ...]]
    state_ids: tuple[ConstructId, ...] = ()
    latent_shape: tuple[int, int] | None = Field(
        default=None, description="Time and state axis lengths per retained latent draw."
    )


class InferenceReportCore(Value):
    """Compact scientific report shared by snapshots and the full report."""

    time_origin: AwareDatetime | None
    inference_metadata: InferenceMetadata
    engine: Assessment[str, ParticleMCMCEvidence]
    inference_diagnostics: ChainDiagnostics | None
    sampler_diagnostics: ParticleSamplerDiagnostics | None
    convergence: ParameterConvergenceReport
    loo_diagnostics: LOODiagnostics | None = None
    posterior_marginals: tuple[PosteriorMarginal, ...] | None = None


class InferenceReportDetail(Value):
    """Retained plot series served in full by the report endpoint."""

    tempering: TemperingDiagnostics | None = None
    trace_data: tuple[TraceSeries, ...] = ()
    rank_histograms: tuple[RankHistogram, ...] = ()
    pareto_k: tuple[ParetoKPoint, ...] = ()
    loo_pit: tuple[LOOPITPoint, ...] = ()
    posterior_pairs: tuple[tuple[ParameterRef, ParameterRef], ...] = ()
    divergent: tuple[bool, ...] | None = None
    initial_latent_delta: tuple[tuple[float, ...], ...] | None = None
    final_latent_delta: tuple[tuple[float, ...], ...] | None = None


class InferenceReport(Value):
    """The compact core composed with retained detail, without filtering or re-parsing."""

    core: InferenceReportCore
    detail: InferenceReportDetail
