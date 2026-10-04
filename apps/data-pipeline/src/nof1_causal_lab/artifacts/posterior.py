"""Persisted posterior results and sampling metadata."""

from collections.abc import Mapping

from pydantic import AwareDatetime, Field

from nof1_causal_lab.artifacts.base import Value
from .checks import Assessment
from .identity import ConstructId, DistributionId
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
    """Run measurements for the production particle sampler."""

    n_samples: int
    duration_seconds: float


class PosteriorDrawsInfo(Value):
    """Axes of aligned joint draws stored in the posterior's fitted payload."""

    n_draws: int = Field(ge=1)
    state_ids: tuple[ConstructId, ...] = ()


class InferenceEvidence(Value):
    """Native execution telemetry; posterior atoms and coordinates belong to the model."""

    distribution: DistributionId
    time_origin: AwareDatetime | None
    duration_seconds: float = Field(ge=0)
    engine: ParticleMCMCEvidence | None
    num_chains: int | None = Field(default=None, ge=1)
    chain_extra_fields: Mapping[str, str] = Field(default_factory=dict)
    observation_log_probs: str | None = None
    observed_rows: str | None = None
    exact_observation_rows: str | None = None
    sampler_diagnostics: ParticleSamplerDiagnostics | None = None
    phase_extra_fields: Mapping[str, Mapping[str, str]] = Field(default_factory=dict)
    warmup_complete_log_posterior_history: str | None = None
    all_complete_log_posterior_history: str | None = None
    initial_latent_delta: str | None = None
    final_latent_delta: str | None = None

    @property
    def array_references(self) -> frozenset[str]:
        """Native buffers that must accompany the retained evidence."""
        return frozenset(
            (
                *self.chain_extra_fields.values(),
                *(ref for fields in self.phase_extra_fields.values() for ref in fields.values()),
                *(
                    ref
                    for ref in (
                        self.observation_log_probs,
                        self.observed_rows,
                        self.exact_observation_rows,
                        self.warmup_complete_log_posterior_history,
                        self.all_complete_log_posterior_history,
                        self.initial_latent_delta,
                        self.final_latent_delta,
                    )
                    if ref is not None
                ),
            )
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
    """Retained plot series served in full by the action result."""

    trace_data: tuple[TraceSeries, ...] = ()
    rank_histograms: tuple[RankHistogram, ...] = ()
    pareto_k: tuple[ParetoKPoint, ...] = ()
    loo_pit: tuple[LOOPITPoint, ...] = ()
    divergent: tuple[bool, ...] | None = None
    initial_latent_delta: tuple[tuple[float, ...], ...] | None = None
    final_latent_delta: tuple[tuple[float, ...], ...] | None = None


class InferenceReport(Value):
    """The compact core composed with retained detail, without filtering or re-parsing."""

    core: InferenceReportCore
    detail: InferenceReportDetail
