"""Predictive assessments and scientific posterior summaries."""

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from nof1_causal_lab.artifacts.base import Value

from .checks import Assessment, ConvergenceSubject, IndicatorCheckSubject, NumericCriterionEvidence
from .effects import HistogramBin
from .identity import IndicatorId, ParameterRef


class ParameterDiagnostics(Value):
    """Measurements on one scientifically identified retained scalar chain."""

    parameter: str
    subject: ParameterRef
    r_hat: float | None
    ess_bulk: float | None
    ess_tail: float | None
    mcse_mean: float | None


class TraceSeries(Value):
    """Every retained draw, grouped in original chain order."""

    parameter: str
    subject: ParameterRef
    chains: tuple[tuple[float, ...], ...]


class RankHistogram(Value):
    """Pooled-rank bin counts, grouped in original chain order."""

    parameter: str
    subject: ParameterRef
    n_bins: int
    expected_per_bin: float
    chains: tuple[tuple[int, ...], ...]


class DensityHistogram(Value):
    """A normalized energy histogram at its native bin centers."""

    bin_centers: tuple[float, ...]
    density: tuple[float, ...]


class EnergyDiagnostics(Value):
    """Energy distributions and the producer's chain-specific BFMI values."""

    energy_hist: DensityHistogram
    energy_transition_hist: DensityHistogram
    bfmi: tuple[float, ...]


class ChainDiagnostics(Value):
    """Compact retained-chain measurements; plot series compose the report detail."""

    num_chains: int
    num_samples: int
    per_parameter: tuple[ParameterDiagnostics, ...]
    num_divergences: int | None = None
    divergence_rate: float | None = None
    tree_depth_mean: float | None = None
    tree_depth_max: int | None = None
    accept_prob_mean: float | None = None
    latent_accept_prob_mean: float | None = None
    parameter_accept_prob_mean: float | None = None
    energy: EnergyDiagnostics | None = None


type ConvergenceAssessmentSubject = ConvergenceSubject | Literal["recorded_parameter_chains"]


class ParameterConvergenceReport(Value):
    """Recorded-chain criteria cover parameters, not latent-path mixing."""

    scope: Literal["recorded_parameter_chains"] = "recorded_parameter_chains"
    checked: int
    status: Literal["passed", "failed", "not_evaluated"]
    messages: tuple[str, ...] = ()
    assessments: tuple[Assessment[ConvergenceAssessmentSubject, NumericCriterionEvidence], ...]


class ParetoKPoint(Value):
    """One PSIS influence measurement with its original row and scientific class."""

    rank: int
    timestep: int
    k: float | Literal["infinity", "-infinity", "undefined"]
    status: Literal["passed", "warning", "failed", "not_evaluated"]


class LOOPITPoint(Value):
    """A retained PIT value and its empirical and reference cumulative probabilities."""

    pit: float
    ecdf: float
    uniform: float


class LOODiagnostics(Value):
    """Exact-emission leave-one-measurement-row-out interpolation diagnostics."""

    elpd_loo: float
    p_loo: float
    se: float
    n_data_points: int
    observation_unit: Literal["measurement_row"] = "measurement_row"
    prediction_task: Literal["interpolation_given_other_measurements"] = (
        "interpolation_given_other_measurements"
    )
    likelihood_source: Literal["exact_emission_on_joint_particle_draws"] = (
        "exact_emission_on_joint_particle_draws"
    )
    n_bad_k: int | None = None
    n_warn_k: int | None = None
    pareto_warning_limit: float = 0.5
    pareto_failure_limit: float = 0.7


class ParticleMCMCEvidence(Value):
    """The production particle-MCMC target and its exact latent transition."""

    engine: Literal["marginal_particle_gibbs"] = "marginal_particle_gibbs"
    latent_transition: Literal["euler_maruyama"] = "euler_maruyama"


class PosteriorEstimate(Value):
    """A posterior estimate reports a mean and a credible interval with explicit semantics."""

    model_config = ConfigDict(allow_inf_nan=False)

    mean: float
    lower: float
    upper: float
    interval_kind: Literal["hdi", "equal_tail"]
    interval_mass: float = Field(
        gt=0, lt=1, description="Posterior probability mass of the interval."
    )

    @model_validator(mode="after")
    def validate_bounds(self) -> "PosteriorEstimate":
        if self.lower > self.upper:
            raise ValueError("Posterior interval lower bound must not exceed its upper bound")
        return self


class PosteriorMarginal(PosteriorEstimate):
    """A posterior marginal summarizes uncertainty in one scalar parameter and supplies its
    density plot.
    """

    parameter: str
    subject: ParameterRef
    x_values: tuple[float, ...]
    density: tuple[float, ...]
    sd: float = Field(ge=0)


class PosteriorPair(Value):
    """A posterior pair supplies joint samples of two parameters to visualize their dependence."""

    param_x: str
    subject_x: ParameterRef
    param_y: str
    subject_y: ParameterRef
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    divergent: tuple[bool, ...] | None = None


class PPCOverlay(Value):
    """A predictive overlay sets one indicator's observed values against simulated ones.

    It carries the predictive median and a few individual replicated series, the
    spaghetti plot of a visual predictive check.
    """

    indicator_id: IndicatorId
    observed: tuple[float | None, ...]
    median: tuple[float | None, ...]
    spaghetti_draws: tuple[tuple[float | None, ...], ...] = Field(default_factory=tuple)


class PPCTestStat(Value):
    """A predictive test statistic compares an observed summary with its distribution under
    replicated data.

    Provides the data for Gabry's ppc_stat plots: histogram of T(y_rep)
    with a vertical line at T(y_observed).
    """

    indicator_id: IndicatorId
    stat_name: Literal["mean", "sd", "min", "max"]
    observed_value: float
    rep_values: tuple[float, ...]
    p_value: float | None
    histogram: tuple[HistogramBin, ...]


class PosteriorPredictiveChecks(Value):
    """Posterior predictive checks report exact-model checks and their supporting plot data."""

    per_variable_warnings: tuple[
        Assessment[IndicatorCheckSubject, NumericCriterionEvidence], ...
    ] = Field(default_factory=tuple)
    checked: bool = False
    n_subsample: int = 0
    overlays: tuple[PPCOverlay, ...] = Field(default_factory=tuple)
    test_stats: tuple[PPCTestStat, ...] = Field(default_factory=tuple)


class ElboScoringDiagnostics(Value):
    """Retained native initialization measurements; never posterior evidence."""

    n_elbo_batch_evaluations: int
    n_elbo_screen_candidates: int
    n_elbo_refine_candidates: int
    best_elbo_candidate_index: int


class PathfinderStartDiagnostics(ElboScoringDiagnostics):
    """Retained native initialization measurements; never posterior evidence."""

    start_idx: int
    n_trajectory_points: int
    n_valid_iterates: int
    n_elbo_candidates: int
    n_lbfgs_iterations: int
    final_log_posterior: float
    best_elbo_this_start: float | None
    scipy_success: bool
    scipy_status: int


class PathfinderDiagnostics(Value):
    """Retained native initialization measurements; never posterior evidence."""

    n_pathfinder_starts: int
    n_pathfinder_starts_finite: int
    pathfinder_parallel_workers: int
    pathfinder_setup_seconds: float
    pathfinder_jax_compile_seconds: float
    pathfinder_jax_compile_batch_sizes: tuple[int, ...]
    pathfinder_runtime_seconds: float
    pathfinder_total_seconds: float
    best_pathfinder_elbo: float
    pathfinder_elbo: float
    pathfinder_elbo_min: float
    pathfinder_elbo_max: float
    pathfinder_elbo_spread: float
    pathfinder_elbos: tuple[float, ...]
    pathfinder_maxiter: int
    pathfinder_lbfgs_memory: int
    pathfinder_elbo_samples: int
    pathfinder_elbo_screen_samples: int
    pathfinder_elbo_refine_candidates: int
    pathfinder_elbo_candidate_batch_size: int
    pathfinder_per_start: tuple[PathfinderStartDiagnostics, ...]


class ParticleInitializationDiagnostics(Value):
    """Retained native initialization measurements; never posterior evidence."""

    pathfinder: PathfinderDiagnostics | None = None
    init_method: str | None = None
    pathfinder_sampling_mode: str | None = None
    pathfinder_init_scale: float | None = None
    prior_released_site_names: tuple[str, ...] | None = None
    prior_released_site_indices: tuple[int, ...] | None = None
    prior_release_scale: float | None = None


class ParticlePreconditionerDiagnostics(Value):
    """Proposal-scale setup, distinct from retained posterior measurements."""

    auto_preconditioner: bool | None = None
    auto_preconditioner_method: str | None = None
    auto_preconditioner_device: str | None = None
    auto_preconditioner_n_pathfinder_starts: int | None = None
    auto_preconditioner_n_pathfinder_starts_finite: int | None = None
    auto_preconditioner_best_pathfinder_elbo: float | None = None
    auto_preconditioner_pathfinder_elbo_spread: float | None = None
    auto_preconditioner_maxiter: int | None = None


class ParameterWarmupDiagnostics(Value):
    """Timing and ownership of proposal initialization and preconditioning."""

    pathfinder_ran: bool
    pathfinder_run_count: int
    pathfinder_consumers: tuple[str, ...]
    init_source: str
    preconditioner_source: str
    auto_preconditioner_method: str
    dim: int
    duration_seconds: float
    init_scale: float
    pathfinder_init_scale: float | None
    pathfinder_setup_seconds: float | None = None
    pathfinder_jax_compile_seconds: float | None = None
    pathfinder_runtime_seconds: float | None = None
    pathfinder_total_seconds: float | None = None
    pathfinder_jax_compile_batch_sizes: tuple[int, ...] | None = None


class ParticleSamplerDiagnostics(Value):
    """Typed exact-sampler settings and transition telemetry from the native producer."""

    latent_kernel: str
    latent_smoother: str
    latent_smoother_algorithm: str
    latent_smoother_family: str
    latent_smoother_selection: str
    latent_smoother_parallel: bool
    latent_delta: float
    parameter_kernel: str
    mcmc_phase_seconds: float
    num_warmup: int
    num_samples: int
    num_chains: int
    n_particles: int
    n_parameter_particles: int
    parameter_proposal: str
    latent_backward_sampling: bool
    amala_delta_init: float
    amala_delta_min: float
    amala_delta_max: float
    amala_target_accept: float
    amala_adaptation_window: int
    amala_adaptation_tolerance: float
    amala_adaptation_rho: float
    amala_adaptation_rho_min: float
    amala_adaptation_gamma: float
    amala_delta_adapted: bool
    amala_kappa: float
    amala_grad_clip: float | None
    dsmc_leaf_proposal: str
    latent_transition_kind: str
    diagnostic_metrics_all: bool
    diagnostic_metrics: tuple[str, ...]
    param_step_size_initial: float
    param_step_size_min: float
    param_step_size_max: float
    param_target_accept: float
    adaptation_scheme: str
    parameter_preconditioned: bool
    diagnostic_summary_phase: str
    parameter_accept_rate: float
    latent_update_fraction: float
    latent_frozen_fraction: float
    latent_block_coords: int | None
    initial_param_step_size: tuple[float, ...]
    final_param_step_size: tuple[float, ...]
    latent_init_method: str
    latent_sign_flip_moves: bool | None = None
    chain_post_warmup_complete_log_posterior_mean: tuple[float, ...]
    latent_move_rms_mean: float | None = None
    parameter_jump_rms_mean: float | None = None
    reference_path_hit_rate_mean: float | None = None
    selected_particle_unique_count_mean: float | None = None
    amala_grad_norm_mean: float | None = None
    amala_grad_norm_max: float | None = None
    parameter_warmup: ParameterWarmupDiagnostics
    initialization: ParticleInitializationDiagnostics
    preconditioner: ParticlePreconditionerDiagnostics


class TemperingDiagnostics(Value):
    """Retained tempering telemetry, separate from evidence of a production engine."""

    n_levels: int
    n_particles: int
    accept_rates: tuple[float, ...]
    beta_schedule: tuple[float, ...]
    ess_history: tuple[float, ...]
