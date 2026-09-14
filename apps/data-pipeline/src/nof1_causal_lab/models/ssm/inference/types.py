"""Inference result and artifact types."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import jax.numpy as jnp

from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
from nof1_causal_lab.artifacts.posterior import PosteriorDrawsInfo
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    build_energy_diagnostics as _build_energy_diagnostics,
)
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    build_rank_histograms as _build_rank_histograms,
)
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    build_trace_data as _build_trace_data,
)
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    compute_posterior_marginals,
    compute_posterior_pairs,
)
from nof1_causal_lab.models.ssm.inference.shared import _filter_public_samples

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from nof1_causal_lab.json_types import JsonObject
    from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult

logger = logging.getLogger(__name__)


InferenceMethod = Literal["marginal_particle_gibbs"]
type ArviZArrayMapping = dict[str, Any]


def _arviz_idata_from_posterior(
    posterior: ArviZArrayMapping,
    *,
    log_likelihood: ArviZArrayMapping | None = None,
) -> Any:
    """Build ArviZ inference data from array-valued posterior dictionaries."""
    import arviz_base as az_base
    import numpy as np

    payload = {"posterior": {name: np.asarray(values) for name, values in posterior.items()}}
    if log_likelihood is not None:
        payload["log_likelihood"] = {
            name: np.asarray(values) for name, values in log_likelihood.items()
        }
    return az_base.from_dict(payload)


class InferenceDiagnostics(TypedDict, total=False):
    """Known live inference diagnostics shared by MAP and particle methods."""

    mcmc: TrajectoryMCMCResult
    public_sites: list[str]
    likelihood_backend: object
    observation_log_probs: jnp.ndarray
    latent_posterior_summary: dict[str, jnp.ndarray]
    warmup_latent_paths: jnp.ndarray
    all_latent_paths: jnp.ndarray
    beta_schedule: jnp.ndarray
    ess_history: jnp.ndarray
    accept_rates: jnp.ndarray
    n_levels: int
    n_csmc_particles: int
    optimizer: str
    success: bool
    status: int
    n_iters: int
    n_function_evals: int
    objective_at_mode: float
    mode_log_posterior: float
    mode_log_likelihood: float
    mode_log_prior: float
    mode_grad_norm: float | None
    mode_inner_solver: str
    mode_inner_iterations: int
    mode_inner_accepted_steps: int
    mode_inner_rel_change: float
    mode_inner_damping: float
    mode_inner_step_alpha: float
    mode_inner_step_norm: float
    mode_inner_log_joint_gain: float | None
    mode_inner_laplace_logdet: float
    mode_inner_min_chol_diag: float
    init_log_posterior_best: float
    n_init_samples: int
    n_ieks_iters: int
    parameter_covariance: jnp.ndarray | NDArray
    covariance_diag: list[float]
    compute_parameter_hessian: bool
    parameter_posterior_strategy: str
    parameter_covariance_method: str
    hessian_condition_number: float | None
    parameter_hessian_min_eig: float | None
    parameter_hessian_max_eig: float | None
    hessian_jitter: float
    marginal_particle_gibbs: JsonObject
    marginal_particle_gibbs_phase_extra_fields: dict[str, dict[str, jnp.ndarray]]
    chain_complete_log_posterior_history: jnp.ndarray
    warmup_complete_log_posterior_history: jnp.ndarray
    all_complete_log_posterior_history: jnp.ndarray


@dataclass(frozen=True, slots=True)
class ParticleMCMCEvidence:
    """Proof that samples came from the production particle-MCMC target."""

    engine: Literal["marginal_particle_gibbs"] = "marginal_particle_gibbs"
    latent_transition: Literal["euler_maruyama"] = "euler_maruyama"


@dataclass
class WarmupProposal:
    """Laplace/IEKS approximation usable only for sampler initialization."""

    _samples: dict[str, jnp.ndarray]
    diagnostics: InferenceDiagnostics = field(default_factory=dict)
    method: Literal["map"] = field(init=False, default="map")

    def get_samples(self) -> dict[str, jnp.ndarray]:
        """Return approximate parameter draws for warmup consumers."""
        return self._samples


@dataclass(frozen=True)
class JointPosteriorDraws:
    """Aligned parameter and latent-trajectory draws; the leading axis identifies one joint draw."""

    parameters: dict[str, jnp.ndarray]
    latent_paths: jnp.ndarray | None = None
    state_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        counts = set()
        for values in self.parameters.values():
            if values.ndim < 1:
                raise ValueError("Posterior parameters require a leading draw axis")
            counts.add(values.shape[0])
        if self.latent_paths is not None:
            if self.latent_paths.ndim != 3:
                raise ValueError("Posterior latent paths require draw, time, and state axes")
            counts.add(self.latent_paths.shape[0])
            if self.state_ids and (
                len(self.state_ids) != self.latent_paths.shape[2]
                or len(set(self.state_ids)) != len(self.state_ids)
            ):
                raise ValueError("Latent state IDs must label the state axis exactly")
        if len(counts) > 1:
            raise ValueError("Posterior parameters and latent paths must share the draw axis")

    def describe(self) -> PosteriorDrawsInfo:
        counts = [values.shape[0] for values in self.parameters.values()]
        if self.latent_paths is not None:
            counts.append(self.latent_paths.shape[0])
        if not counts:
            raise ValueError("Posterior contains no draws")
        return PosteriorDrawsInfo(
            n_draws=counts[0],
            state_ids=self.state_ids,
            parameter_shapes={
                name: list(values.shape[1:]) for name, values in self.parameters.items()
            },
            latent_shape=(self.latent_paths.shape[1], self.latent_paths.shape[2])
            if self.latent_paths is not None
            else None,
        )


@dataclass
class ParticleMCMCPosterior:
    """Joint posterior draws produced by the invariant particle-MCMC engine."""

    draws: JointPosteriorDraws
    diagnostics: InferenceDiagnostics = field(default_factory=dict)
    evidence: ParticleMCMCEvidence = field(default_factory=ParticleMCMCEvidence)
    method: Literal["marginal_particle_gibbs"] = field(
        init=False, default="marginal_particle_gibbs"
    )

    def get_samples(self) -> dict[str, jnp.ndarray]:
        """Return parameter draws aligned with the retained latent trajectories."""
        return self.draws.parameters

    def get_inference_diagnostics(self) -> JsonObject:
        """Report engine telemetry without a public sampler-specific schema."""
        result: JsonObject = {}
        mcmc = self.get_mcmc_diagnostics()
        if mcmc is not None:
            result["mcmc"] = mcmc
        smc = self.get_smc_diagnostics()
        if smc is not None:
            result["smc"] = smc
        if "marginal_particle_gibbs" in self.diagnostics:
            result["marginal_particle_gibbs"] = self.diagnostics["marginal_particle_gibbs"]
        return result

    def get_mcmc_diagnostics(self) -> JsonObject | None:
        """Extract JSON-serializable MCMC diagnostics."""
        mcmc = self.diagnostics.get("mcmc")
        if mcmc is None:
            return None

        from numpyro.diagnostics import summary as numpyro_summary

        result: JsonObject = {}

        chain_samples = mcmc.get_samples(group_by_chain=True)
        public_sites = self.diagnostics.get("public_sites")
        if public_sites is not None:
            chain_samples = _filter_public_samples(chain_samples, set(public_sites))

        summ = numpyro_summary(chain_samples)
        import arviz_base as az_base
        from arviz_stats.sampling_diagnostics import ess, mcse

        if getattr(mcmc, "backend", None) in {
            "aux_kalman_mcmc",
            "pit_particle_mgrad",
            "marginal_particle_gibbs",
        }:
            idata = _arviz_idata_from_posterior(chain_samples)
        else:
            idata = az_base.from_numpyro(mcmc)
        ess_tail = ess(idata, method="tail")
        mcse_mean = mcse(idata, method="mean")

        import math

        import numpy as np

        def metric(values, indices):
            value = float(np.asarray(values)[indices])
            return value if math.isfinite(value) else None

        per_param: list[JsonObject] = []
        for name, stats in summ.items():
            for indices in np.ndindex(np.shape(stats["r_hat"])):
                coordinate = ParameterCoordinate(site_name=name, indices=indices)
                per_param.append(
                    {
                        "parameter": coordinate.label,
                        "coordinate": coordinate.model_dump(mode="json"),
                        "r_hat": metric(stats["r_hat"], indices),
                        "ess_bulk": metric(stats["n_eff"], indices),
                        "ess_tail": metric(ess_tail[name].values, indices)
                        if name in ess_tail
                        else None,
                        "mcse_mean": metric(mcse_mean[name].values, indices)
                        if name in mcse_mean
                        else None,
                    }
                )
        result["per_parameter"] = list(per_param)

        extra = mcmc.get_extra_fields()
        if "diverging" in extra:
            div = extra["diverging"]
            result["num_divergences"] = int(jnp.sum(div))
            result["divergence_rate"] = float(jnp.mean(div))
        if "num_steps" in extra:
            steps = extra["num_steps"]
            result["tree_depth_mean"] = float(jnp.mean(steps))
            result["tree_depth_max"] = int(jnp.max(steps))
        if "accept_prob" in extra:
            ap = extra["accept_prob"]
            result["accept_prob_mean"] = float(jnp.mean(ap))
        if "latent_accept_prob" in extra:
            ap = extra["latent_accept_prob"]
            result["latent_accept_prob_mean"] = float(jnp.mean(ap))
        if "parameter_accept_prob" in extra:
            ap = extra["parameter_accept_prob"]
            result["parameter_accept_prob_mean"] = float(jnp.mean(ap))
        if "energy" in extra:
            energy = extra["energy"]
            n_ch = int(mcmc.num_chains)
            if n_ch > 1 and energy.ndim == 1 and energy.shape[0] % n_ch == 0:
                energy = energy.reshape(n_ch, -1)
            result["energy"] = _build_energy_diagnostics(energy)

        result["num_chains"] = int(mcmc.num_chains)
        result["num_samples"] = int(mcmc.num_samples)

        if chain_samples is not None:
            result["trace_data"] = list(_build_trace_data(chain_samples, max_points=200))
            result["rank_histograms"] = list(_build_rank_histograms(chain_samples, n_bins=20))

        return result

    def get_smc_diagnostics(self) -> JsonObject | None:
        """Extract JSON-serializable SMC diagnostics."""
        beta = self.diagnostics.get("beta_schedule")
        if beta is None:
            return None

        return {
            "beta_schedule": [float(b) for b in beta],
            "ess_history": [float(e) for e in self.diagnostics.get("ess_history", [])],
            "accept_rates": [float(a) for a in self.diagnostics.get("accept_rates", [])],
            "n_levels": int(self.diagnostics.get("n_levels", len(beta))),
            "n_particles": int(self.diagnostics.get("n_csmc_particles", 0)),
        }

    def get_loo_diagnostics(self, *, observations: jnp.ndarray) -> JsonObject | None:
        """PSIS leave-one-measurement-row-out from the joint particle posterior.

        The exact emission factors condition on each sampled latent state. The
        held-out row is interpolated using all other rows, including future
        measurements. This does not estimate leave-future-out forecast skill.
        """
        from arviz_stats.loo import loo

        factors = self.diagnostics["observation_log_probs"]
        observed_rows = jnp.any(~jnp.isnan(observations), axis=1)
        factors = factors[:, :, observed_rows]
        if factors.shape[1] == 0 or factors.shape[2] == 0:
            return None
        mcmc = self.diagnostics["mcmc"]
        posterior = mcmc.get_samples(group_by_chain=True)
        idata = _arviz_idata_from_posterior(
            posterior,
            log_likelihood={"measurement_row": factors},
        )
        estimate = loo(idata)
        result: JsonObject = {
            "elpd_loo": float(estimate.elpd),
            "p_loo": float(estimate.p),
            "se": float(estimate.se),
            "n_data_points": int(estimate.n_data_points),
            "observation_unit": "measurement_row",
            "prediction_task": "interpolation_given_other_measurements",
            "likelihood_source": "exact_emission_on_joint_particle_draws",
        }
        if estimate.pareto_k is not None:
            values = jnp.asarray(estimate.pareto_k.values)
            result["pareto_k"] = [float(value) for value in values]
            result["n_bad_k"] = int((values > 0.7).sum())
        return result

    def get_posterior_marginals(self, n_bins: int = 50) -> list[JsonObject]:
        """Compute marginal posterior density data for visualization."""
        return compute_posterior_marginals(self.draws.parameters, n_bins)

    def get_posterior_pairs(self, max_params: int = 6, max_samples: int = 200) -> list[JsonObject]:
        """Compute pairwise scatter data for joint posterior visualization."""
        return compute_posterior_pairs(
            self.draws.parameters,
            self.diagnostics.get("mcmc"),
            max_params,
            max_samples,
        )
