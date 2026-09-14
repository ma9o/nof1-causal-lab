"""NumPyro State-Space Model.

Bayesian State-Space Model definition using NumPyro.
This module defines the probabilistic model only — inference is in inference.py.

Supports:
- Time-series trajectories
- Any noise family (Gaussian, Poisson, Student-t, Gamma)
"""

from __future__ import annotations

from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import MultivariateNormal

from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.dynamical_model import continuous_state_evolution
from nof1_causal_lab.models.ssm.execution.parameters import assemble_model_matrices, sample_sites

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx import StochasticContinuousTimeStateEvolution

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.compile.bindings import CompiledParameterBinding
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
)
from nof1_causal_lab.models.ssm.execution.contracts import (
    LikelihoodExtraParams,
    MeasurementParams,
)
from nof1_causal_lab.models.ssm.likelihood_extra_params import (
    assemble_sampled_extra_params,
)
from nof1_causal_lab.models.ssm.parameter_layout import SSMParameterLayout
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    build_prior_runtime_bundle,
    likelihood_sites,
)


@jax.custom_vjp
def _nan_safe_ll(ll):
    """Return ll if finite, else -1e30. Gradient is zeroed when ll is non-finite."""
    return jnp.where(jnp.isfinite(ll), ll, -1e30)


def _nan_safe_ll_fwd(ll):
    y = _nan_safe_ll(ll)
    return y, jnp.isfinite(ll)


def _nan_safe_ll_bwd(is_finite, g):
    return (jnp.where(is_finite, jnp.nan_to_num(g, nan=0.0), 0.0),)


_nan_safe_ll.defvjp(_nan_safe_ll_fwd, _nan_safe_ll_bwd)


class SSMModel:
    """NumPyro state-space model definition.

    Defines the probabilistic model for Bayesian state-space models.
    Inference is handled externally by ssm.inference.fit().

    Features:
    - Continuous-time dynamics via stochastic differential equations
    - Exact particle likelihoods; Gaussian approximations only for sampler initialization
    """

    def __init__(
        self,
        spec: ModelSpec,
        priors: dict[str, dist.Distribution] | None = None,
        prior_runtime_bundle: PriorRuntimeBundle | None = None,
    ):
        """Initialize state-space model.

        Args:
            spec: Statistical model specification
            priors: Optional execution laws; otherwise compile the priors owned by ModelSpec.
        """
        self.spec = spec
        self.priors = priors
        self._parameter_layout = SSMParameterLayout.from_spec(spec)
        self._artifact_cache: dict[tuple[object, ...], object] = {}
        self.observation_support: ObservationSupportRuntime | None = None
        self.transition_inputs: jnp.ndarray | None = None
        self._prior_runtime_bundle = prior_runtime_bundle
        self._prior_site_index = (
            {site.name: site for site in prior_runtime_bundle.registry}
            if prior_runtime_bundle is not None
            else None
        )

    def get_cached_artifact[T](
        self,
        cache_key: tuple[object, ...],
        factory: Callable[[], T],
    ) -> T:
        """Construct an artifact once per model instance and reuse it afterwards."""
        if cache_key not in self._artifact_cache:
            self._artifact_cache[cache_key] = factory()
        return cast("T", self._artifact_cache[cache_key])

    def set_observation_support(
        self, observation_support: ObservationSupportRuntime | None
    ) -> None:
        """Attach prepared observation-support metadata and invalidate backend caches."""
        self.observation_support = observation_support
        self._artifact_cache = {
            key: value
            for key, value in self._artifact_cache.items()
            if not (isinstance(key, tuple) and key and key[0] == "backend")
        }

    def set_transition_inputs(self, transition_inputs: jnp.ndarray | None) -> None:
        """Attach prepared known-input trajectories aligned to transition intervals."""
        self.transition_inputs = transition_inputs

    @property
    def vector_field(self):
        """Unified dynamics representation as a :class:`VectorField`.

        The vector field derives from the scientific mechanisms and is what consumers
        (``compute_steady_state``, ``simulate``, the per-step linearisation in
        the IEKS/Laplace warmup backend, …) all consume uniformly.
        """

        def _build():
            from nof1_causal_lab.models.ssm.dynamics.spec import compile_dynamics

            return compile_dynamics(numeric.dynamics_components(self.spec)).vector_field

        return self.get_cached_artifact(("vector_field",), _build)

    @cached_property
    def parameter_bindings(self) -> list[CompiledParameterBinding]:
        """Scientific coordinates derived from this model's immutable source."""
        from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

        return parameter_bindings(self.spec)[0]

    @property
    def parameter_layout(self) -> SSMParameterLayout:
        """Return the derived parameter layout for this model."""
        return self._parameter_layout

    def get_prior_runtime_bundle(self) -> PriorRuntimeBundle:
        """Return canonical prior runtime state for this model instance."""
        if self._prior_runtime_bundle is None:
            from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors

            priors = self.priors
            if priors is None:
                priors, _, _ = compile_priors(
                    self.spec,
                    edge_lag_days=numeric.edge_lag_days(self.spec),
                )
            self._prior_runtime_bundle = build_prior_runtime_bundle(self.spec, priors)
            self._prior_site_index = {
                site.name: site for site in self._prior_runtime_bundle.registry
            }
        return self._prior_runtime_bundle

    def _prior_distribution(self, site_name: str) -> dist.Distribution:
        """Resolve a sample-site prior from canonical runtime semantics."""
        runtime = self.get_prior_runtime_bundle()
        assert self._prior_site_index is not None
        site = self._prior_site_index.get(site_name)
        if site is None:
            raise ValueError(f"Prior runtime bundle has no site named {site_name!r}")
        return runtime.priors[site_name]

    def _sample_likelihood_extra_params(self, spec: ModelSpec) -> LikelihoodExtraParams:
        """Sample the shared likelihood-site catalog and assemble its semantics."""
        return assemble_sampled_extra_params(
            spec, sample_sites(likelihood_sites(spec), self._prior_distribution)
        )

    def _sample_parameters(self) -> dict[str, jnp.ndarray]:
        """Sample declared sites and emit the canonical scientific matrices."""
        sites = chain.from_iterable(
            block.iter_sites() for block in numeric.parameter_blocks(self.spec)
        )
        matrices, min_eigenvalue = assemble_model_matrices(
            self.spec, sample_sites(sites, self._prior_distribution)
        )
        for name, value in matrices.items():
            # Empty input/static-factor blocks have no public deterministic site.
            if name not in {"input_effect", "static_state_sds"} or value.size:
                numpyro.deterministic(name, value)
        numpyro.factor(
            "t0_correlation_positive_definite",
            jnp.where(
                min_eigenvalue > INITIAL_STATE_COV_MIN_EIGENVALUE,
                0.0,
                -1e6 * (INITIAL_STATE_COV_MIN_EIGENVALUE - min_eigenvalue),
            ),
        )
        return matrices

    def _sample_runtime_dynamics(
        self,
        diffusion_cov: jnp.ndarray,
        input_effect: jnp.ndarray,
    ) -> StochasticContinuousTimeStateEvolution:
        """Sample vector-field parameters inside the NumPyro trace."""
        from nof1_causal_lab.models.ssm.dynamics.spec import compile_dynamics

        compiled = compile_dynamics(numeric.dynamics_components(self.spec))
        return continuous_state_evolution(
            vector_field=compiled.vector_field,
            vf_params=compiled.sample_params(self._prior_distribution),
            diffusion_cov=diffusion_cov,
            input_effect=input_effect,
        )

    def model(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        likelihood_backend=None,
    ) -> None:
        """NumPyro model function.

        Args:
            observations: (N, n_manifest) observed data
            times: (N,) observation times
            likelihood_backend: Laplace likelihood backend instance. Required —
                construct it in the inference warmup layer.
        """
        if likelihood_backend is None:
            raise ValueError(
                "likelihood_backend is required. Construct it in the inference warmup layer."
            )

        spec = self.spec
        sampled = self._sample_parameters()

        diffusion_chol = sampled["diffusion"]
        input_effect = sampled["input_effect"]
        lambda_mat = sampled["lambda"]
        manifest_means = sampled["manifest_means"]
        t0_means = sampled["t0_means"]

        diffusion_cov = diffusion_chol @ diffusion_chol.T
        manifest_cov = sampled["manifest_cov"]
        t0_cov = sampled["t0_cov"]
        extra_params = self._sample_likelihood_extra_params(spec)
        dynamics = self._sample_runtime_dynamics(diffusion_cov, input_effect)

        meas_params = MeasurementParams(
            lambda_mat=lambda_mat,
            manifest_means=manifest_means,
            manifest_cov=manifest_cov,
        )

        time_intervals = jnp.diff(times, prepend=times[0])
        time_intervals = time_intervals.at[0].set(MIN_DT)

        init = MultivariateNormal(loc=t0_means, covariance_matrix=t0_cov)
        lnc = likelihood_backend.compute_log_likelihood(
            dynamics,
            meas_params,
            init,
            observations,
            time_intervals,
            extra_params=extra_params or None,
            transition_inputs=self.transition_inputs,
        )

        # lnc is (T,) cumulative log-normalizing constants from the filter.
        # lnc[-1] = total log p(y|θ).
        # diff(lnc) exposes per-timestep contributions to the initialization
        # objective. Reported LOO uses emission factors on joint particle draws.
        if lnc.ndim == 0:
            total_ll = _nan_safe_ll(lnc)
            numpyro.factor("log_likelihood", total_ll)
        else:
            total_ll = _nan_safe_ll(lnc[-1])
            numpyro.factor("log_likelihood", total_ll)
            ll_per_timestep = jnp.diff(lnc, prepend=0.0)
            numpyro.deterministic("ll_per_timestep", ll_per_timestep)
