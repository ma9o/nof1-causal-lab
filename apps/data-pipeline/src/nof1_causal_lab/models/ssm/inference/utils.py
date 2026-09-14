"""Shared utilities for inference backends.

Particle inference and MAP initialization share NumPyro's prior replay and
transformations. Application metadata supports initialization proposals and
reporting; it never reconstructs the prior density or deterministic model sites.
"""

from __future__ import annotations

import functools
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from dynestyx.inference.particle_runtime import Parameterization, prepare_parameterization
from numpyro import handlers

from nof1_causal_lab.models.ssm.constants import INTERNAL_DIAGNOSTIC_SITES, MIN_DT
from nof1_causal_lab.models.ssm.execution.dynamical_model import assemble_likelihood_inputs
from nof1_causal_lab.models.ssm.inference.shared import _filter_public_samples, _trace_public_sites
from nof1_causal_lab.models.ssm.parameterization import (
    build_site_registry,
)


class SiteInfoEntry(TypedDict):
    """Typed trace metadata for one unconstrained NumPyro sample site."""

    shape: tuple[int, ...]
    distribution: dist.Distribution
    transform: dist.transforms.Transform
    value: jnp.ndarray


type SiteInfo = dict[str, SiteInfoEntry]


# ---------------------------------------------------------------------------
# Model tracing
# ---------------------------------------------------------------------------


def _discover_sites(
    model, observations, times, rng_key, likelihood_backend, reparam=None
) -> SiteInfo:
    """Trace model once to discover sample sites (names, shapes, transforms).

    Site discovery is structural: it only needs the latent sample/deterministic
    sites emitted by ``model.model``. Tracing through the real likelihood backend
    can trigger large JAX/XLA compilations for support-aware Laplace before
    inference even starts, so discovery always replays the model with the dummy
    backend instead.
    """
    _ = likelihood_backend
    model_fn = functools.partial(model.model, likelihood_backend=_DummyLikelihoodBackend())
    if reparam is not None:
        model_fn = handlers.reparam(model_fn, config=reparam)
    with handlers.seed(rng_seed=rng_key):
        trace = handlers.trace(model_fn).get_trace(observations, times)

    site_info: SiteInfo = {}
    for name, site in trace.items():
        if (
            site["type"] == "sample"
            and not site.get("is_observed", False)
            and name not in INTERNAL_DIAGNOSTIC_SITES
        ):
            d = site["fn"]
            site_info[name] = {
                "shape": site["value"].shape,
                "distribution": d,
                "transform": dist.transforms.biject_to(d.support),
                "value": site["value"],
            }
    return site_info


# ---------------------------------------------------------------------------
# Pure-JAX deterministic site assembly
# ---------------------------------------------------------------------------


class _DummyLikelihoodBackend:
    """Dummy backend for model replay — returns zero log-likelihood."""

    checkpoint_loglik = False

    def compute_log_likelihood(self, *_args, **_kwargs):
        return jnp.array(0.0)


def prepare_model_parameters(model, observations, times, trace_key, reparam):
    """Use NumPyro's prior replay and transformations for every inference path."""
    prior_model = functools.partial(model.model, likelihood_backend=_DummyLikelihoodBackend())
    public_sites = _trace_public_sites(prior_model, observations, times)
    site_info = _discover_sites(
        model, observations, times, trace_key, _DummyLikelihoodBackend(), reparam=reparam
    )
    if reparam is not None:
        prior_model = handlers.reparam(prior_model, config=reparam)
    parameters = prepare_parameterization(
        prior_model,
        trace_key,
        model_args=(observations, times),
        initial_values={name: info["value"] for name, info in site_info.items()},
    )
    return parameters, site_info, public_sites


def extract_constrained_samples(
    particles: jnp.ndarray,
    parameters: Parameterization,
    public_sites: set[str],
) -> dict[str, jnp.ndarray]:
    """Replay the prior program and retain its authored parameter/deterministic sites."""
    return _filter_public_samples(jax.vmap(parameters.constrain)(particles), public_sites)


# ---------------------------------------------------------------------------
# Differentiable evaluators
# ---------------------------------------------------------------------------


def _build_eval_fns(
    model,
    observations,
    times,
    parameters: Parameterization,
    likelihood_backend,
    *,
    include_likelihood_aux: bool = False,
    runtime_observations_times: bool = False,
):
    """Build differentiable functions for log-likelihood and log-prior.

    Args:
        likelihood_backend: Likelihood backend instance to use for evaluation.
        parameters: The same library-owned prior replay used by particle inference.

    Returns:
        When ``runtime_observations_times=False``:
        log_lik_fn(z) -> scalar log p(y|theta)
        log_prior_unc_fn(z) -> scalar log p_unc(z) = log p(T(z)) + log|J|
        log_lik_with_aux_fn(z) -> (scalar log p(y|theta), aux pytree), when requested

        When ``runtime_observations_times=True``:
        log_lik_fn(z, observations, times) -> scalar log p(y|theta)
        log_prior_unc_fn(z) -> scalar log p_unc(z) = log p(T(z)) + log|J|
        log_lik_with_aux_fn(z, observations, times) -> (scalar log p(y|theta), aux pytree),
        when requested
    """
    runtime_registry = build_site_registry(model.spec)

    def _evaluate_likelihood(
        z,
        eval_observations,
        eval_times,
        latent_mode_init,
        *,
        with_aux: bool,
    ):
        original_samples = parameters.constrain(z)
        dynamics, measurement_params, initial_state, extra_params = assemble_likelihood_inputs(
            original_samples,
            model.spec,
            registry=runtime_registry,
        )
        time_intervals = (
            jnp.diff(eval_times, prepend=eval_times[0])
            .at[0]
            .set(jnp.asarray(MIN_DT, dtype=eval_times.dtype))
        )
        backend_fn = (
            likelihood_backend.compute_log_likelihood_with_aux
            if with_aux
            else likelihood_backend.compute_log_likelihood
        )
        backend_kwargs = {
            "extra_params": extra_params,
        }
        if latent_mode_init is None:
            evaluated = backend_fn(
                dynamics,
                measurement_params,
                initial_state,
                eval_observations,
                time_intervals,
                **backend_kwargs,
            )
        else:
            evaluated = backend_fn(
                dynamics,
                measurement_params,
                initial_state,
                eval_observations,
                time_intervals,
                **backend_kwargs,
                latent_mode_init=latent_mode_init,
            )
        aux = None
        if with_aux:
            lnc, aux = evaluated
        else:
            lnc = evaluated
        total_ll = lnc if lnc.ndim == 0 else lnc[-1]
        total_ll = jnp.where(jnp.isfinite(total_ll), total_ll, -jnp.inf)
        return (total_ll, aux) if with_aux else total_ll

    def _likelihood_function(*, with_aux: bool):
        if runtime_observations_times:

            def _runtime(z, runtime_observations, runtime_times, latent_mode_init=None):
                return _evaluate_likelihood(
                    z,
                    runtime_observations,
                    runtime_times,
                    latent_mode_init,
                    with_aux=with_aux,
                )

            return _runtime

        def _bound(z, latent_mode_init=None):
            return _evaluate_likelihood(
                z,
                observations,
                times,
                latent_mode_init,
                with_aux=with_aux,
            )

        return _bound

    log_lik_base = _likelihood_function(with_aux=False)
    log_lik_fn = (
        jax.checkpoint(log_lik_base) if likelihood_backend.checkpoint_loglik else log_lik_base
    )

    # The aux payload can include latent-mode state reused across outer evaluations.
    # Rematerializing that path leaks tracers through the returned aux tree.
    log_lik_with_aux_fn = _likelihood_function(with_aux=True)

    if include_likelihood_aux:
        return log_lik_fn, parameters.log_prior, log_lik_with_aux_fn
    return log_lik_fn, parameters.log_prior
