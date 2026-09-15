"""Compile-stable predictive runtime for the model's current uncertainty.

Builds prior predictive samples directly from compiled prior semantics or
native NumPyro priors without tracing back through ``SSMModel.model()``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from nof1_causal_lab.artifacts.expressions import expression_coefficients
from nof1_causal_lab.artifacts.likelihood import DistributionFamily
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.predictive_simulation import (
    sample_model_observations,
)
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.dynamics.serialization import dynamics_spec_to_dict
from nof1_causal_lab.models.ssm.dynamics.simulator import SimulationConfig, simulate_model_path
from nof1_causal_lab.models.ssm.dynamics.spec import (
    compile_dynamics,
)
from nof1_causal_lab.models.ssm.execution.dynamical_model import build_dynamical_model
from nof1_causal_lab.models.ssm.execution.observation_families import (
    any_family_needs_level_metadata,
)
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    assemble_deterministics_from_registry,
    assemble_extra_params_from_registry,
    build_site_registry,
    sample_prior_parameters,
)

if TYPE_CHECKING:
    import dynestyx as dsx

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.dynamics.spec import CompiledDynamics, DynamicsSpec

logger = logging.getLogger(__name__)


class PredictiveKeys(NamedTuple):
    """Independent random streams for one predictive simulation."""

    parameters: jax.Array
    latents: jax.Array
    observations: jax.Array


def predictive_keys(seed: int) -> PredictiveKeys:
    """Derive independent parameter, latent, and observation streams."""
    parameter_key, latent_key, observation_key = random.split(random.PRNGKey(seed), 3)
    return PredictiveKeys(parameter_key, latent_key, observation_key)


def _ensure_discrete_metadata(spec: ModelSpec) -> None:
    """Require hydrated level counts before sampling discrete emissions."""
    needs_levels = any_family_needs_level_metadata(numeric.observation_families(spec))
    if needs_levels and numeric.observation_level_counts(spec) is None:
        raise ValueError(
            "Prior predictive for ordered/categorical emissions requires hydrated "
            "manifest_level_counts."
        )


def _assemble_extra_params_batched(
    spec: ModelSpec,
    constrained_samples: dict[str, jnp.ndarray],
    registry,
    *,
    n_draws: int,
) -> dict[str, jnp.ndarray]:
    """Assemble per-draw observation/process hyperparameters."""
    if not any(site.assembly_group == "likelihood" for site in registry):
        return {}

    def _assemble_one(draw_idx):
        sampled_values = {
            site_name: values[draw_idx] for site_name, values in constrained_samples.items()
        }
        return assemble_extra_params_from_registry(spec, sampled_values, registry)

    return jax.vmap(_assemble_one)(jnp.arange(n_draws, dtype=jnp.int32))


def _ensure_gaussian_process_diffusion(spec: ModelSpec) -> None:
    non_gaussian = [
        str(dist.value if isinstance(dist, DistributionFamily) else dist)
        for dist in numeric.diffusion_families(spec)
        if DistributionFamily(dist) != DistributionFamily.GAUSSIAN
    ]
    if non_gaussian:
        raise ValueError(
            "Vector-field prior predictive simulation currently requires Gaussian process "
            f"diffusion; got {non_gaussian}."
        )


def _predictive_models(
    spec: ModelSpec, samples, times, *, dynamics: DynamicsSpec | None = None
) -> dsx.DynamicalModel:
    """Batch the same model constructor used by the particle target."""
    return eqx.filter_vmap(
        lambda draw: build_dynamical_model(spec, draw, t0=times[0], dynamics=dynamics)
    )(samples)


# Refine the SDE step using declared relaxation rates, including fixed rates.
# This controls the linear restoring contribution; it is not a global stability
# certificate for a composed nonlinear drift. Diffrax evaluates the full
# expression at every step. Scientific coefficient metadata identifies rates;
# numerical sample-site names carry no meaning.
_SDE_CFL_SAFETY = 0.25
_SDE_MAX_STEPS = 16384
_PREDICTIVE_MICROBATCH_SIZE = 32
_LATENT_CACHE_MAX_ENTRIES = 2
_LATENT_CACHE_ENGINE_VERSION = 3
_latent_cache: OrderedDict[str, jax.Array] = OrderedDict()
_latent_cache_lock = threading.Lock()


def _update_array_digest(digest: Any, label: str, value: Any) -> None:
    array = np.asarray(jax.device_get(value))
    digest.update(label.encode())
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))


def _prior_predictive_latent_cache_key(
    dynamics: DynamicsSpec,
    vf_params: Any,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    rng_key: jax.Array,
) -> str:
    """Fingerprint only inputs that can change the latent trajectories."""
    digest = hashlib.sha256()
    digest.update(f"latent-cache-v{_LATENT_CACHE_ENGINE_VERSION}".encode())
    digest.update(
        json.dumps(
            dynamics_spec_to_dict(dynamics),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(jax.default_backend().encode())
    for leaf_index, leaf in enumerate(jax.tree.leaves(vf_params)):
        _update_array_digest(digest, f"vf:{leaf_index}", leaf)
    for name in ("t0_cov", "t0_means", "diffusion"):
        _update_array_digest(digest, name, samples[name])
    _update_array_digest(digest, "times", times)
    _update_array_digest(digest, "rng-key", rng_key)
    return digest.hexdigest()


def _cached_latents(key: str) -> jax.Array | None:
    with _latent_cache_lock:
        cached = _latent_cache.pop(key, None)
        if cached is not None:
            _latent_cache[key] = cached
        return cached


def _cache_latents(key: str, latents: jax.Array) -> None:
    with _latent_cache_lock:
        _latent_cache[key] = latents
        _latent_cache.move_to_end(key)
        while len(_latent_cache) > _LATENT_CACHE_MAX_ENTRIES:
            _latent_cache.popitem(last=False)


def _predictive_max_rates(
    compiled: CompiledDynamics, samples: dict[str, jnp.ndarray]
) -> jnp.ndarray:
    """Read the fastest declared relaxation rate per draw from coefficient metadata."""
    n_draws = int(next(iter(samples.values())).shape[0])
    rates = [
        jnp.max(jnp.abs(samples[site.name]).reshape(n_draws, -1), axis=1)
        for site in compiled.site_registry
        if site.site_kind == SiteKind.DYNAMICS_DECAY
    ]
    for component in compiled.spec.components:
        rates.extend(
            jnp.full(n_draws, operand.value)
            for operand in expression_coefficients(component.expression)
            if operand.meaning.quantity == SiteKind.DYNAMICS_DECAY
            and isinstance(operand.value, (int, float))
        )
    return jnp.max(jnp.stack(rates), axis=0) if rates else jnp.zeros(n_draws)


def _predictive_sde_step(max_rate: jnp.ndarray, span: float) -> jnp.ndarray:
    # Traced (not host) arithmetic: the per-draw step size stays a jnp scalar
    # so every draw reuses ONE compiled program — a host float here bakes into
    # the XLA graph as a constant and forces a retrace + recompile per draw.
    sde_dt = jnp.minimum(span / 200.0, _SDE_CFL_SAFETY / jnp.maximum(max_rate, 1e-30))
    # Keep the step count within the solver budget for pathologically fast draws.
    return jnp.maximum(sde_dt, span / _SDE_MAX_STEPS)


def _predictive_sde_config(max_rate: jnp.ndarray, span: float) -> SimulationConfig:
    if span <= 0.0:
        return SimulationConfig()
    return SimulationConfig(
        sde_dt=_predictive_sde_step(max_rate, span),
        max_steps=_SDE_MAX_STEPS + 16,
        use_indexed_brownian_path=True,
    )


def _predictive_draw_order(max_rates: jnp.ndarray, span: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Order draws by expected solver work and return the inverse permutation."""
    n_draws = int(max_rates.shape[0])
    if span <= 0.0:
        identity = jnp.arange(n_draws)
        return identity, identity

    sde_dt = _predictive_sde_step(max_rates, span)
    step_counts = jnp.ceil(span / sde_dt)

    order = jnp.argsort(step_counts, stable=True)
    return order, jnp.argsort(order)


def _simulate_model_predictive_latent_draw(
    model: dsx.DynamicalModel,
    times: jnp.ndarray,
    key: jnp.ndarray,
    span: float,
    max_rate: jnp.ndarray,
) -> jnp.ndarray:
    """Draw the native initial law and execute the model's exact state evolution."""
    key_init, key_latent = random.split(key)
    return simulate_model_path(
        model,
        model.initial_condition.sample(key_init),
        times,
        config=_predictive_sde_config(max_rate, span),
        key=key_latent,
    )


def _simulate_model_predictive_draws_microbatched(
    models: dsx.DynamicalModel,
    times: jnp.ndarray,
    keys: jnp.ndarray,
    span: float,
    max_rates: jnp.ndarray,
) -> jnp.ndarray:
    arrays, structure = eqx.partition(models, eqx.is_array)

    def simulate_one(args):
        model_arrays, key, max_rate = args
        return _simulate_model_predictive_latent_draw(
            eqx.combine(model_arrays, structure),
            times,
            key,
            span,
            max_rate,
        )

    return jax.lax.map(
        simulate_one,
        (arrays, keys, max_rates),
        batch_size=_PREDICTIVE_MICROBATCH_SIZE,
    )


_simulate_model_predictive_draws = eqx.filter_jit(_simulate_model_predictive_draws_microbatched)


def _simulate_vector_field_predictive_latents(
    spec: ModelSpec,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    *,
    rng_key: jax.Array,
    dynamics: DynamicsSpec | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    _ensure_gaussian_process_diffusion(spec)
    dynamics = numeric.dynamics_components(spec) if dynamics is None else dynamics
    compiled = compile_dynamics(dynamics)
    n_draws = int(next(iter(samples.values())).shape[0])
    draw_keys = random.split(rng_key, n_draws)
    models = _predictive_models(spec, samples, times, dynamics=dynamics)
    span = float(times[-1] - times[0]) if int(times.shape[0]) > 1 else 0.0
    cache_key = _prior_predictive_latent_cache_key(
        dynamics,
        models.state_evolution,
        samples,
        times,
        rng_key,
    )
    latents = _cached_latents(cache_key)
    if latents is None:
        max_rates = _predictive_max_rates(compiled, samples)
        order, inverse_order = _predictive_draw_order(max_rates, span)
        sorted_models = jax.tree.map(
            lambda leaf: leaf[order] if eqx.is_array(leaf) else leaf, models
        )
        sorted_latents = _simulate_model_predictive_draws(
            sorted_models,
            times,
            draw_keys[order],
            span,
            max_rates[order],
        )
        latents = sorted_latents[inverse_order]
        _cache_latents(cache_key, latents)
        logger.info("Prior-predictive latent cache miss %s", cache_key[:12])
    else:
        logger.info("Prior-predictive latent cache hit %s", cache_key[:12])
    linear_predictors = eqx.filter_vmap(
        lambda model, path: jax.vmap(model.observation_model.linear_predictor)(path)
    )(models, latents)
    return latents, linear_predictors


def sample_prior_parameters_from_runtime(
    spec: ModelSpec,
    runtime: PriorRuntimeBundle,
    *,
    num_samples: int,
    rng_key: jax.Array,
) -> dict[str, jnp.ndarray]:
    """Sample and assemble the parameter layer of a prior predictive run."""
    constrained_samples = sample_prior_parameters(
        rng_key,
        runtime.registry,
        runtime.priors,
        n_samples=num_samples,
    )
    deterministic_samples = assemble_deterministics_from_registry(
        constrained_samples,
        spec,
        n_draws=num_samples,
    )
    extra_params = _assemble_extra_params_batched(
        spec,
        constrained_samples,
        runtime.registry,
        n_draws=num_samples,
    )
    return {
        **constrained_samples,
        **deterministic_samples,
        **extra_params,
    }


def simulate_predictive_latents(
    spec: ModelSpec,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    *,
    rng_key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate exact nonlinear latent paths and their linear predictors."""
    return _simulate_vector_field_predictive_latents(
        spec,
        samples,
        times,
        rng_key=rng_key,
    )


def sample_predictive_emissions(
    spec: ModelSpec,
    samples: dict[str, jnp.ndarray],
    linear_predictors: jnp.ndarray,
    times: jnp.ndarray,
    *,
    observation_support,
    observation_mask: jnp.ndarray | None,
    num_samples: int,
    rng_key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample the observation layer conditional on cached latent predictors."""
    indices = jnp.linspace(
        0, linear_predictors.shape[0] - 1, min(num_samples, linear_predictors.shape[0])
    ).astype(int)
    return sample_model_observations(
        _predictive_models(
            spec, {name: values[indices] for name, values in samples.items()}, times
        ),
        linear_predictors[indices],
        times,
        rng_key=rng_key,
        observation_support=observation_support,
        observation_mask=observation_mask,
        manifest_names=list(numeric.observation_names(spec)),
    )


def sample_prior_predictive_from_runtime(
    spec: ModelSpec,
    runtime: PriorRuntimeBundle,
    times: jnp.ndarray,
    *,
    observation_support=None,
    observation_mask: jnp.ndarray | None = None,
    num_samples: int = 100,
    seed: int = 0,
) -> dict[str, jnp.ndarray]:
    """Sample prior predictive draws from a prepared runtime bundle."""
    _ensure_discrete_metadata(spec)

    keys = predictive_keys(seed)

    samples = sample_prior_parameters_from_runtime(
        spec,
        runtime,
        num_samples=num_samples,
        rng_key=keys.parameters,
    )
    return simulate_predictive_draws(
        spec,
        samples,
        times,
        observation_support=observation_support,
        observation_mask=observation_mask,
        seed=seed,
    )


def simulate_predictive_draws(
    spec: ModelSpec,
    samples: dict[str, jnp.ndarray],
    times: jnp.ndarray,
    *,
    observation_support=None,
    observation_mask: jnp.ndarray | None = None,
    seed: int = 0,
    initial_states: jax.Array | None = None,
    process_noise: bool = True,
    observation_noise: bool = True,
    clamps=(),
) -> dict[str, jnp.ndarray]:
    """Generate one shared path/observation batch from aligned parameter draws."""
    if process_noise:
        _ensure_gaussian_process_diffusion(spec)
    _ensure_discrete_metadata(spec)
    n_draws = int(next(iter(samples.values())).shape[0])
    samples = dict(samples)
    samples.update(
        _assemble_extra_params_batched(spec, samples, build_site_registry(spec), n_draws=n_draws)
    )
    keys = predictive_keys(seed)
    reference_latents = None
    if initial_states is not None or not process_noise or clamps:
        latents, linear_predictors, reference_latents = _simulate_designed_latents(
            spec, samples, times, keys.latents, initial_states, process_noise, clamps
        )
    else:
        latents, linear_predictors = simulate_predictive_latents(
            spec, samples, times, rng_key=keys.latents
        )
    observations, observations_mask, expected_observations = sample_predictive_emissions(
        spec,
        samples,
        linear_predictors,
        times,
        observation_support=observation_support,
        observation_mask=observation_mask,
        num_samples=n_draws,
        rng_key=keys.observations,
    )
    samples["latents"] = latents
    samples["linear_predictors"] = linear_predictors
    samples["observations"] = observations if observation_noise else expected_observations
    samples["observations_mask"] = observations_mask
    samples["expected_observations"] = expected_observations
    if reference_latents is not None:
        models = _predictive_models(spec, samples, times)
        reference_predictors = eqx.filter_vmap(
            lambda model, path: jax.vmap(model.observation_model.linear_predictor)(path)
        )(models, reference_latents)
        reference_observations, _, reference_means = sample_predictive_emissions(
            spec,
            samples,
            reference_predictors,
            times,
            observation_support=observation_support,
            observation_mask=observation_mask,
            num_samples=n_draws,
            rng_key=keys.observations,
        )
        samples["reference_latents"] = reference_latents
        samples["reference_observations"] = (
            reference_observations if observation_noise else reference_means
        )
    return samples


def _simulate_designed_latents(spec, samples, times, key, initial_states, process_noise, clamps):
    """Execute the same nonlinear field with explicit starts, noise and paired do-operations."""
    from nof1_causal_lab.models.ssm.counterfactual.orchestration import (
        vmap_simulate_clamps_from_state,
    )
    from nof1_causal_lab.models.ssm.dynamics.posterior import posterior_dynamics_from_samples

    models = _predictive_models(spec, samples, times)
    draw_keys = random.split(key, next(iter(samples.values())).shape[0])
    if initial_states is None:
        initial_states = eqx.filter_vmap(
            lambda model, k: model.initial_condition.sample(random.split(k)[0])
        )(models, draw_keys)
    dynamics = posterior_dynamics_from_samples(spec, samples)
    span = float(times[-1] - times[0])
    max_rates = _predictive_max_rates(compile_dynamics(numeric.dynamics_components(spec)), samples)
    # Paired paths use identical step sizes and random streams in each segment.
    reference, latents, _ = vmap_simulate_clamps_from_state(
        dynamics.vector_field,
        dynamics.param_samples,
        initial_states,
        list(clamps),
        time_grid=times,
        config=_predictive_sde_config(jnp.max(max_rates), span)
        if process_noise
        else SimulationConfig(),
        keys=jax.vmap(lambda k: random.split(k)[1])(draw_keys) if process_noise else None,
        diffusion_cov=samples["diffusion"] @ jnp.swapaxes(samples["diffusion"], -1, -2)
        if process_noise
        else None,
    )
    predictors = eqx.filter_vmap(
        lambda model, path: jax.vmap(model.observation_model.linear_predictor)(path)
    )(models, latents)
    return latents, predictors, reference if clamps else None
