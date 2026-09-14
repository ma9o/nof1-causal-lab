"""Numerical primitives for conditional trajectory kernels."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

FloatScalar = Float[Array, ""]


def _masked_normal_log_prob(values, mean, variance, free_mask):
    """Density with respect to the free coordinates, including its normalization."""
    per_coordinate = -0.5 * (
        jnp.log(2.0 * jnp.pi * variance) + jnp.square(values - mean) / variance
    )
    return jnp.sum(jnp.where(free_mask, per_coordinate, 0.0), axis=-1)


def _masked_mean(values, mask, *, axis=None):
    """Average diagnostics over sampled coordinates; an empty selection contributes zero."""
    return jnp.sum(jnp.where(mask, values, 0.0), axis=axis) / jnp.maximum(
        jnp.sum(mask, axis=axis), 1
    )


def _normalize_log_probs(
    logits: Float[Array, "*shape"], *, axis: int = -1
) -> Float[Array, "*shape"]:
    return logits - jax.scipy.special.logsumexp(logits, axis=axis, keepdims=True)


def _observation_log_probs_by_param(
    contexts,
    particles_t: Float[Array, "P D"],
    time_idx: Int[Array, ""],
    runtime_observations: Float[Array, "T M"],
    obs_increment_fn,
) -> Float[Array, "P K"]:
    def _one_param(context):
        return jax.vmap(
            lambda particle: obs_increment_fn(
                context,
                particle,
                time_idx,
                runtime_observations,
            )
        )(particles_t)

    return jnp.swapaxes(jax.vmap(_one_param)(contexts), 0, 1)


def _select_pytree(ensemble, index: Int[Array, "..."]):
    return jax.tree_util.tree_map(lambda leaf: leaf[index], ensemble)
