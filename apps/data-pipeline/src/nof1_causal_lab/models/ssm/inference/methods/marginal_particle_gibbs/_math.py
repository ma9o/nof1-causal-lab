"""Numerical primitives for conditional trajectory kernels."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

FloatScalar = Float[Array, ""]


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
