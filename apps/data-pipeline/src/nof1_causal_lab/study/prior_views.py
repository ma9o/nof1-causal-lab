"""Ephemeral prior curves on the authoring and quantity scales, evaluated by NumPyro."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from nof1_causal_lab.artifacts.posterior_diagnostics import DensityCurve


def prior_density(prior: dist.Distribution) -> DensityCurve:
    """Plot scalar continuous laws; joint, batched and point masses have no scalar PDF.

    A small deterministic native draw selects only the viewport. Every ordinate
    is the native log_prob, never a density estimate or an inference input.
    """
    if prior.batch_shape or prior.event_shape or prior.is_discrete or isinstance(prior, dist.Delta):
        return DensityCurve()
    return _density_curve(prior)


@lru_cache(maxsize=128)
def _density_curve(prior: dist.Distribution) -> DensityCurve:
    draws = np.asarray(prior.sample(jax.random.PRNGKey(0), (256,)))
    low, high = np.quantile(draws, [0.01, 0.99])
    x = jnp.linspace(low, high, 100)
    y = np.asarray(jnp.exp(prior.log_prob(x)))
    points = tuple(
        (float(value), float(density))
        for value, density in zip(np.asarray(x), y, strict=True)
        if np.isfinite(value) and np.isfinite(density)
    )
    return DensityCurve(
        x=tuple(point[0] for point in points), density=tuple(point[1] for point in points)
    )
