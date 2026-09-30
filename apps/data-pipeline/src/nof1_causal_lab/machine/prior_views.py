"""Ephemeral prior curves on the authoring and quantity scales, evaluated by NumPyro."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from pydantic import TypeAdapter

from nof1_causal_lab.machine.view_models import DensityPoint
from nof1_causal_lab.numpyro_json import NumPyroDistribution, distribution_shape

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ParameterId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec

_PRIOR = TypeAdapter(NumPyroDistribution)


def prior_density(prior: dist.Distribution) -> tuple[DensityPoint, ...]:
    """Plot scalar continuous laws; joint, batched and point masses have no scalar PDF.

    A small deterministic native draw selects only the viewport. Every ordinate
    is the native log_prob, never a density estimate or an inference input.
    """
    if prior.batch_shape or prior.event_shape or prior.is_discrete or isinstance(prior, dist.Delta):
        return ()
    return _density_curve(_PRIOR.dump_json(prior))


def quantity_prior_densities(model: ModelSpec) -> dict[ParameterId, tuple[DensityPoint, ...]]:
    """Plot each scalar law on the quantity scale where fitting reports its posterior.

    Laws pass through the compilation the engine conditions on, so persistence and
    interval-effect priors share an axis with the decay and rate posteriors.
    """
    from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_parameter_law
    from nof1_causal_lab.models.ssm.compile.prior_indexing import build_semantic_prior_bindings

    semantics = build_semantic_prior_bindings(model).by_parameter
    return {
        parameter.id: prior_density(
            compile_parameter_law(model, parameter, semantics[parameter.id])[0]
        )
        for parameter in model.execution_parameters
        if parameter.distribution is not None
        and not any(distribution_shape(model.distributions[parameter.distribution]))
    }


@lru_cache(maxsize=128)
def _density_curve(constructor: bytes) -> tuple[DensityPoint, ...]:
    prior = _PRIOR.validate_json(constructor)
    draws = np.asarray(prior.sample(jax.random.PRNGKey(0), (256,)))
    low, high = np.quantile(draws, [0.01, 0.99])
    x = jnp.linspace(low, high, 100)
    y = np.asarray(jnp.exp(prior.log_prob(x)))
    return tuple(
        DensityPoint(x=float(value), y=float(density))
        for value, density in zip(np.asarray(x), y, strict=True)
        if np.isfinite(value) and np.isfinite(density)
    )
