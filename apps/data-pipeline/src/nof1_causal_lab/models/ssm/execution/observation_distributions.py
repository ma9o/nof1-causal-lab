"""Native observation laws shared by likelihoods, draws, and moments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints

from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.execution.contracts import LikelihoodExtraParams


def gaussian_distribution(mean: jax.Array, covariance: jax.Array) -> dist.Distribution:
    return dist.MultivariateNormal(mean, covariance_matrix=symmetrize_with_jitter(covariance))


def categorical_distribution(probs: jax.Array) -> dist.Distribution:
    """Logits preserve exact zero probabilities, including padded categories."""
    return dist.Categorical(logits=jnp.log(probs))


def binary_logits_distribution(logits: jax.Array) -> dist.Distribution:
    """Stable predictor tails with the same binary categorical law as mean space."""
    return dist.Categorical(
        logits=jnp.stack([jax.nn.log_sigmoid(-logits), jax.nn.log_sigmoid(logits)], axis=-1)
    )


_MEAN_DOMAINS = {
    DistributionFamily.GAUSSIAN: constraints.real,
    DistributionFamily.STUDENT_T: constraints.real,
    DistributionFamily.POISSON: constraints.nonnegative,
    DistributionFamily.GAMMA: constraints.positive,
    DistributionFamily.BERNOULLI: constraints.unit_interval,
    DistributionFamily.NEGATIVE_BINOMIAL: constraints.nonnegative,
    DistributionFamily.BETA: constraints.open_interval(0.0, 1.0),
}


def safe_observation_mean(family: DistributionFamily, mean: jax.Array):
    """Preserve valid means; use interior placeholders only for invalid channels."""
    valid = jnp.isfinite(mean) & _MEAN_DOMAINS[family](mean)
    return jnp.where(valid, mean, 0.5), valid


def mean_parameter_distribution(
    family: DistributionFamily,
    mean: jax.Array,
    scale: jax.Array | float,
    extra_params: LikelihoodExtraParams,
) -> dist.Distribution:
    """Construct the exact independent-channel law for an already valid mean."""
    match family:
        case DistributionFamily.GAUSSIAN:
            return dist.Normal(mean, scale)
        case DistributionFamily.STUDENT_T:
            return dist.StudentT(extra_params.get("obs_df", 5.0), mean, scale)
        case DistributionFamily.POISSON:
            return dist.Poisson(mean)
        case DistributionFamily.GAMMA:
            shape = extra_params.get("obs_shape", 1.0)
            return dist.Gamma(shape, shape / mean)
        case DistributionFamily.BERNOULLI:
            # BernoulliProbs clamps its density at p=0/1. Binary categorical logits
            # retain the exact same probability law for both density and draws.
            return categorical_distribution(jnp.stack([1.0 - mean, mean], axis=-1))
        case DistributionFamily.NEGATIVE_BINOMIAL:
            # NegativeBinomial2 requires mean > 0. Its mean=0 limit is a point
            # mass at zero. A deterministic native selector includes that limit
            # without an epsilon rate or an additional zero-inflation parameter.
            zero = mean == 0.0
            selector = dist.Categorical(
                logits=jnp.stack(
                    [jnp.where(zero, 0.0, -jnp.inf), jnp.where(zero, -jnp.inf, 0.0)], axis=-1
                )
            )
            return dist.MixtureGeneral(
                selector,
                [
                    dist.DiscreteUniform(jnp.zeros_like(mean, dtype=jnp.int32), 0),
                    dist.NegativeBinomial2(
                        jnp.where(zero, 1.0, mean), extra_params.get("obs_r", 5.0)
                    ),
                ],
                support=constraints.nonnegative_integer,
            )
        case DistributionFamily.BETA:
            concentration = extra_params.get("obs_concentration", 10.0)
            return dist.Beta(mean * concentration, (1.0 - mean) * concentration)
        case _:
            raise ValueError(f"Mean-parameter observations are not defined for {family.value!r}")


def sample_mean_observation(family, key, mean, scale, extra_params):
    safe_mean, valid = safe_observation_mean(family, mean)
    draw = mean_parameter_distribution(family, safe_mean, scale, extra_params).sample(key)
    return jnp.where(valid, draw, jnp.nan)
