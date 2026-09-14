"""Parity tests for shared predictor- and mean-space observation draws."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.likelihood import DistributionFamily
from nof1_causal_lab.models.ssm.execution.emissions import (
    emission_log_prob_bernoulli,
    get_mean_param_log_prob_fn,
    get_mean_param_sample_fn,
)
from nof1_causal_lab.models.ssm.execution.observation_distributions import (
    mean_parameter_distribution,
)
from nof1_causal_lab.models.ssm.execution.observation_families import FAMILY_REGISTRY

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.execution.contracts import LikelihoodExtraParams


@pytest.mark.parametrize(
    ("family", "link", "predictor", "mean", "extra_params", "std"),
    [
        ("delta", "identity", -0.4, -0.4, {}, 0.0),
        ("student_t", "identity", 0.4, 0.4, {"obs_df": 4.0}, 0.7),
        ("poisson", "log", 0.4, float(jnp.exp(0.4)), {}, 1.0),
        ("gamma", "log", 0.4, float(jnp.exp(0.4)), {"obs_shape": 2.0}, 1.0),
        ("gamma", "inverse", 2.0, 0.5, {"obs_shape": 2.0}, 1.0),
        ("bernoulli", "logit", 0.4, float(jax.nn.sigmoid(0.4)), {}, 1.0),
        (
            "bernoulli",
            "probit",
            0.4,
            float(jax.scipy.stats.norm.cdf(0.4)),
            {},
            1.0,
        ),
        (
            "negative_binomial",
            "log",
            0.4,
            float(jnp.exp(0.4)),
            {"obs_r": 3.0},
            1.0,
        ),
        (
            "beta",
            "logit",
            0.4,
            float(jax.nn.sigmoid(0.4)),
            {"obs_concentration": 8.0},
            1.0,
        ),
        (
            "beta",
            "probit",
            0.4,
            float(jax.scipy.stats.norm.cdf(0.4)),
            {"obs_concentration": 8.0},
            1.0,
        ),
    ],
)
def test_predictor_and_mean_samplers_use_the_same_draw(
    family: str,
    link: str,
    predictor: float,
    mean: float,
    extra_params: LikelihoodExtraParams,
    std: float,
) -> None:
    key = jax.random.PRNGKey(42)
    point_fn = FAMILY_REGISTRY[DistributionFamily(family)].posterior_predictive_fns[link]
    point_draw = point_fn(
        jnp.asarray(predictor),
        key,
        jnp.asarray(std),
        jnp.asarray(extra_params.get("obs_df", 4.0)),
        jnp.asarray(extra_params.get("obs_shape", 2.0)),
        jnp.asarray(extra_params.get("obs_r", 3.0)),
        jnp.asarray(extra_params.get("obs_concentration", 8.0)),
        jnp.asarray(1),
        jnp.zeros(1),
        jnp.zeros(1),
        jnp.zeros(1),
    )
    mean_draw = get_mean_param_sample_fn(family, extra_params)(
        key,
        jnp.asarray([mean]),
        jnp.asarray([[std**2]]),
    )[0]

    np.testing.assert_array_equal(point_draw, mean_draw)


@pytest.mark.parametrize(
    ("family", "extra_params", "invalid_mean"),
    [
        ("poisson", {}, -1.0),
        ("gamma", {"obs_shape": 2.0}, 0.0),
        ("bernoulli", {}, 1.1),
        ("negative_binomial", {"obs_r": 3.0}, -1.0),
        ("beta", {"obs_concentration": 8.0}, 0.0),
    ],
)
def test_mean_samplers_surface_invalid_domains_as_nan(
    family: str,
    extra_params: LikelihoodExtraParams,
    invalid_mean: float,
) -> None:
    draw = get_mean_param_sample_fn(family, extra_params)(
        jax.random.PRNGKey(0),
        jnp.asarray([invalid_mean]),
        jnp.eye(1),
    )

    assert jnp.isnan(draw[0])


def test_zero_mean_negative_binomial_is_an_exact_point_mass():
    law = mean_parameter_distribution(
        DistributionFamily.NEGATIVE_BINOMIAL, jnp.array([0.0, 2.0]), 1.0, {"obs_r": 4.0}
    )
    assert law.sample(jax.random.PRNGKey(3))[0] == 0.0
    assert law.log_prob(jnp.array([0, 0]))[0] == 0.0
    assert jnp.isneginf(law.log_prob(jnp.array([1, 1]))[0])
    np.testing.assert_allclose(law.mean, [0.0, 2.0])
    np.testing.assert_allclose(law.variance, [0.0, 3.0])


def test_beta_sampler_and_density_preserve_small_authored_shapes():
    mean = jnp.array([0.001])
    concentration = 0.1
    key = jax.random.PRNGKey(12)
    native = dist.Beta(mean * concentration, (1.0 - mean) * concentration)
    extras: LikelihoodExtraParams = {"obs_concentration": concentration}
    sample = get_mean_param_sample_fn("beta", extras)(key, mean, jnp.eye(1))
    np.testing.assert_array_equal(sample, native.sample(key))
    density = get_mean_param_log_prob_fn("beta", extras)(
        jnp.array([0.2]), mean, jnp.eye(1), jnp.ones(1)
    )
    np.testing.assert_allclose(density, native.log_prob(0.2).sum(), atol=1e-6)


def test_binary_boundaries_and_predictor_tails_remain_exact():
    likelihood = get_mean_param_log_prob_fn("bernoulli")
    mean = jnp.array([0.0, 1.0])
    assert likelihood(mean, mean, jnp.eye(2), jnp.ones(2)) == 0.0
    assert jnp.isneginf(likelihood(1.0 - mean, mean, jnp.eye(2), jnp.ones(2)))
    np.testing.assert_array_equal(
        get_mean_param_sample_fn("bernoulli")(jax.random.key(9), mean, jnp.eye(2)), mean
    )
    assert float(
        emission_log_prob_bernoulli(jnp.zeros(1), jnp.array([100.0]), jnp.eye(1), jnp.ones(1))
    ) == pytest.approx(-100.0)


def test_negative_binomial_density_has_the_exact_mean_gradient():
    likelihood = get_mean_param_log_prob_fn("negative_binomial", {"obs_r": 3.0})
    y, mean = jnp.array([0.0, 4.0]), jnp.array([1.5, 8.0])

    def fn(mu):
        return likelihood(y, mu, jnp.eye(2), jnp.ones(2))

    expected = y / mean - (3.0 + y) / (3.0 + mean)
    np.testing.assert_allclose(jax.grad(fn)(mean), expected, atol=2e-6)


@pytest.mark.parametrize(
    ("family", "extras"), [("gamma", {"obs_shape": 2.0}), ("beta", {"obs_concentration": 3.0})]
)
def test_missing_invalid_observation_has_zero_gradient(family, extras):
    likelihood = get_mean_param_log_prob_fn(family, extras)

    def fn(mu):
        return likelihood(jnp.array([0.2, jnp.nan]), mu, jnp.eye(2), jnp.array([1.0, 0.0]))

    gradient = jax.grad(fn)(jnp.array([0.4, 0.6]))
    assert jnp.isfinite(gradient[0])
    assert gradient[1] == 0.0
