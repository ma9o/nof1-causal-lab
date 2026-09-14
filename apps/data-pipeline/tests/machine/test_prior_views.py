"""Prior display is an ephemeral read of a native law, never authored science."""

import math
from itertools import pairwise

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from pydantic import TypeAdapter

from nof1_causal_lab.machine.prior_views import prior_density
from nof1_causal_lab.numpyro_json import NumPyroDistribution
from nof1_causal_lab.prior_distributions import interval_effect_to_rate


def test_prior_curves_preserve_native_gamma_and_transforms_without_mutating_the_law():
    prior = dist.Gamma(2.0, 3.0)
    adapter = TypeAdapter(NumPyroDistribution)
    before = adapter.dump_json(prior)
    for law, rate in ((prior, 3.0), (interval_effect_to_rate(prior, 2.0), 6.0)):
        curve = prior_density(law)
        assert len(curve) == 100
        assert all(left.x < right.x for left, right in pairwise(curve))
        np.testing.assert_allclose(
            [point.y for point in curve],
            [rate**2 * point.x * math.exp(-rate * point.x) for point in curve],
            rtol=2e-6,
        )
        assert prior_density(law) == curve
    assert adapter.dump_json(prior) == before
    for law in (
        dist.Delta(1.0),
        dist.Bernoulli(0.5),
        dist.Normal(jnp.zeros(2), 1.0),
        dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
    ):
        assert prior_density(law) == ()
