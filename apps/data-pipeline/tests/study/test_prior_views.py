"""Prior display is an ephemeral read of a native law, never authored science."""

import math
from itertools import pairwise
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.numpyro_json import NumPyroDistribution
from nof1_causal_lab.prior_distributions import interval_effect_to_rate, persistence_to_decay
from nof1_causal_lab.study.prior_views import prior_density, quantity_prior_densities

pytestmark = [
    pytest.mark.inference(concern="sampling"),
    pytest.mark.inference(concern="predictive"),
]


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


def test_quantity_curves_put_authored_laws_on_the_posterior_scale():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/models"
            / "prior_views/quantity_curves_put_authored_laws_on_the_posterior_scale_complete_test_model.json"
        ).read_text()
    )
    curves = quantity_prior_densities(model)
    transforms = {parameter.distribution_transform for parameter in model.parameters}
    assert "dt_persistence_to_ct_decay" in transforms
    for parameter in model.parameters:
        law = model.distribution_for(parameter.id)
        match parameter.distribution_transform:
            case "dt_persistence_to_ct_decay":
                # Posteriors report decay rates, so persistence priors move to that axis.
                native = persistence_to_decay(law, 1.0)
            case "dt_effect_to_ct_rate":
                native = interval_effect_to_rate(law, 1.0)
            case "identity" if law is not None:
                native = law
            case _:
                continue
        np.testing.assert_allclose(
            [(point.x, point.y) for point in curves[parameter.id]],
            [(point.x, point.y) for point in prior_density(native)],
            rtol=1e-6,
            err_msg=parameter.name,
        )
