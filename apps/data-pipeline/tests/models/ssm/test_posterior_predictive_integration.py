"""Predictive runtime and diagnostic-report integration on a small mixed model."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorPredictiveChecks
from nof1_causal_lab.models.posterior_predictive import run_posterior_predictive_checks
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.model import SSMModel
from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
    sample_prior_predictive_from_runtime,
)
from tests.dynamics_fixtures import potential_term
from tests.model_fixtures import default_lambda_block, model_fixture

pytestmark = pytest.mark.predictive


def test_predictive_draws_feed_mixed_family_diagnostics():
    # Exhaustive family/link domains are checked by the observation-sampling
    # test. This case checks the complete prior/runtime/report connection.
    spec = model_fixture(
        n_latent=1,
        n_manifest=2,
        dynamics_spec=DynamicsSpec(
            1, (potential_term(target=0, center=0.0, stiffness=0.4, quartic=0.2),)
        ),
        lambda_block=replace(default_lambda_block(2, 1), template=jnp.ones((2, 1))),
        manifest_dists=["gaussian", "poisson"],
        manifest_links=["identity", "log"],
        manifest_names=["signal", "count"],
    )
    runtime = SSMModel(spec).get_prior_runtime_bundle()
    times = jnp.array([0.0, 0.1, 0.25, 0.4, 0.7, 1.0], dtype=jnp.float32)
    samples = sample_prior_predictive_from_runtime(spec, runtime, times, num_samples=3, seed=7)
    assert samples["latents"].shape == (3, 6, 1)
    assert samples["observations"].shape == (3, 6, 2)
    assert samples["observations_mask"].shape == (3, 6, 2)
    assert bool(samples["observations_mask"].all())
    assert all(bool(jnp.isfinite(value).all()) for value in samples.values())
    counts = samples["observations"][..., 1]
    assert bool((counts >= 0).all())
    np.testing.assert_array_equal(counts, jnp.floor(counts))

    # Known observations avoid a second simulation just to construct test data.
    observations = jnp.array(
        [[0.2, 1.0], [jnp.nan, 2.0], [-0.1, 0.0], [0.1, 3.0], [-0.2, 2.0], [0.3, 4.0]]
    )
    indicator_ids = ["indicator:signal", "indicator:count"]
    result = run_posterior_predictive_checks(
        samples=samples,
        observations=observations,
        times=times,
        indicator_ids=indicator_ids,
        spec=spec,
        n_subsample=8,
    )

    assert isinstance(result, PosteriorPredictiveChecks)
    assert result.checked is True
    assert result.n_subsample == 3
    assert [overlay.indicator_id for overlay in result.overlays] == indicator_ids
    assert len(result.test_stats) == 8
    assert {
        (warning.indicator_id, warning.check_type) for warning in result.per_variable_warnings
    } == {
        (indicator_id, check)
        for indicator_id in indicator_ids
        for check in ("calibration", "autocorrelation", "variance")
    }
