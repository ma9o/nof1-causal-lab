"""Numerical acceptance of Dynestyx model interpretation with local inference."""

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.inference.problem import build_particle_problem
from nof1_causal_lab.models.ssm.model import SSMModel

from tests.dynamics_fixtures import potential_term
from tests.model_fixtures import default_lambda_block, model_fixture

pytestmark = pytest.mark.cpu_expensive


def nonlinear_model():
    loading = replace(default_lambda_block(2, 1), template=jnp.ones((2, 1)))
    spec = model_fixture(
        n_latent=1,
        n_manifest=2,
        lambda_block=loading,
        dynamics_spec=DynamicsSpec(
            1,
            (
                potential_term(
                    target=0,
                    center=None,
                    stiffness=FixedCoefficient(value=0.4),
                    quartic=FixedCoefficient(value=0.2),
                ),
            ),
        ),
        manifest_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.POISSON],
        manifest_links=[LinkFunction.IDENTITY, LinkFunction.LOG],
    )
    return SSMModel(spec)


def test_nonlinear_mixed_missing_irregular_particle_fit_and_exact_diagnostics():
    model = nonlinear_model()
    times = jnp.array([0.0, 0.05, 0.17, 0.4, 0.9])
    observations = jnp.array(
        [[0.2, 1.0], [jnp.nan, 2.0], [0.3, jnp.nan], [jnp.nan, jnp.nan], [-0.2, 0.0]]
    )
    problem = build_particle_problem(
        model,
        observations,
        times,
        scheme="euler_maruyama",
        trace_key=jax.random.key(7),
        reparam=None,
    )
    runtime = problem.runtime
    context = runtime.context(runtime.initial_position, times)
    path = jnp.array([[0.1], [0.15], [-0.2], [0.05], [0.3]])
    # Independent emission expressions, including all- and partially-missing rows.
    variance = context[0].observation_model.measurement.manifest_cov[0, 0]
    expected = jnp.array(
        [
            dist.Normal(path[0, 0], jnp.sqrt(variance)).log_prob(0.2)
            + dist.Poisson(jnp.exp(path[0, 0])).log_prob(1.0),
            dist.Poisson(jnp.exp(path[1, 0])).log_prob(2.0),
            dist.Normal(path[2, 0], jnp.sqrt(variance)).log_prob(0.3),
            0.0,
            dist.Normal(path[4, 0], jnp.sqrt(variance)).log_prob(-0.2)
            + dist.Poisson(jnp.exp(path[4, 0])).log_prob(0.0),
        ]
    )
    np.testing.assert_allclose(
        runtime.observation_log_probs(context, path, observations), expected, atol=2e-5
    )
    transition = runtime.transition_log_prob(context, path[0], path[1], jnp.asarray(1))
    # Genuine nonlinear drift, rather than agreement between two linear surrogates.
    center = context[0].state_evolution.drift.args.params[0]["center"]
    drift = -0.4 * (path[0] - center) - 0.2 * (path[0] - center) ** 3
    law = dist.MultivariateNormal(
        path[0] + 0.05 * drift,
        covariance_matrix=0.05
        * context[0].state_evolution.diffusion.gram_matrix(x=None, u=None, t=0, state_dim=1)
        + jnp.eye(1) * 1e-8,
    )
    np.testing.assert_allclose(transition, law.log_prob(path[1]), atol=2e-5)
    result = fit(
        model,
        observations,
        times,
        method="marginal_particle_gibbs",
        num_warmup=2,
        num_samples=4,
        num_chains=1,
        seed=9,
        n_particles=4,
        n_parameter_particles=2,
        param_step_size=0.0005,
        latent_delta=0.2,
        init_method="random",
        auto_preconditioner_method="none",
        init_scale=0.0,
        retain_latent_paths=True,
        reparam=None,
    )
    assert "likelihood_backend" not in result.diagnostics
    latent_paths = result.draws.latent_paths
    assert latent_paths is not None
    assert latent_paths.shape == (4, 5, 1)
    factors = result.diagnostics["observation_log_probs"]
    assert factors.shape == (1, 4, 5)
    assert jnp.all(jnp.isfinite(factors))
    assert jnp.all(factors[..., 3] == 0.0)
    for values in result.get_samples().values():
        assert jnp.all(jnp.isfinite(values))
