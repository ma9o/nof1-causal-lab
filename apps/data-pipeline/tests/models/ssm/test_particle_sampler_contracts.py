"""Particle configuration and initialization contracts without compiling a sampler."""

from typing import Literal, cast
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest
from dynestyx.inference.particle_runtime import ParticleRuntime

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs import (
    fit_marginal_particle_gibbs,
    runner,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.kernel import (
    MarginalParticleGibbsKernel,
    build_marginal_particle_gibbs_kernel,
)
from nof1_causal_lab.models.ssm.model import SSMModel


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"parameter_proposal": "bogus"}, "parameter_proposal"),
        ({"latent_smoother": "bogus"}, "latent_smoother"),
        ({"dsmc_leaf_proposal": "paid_mix"}, "requires pilot"),
    ],
)
def test_kernel_rejects_invalid_configuration_before_accessing_target(options, error):
    # No target attributes are available: validation must precede numerical setup.
    target = Mock(spec_set=[])
    with pytest.raises(ValueError, match=error):
        build_marginal_particle_gibbs_kernel(
            target,
            num_particles=2,
            num_parameter_particles=2,
            param_step_size=0.1,
            **options,
        )


def test_fit_rejects_unknown_adaptation_before_building_model():
    with pytest.raises(ValueError, match="adaptation_scheme"):
        fit_marginal_particle_gibbs(
            Mock(spec_set=SSMModel),
            jnp.zeros((2, 1)),
            jnp.array([0.0, 1.0]),
            adaptation_scheme=cast("Literal['simple', 'dual_averaging']", "bogus"),
        )


@pytest.mark.parametrize(
    ("invalid", "error"),
    [
        ("position_shape", "init_positions must have shape"),
        ("path_shape", "leading dimension num_chains"),
        ("position_finite", r"non-finite for chain\(s\) \[1\]"),
        ("path_finite", r"non-finite for chain\(s\) \[1\]"),
    ],
)
def test_runner_rejects_invalid_initial_chains_before_any_step(monkeypatch, invalid, error):
    positions = np.zeros((2, 1), dtype=np.float32)
    paths = np.zeros((2, 3, 1), dtype=np.float32)
    if invalid == "position_shape":
        positions = positions[:1]
    elif invalid == "path_shape":
        paths = paths[:1]
    elif invalid == "position_finite":
        positions[1, 0] = np.inf
    else:
        paths[1, 1, 0] = np.nan

    target = Mock(
        spec=ParticleRuntime,
        observations=jnp.zeros((3, 1)),
        times=jnp.arange(3.0),
        initial_position=jnp.zeros(1),
        context=Mock(return_value=jnp.array(0.0)),
        initial_path=Mock(side_effect=AssertionError("Supplied paths must not be regenerated")),
        initial_moments=Mock(return_value=(jnp.zeros(1), jnp.eye(1))),
        log_posterior_from_context=lambda z, _context, path, _obs: (
            jnp.sum(z) + jnp.sum(path),
            jnp.sum(path),
        ),
    )
    kernel = Mock(
        spec=MarginalParticleGibbsKernel,
        target_accept=0.35,
        exact_constraints=None,
        adapt_amala_delta=False,
        initial_param_step_size=0.01,
        min_scale=1e-6,
        max_scale=1.0,
    )
    step = Mock(side_effect=AssertionError("Invalid initial states must fail before sampling"))
    monkeypatch.setattr(runner, "_run_batched_step", step)
    with pytest.raises(ValueError, match=error):
        runner.run_marginal_particle_gibbs(
            target,
            kernel=kernel,
            num_warmup=0,
            num_samples=1,
            num_chains=2,
            seed=0,
            adaptation_rate=0.1,
            init_scale=0.0,
            latent_delta=0.2,
            retain_latent_paths=True,
            init_positions=jnp.asarray(positions),
            initial_latent_trajectories=jnp.asarray(paths),
        )
    step.assert_not_called()
    target.initial_path.assert_not_called()
