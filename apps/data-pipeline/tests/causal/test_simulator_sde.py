"""Diffrax SDE replay, noise moments, and deterministic-limit contracts."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import jit, vmap

from nof1_causal_lab.models.ssm.dynamics import (
    Intervention,
    SimulationConfig,
    VectorField,
    simulate,
)
from nof1_causal_lab.models.ssm.dynamics.edges import DenseLinear


def _linear_setup():
    vf = VectorField(n_latent=1, components=(DenseLinear(),))
    params = ({"drift": jnp.array([[-1.0]]), "cint": jnp.array([0.5])},)
    return vf, params, jnp.zeros(1), jnp.linspace(0.0, 2.0, 9)


class TestSimulateSDEMode:
    @pytest.mark.simulation
    @pytest.mark.parametrize("indexed", [False, True], ids=["brownian-tree", "indexed"])
    def test_replay_preserves_keys_and_produces_finite_distinct_paths(self, indexed):
        vf, params, y0, time_grid = _linear_setup()
        config = SimulationConfig(sde_dt=0.02, use_indexed_brownian_path=indexed)
        keys = jnp.stack([jr.PRNGKey(19), jr.PRNGKey(19), jr.PRNGKey(20)])
        trajectories = jit(
            vmap(
                lambda key: simulate(
                    vf,
                    params,
                    Intervention.none(),
                    y0,
                    time_grid,
                    config=config,
                    key=key,
                    diffusion_cov=jnp.eye(1) * 0.2,
                )
            )
        )(keys)

        assert trajectories.shape == (3, len(time_grid), 1)
        assert bool(jnp.isfinite(trajectories).all())
        assert jnp.array_equal(trajectories[0], trajectories[1])
        assert not jnp.allclose(trajectories[0], trajectories[2], atol=1e-3)

    @pytest.mark.simulation
    @pytest.mark.parametrize("indexed", [False, True], ids=["brownian-tree", "indexed"])
    def test_sample_moments_match_ornstein_uhlenbeck_solution(self, indexed):
        vf, params, y0, time_grid = _linear_setup()
        config = SimulationConfig(sde_dt=0.02, use_indexed_brownian_path=indexed)
        # One compiled batch checks both drift and diffusion against the analytic law.
        samples = jit(
            vmap(
                lambda key: simulate(
                    vf,
                    params,
                    Intervention.none(),
                    y0,
                    time_grid,
                    config=config,
                    key=key,
                    diffusion_cov=jnp.eye(1) * 0.2,
                )
            )
        )(jr.split(jr.PRNGKey(23), 512))
        final_states = samples[:, -1, 0]
        expected_mean = 0.5 * (1.0 - jnp.exp(-2.0))
        expected_variance = 0.1 * (1.0 - jnp.exp(-4.0))

        assert jnp.mean(final_states) == pytest.approx(float(expected_mean), abs=0.05)
        assert jnp.var(final_states) == pytest.approx(float(expected_variance), abs=0.02)

    @pytest.mark.simulation
    def test_zero_diffusion_matches_ode(self):
        vf, params, y0, time_grid = _linear_setup()
        det = simulate(vf, params, Intervention.none(), y0, time_grid)
        sde = simulate(
            vf,
            params,
            Intervention.none(),
            y0,
            time_grid,
            key=jr.PRNGKey(42),
            diffusion_cov=jnp.eye(1) * 1e-10,
        )
        assert jnp.allclose(det, sde, atol=5e-3)

    def test_requires_both_key_and_diffusion(self):
        vf, params, y0, time_grid = _linear_setup()
        with pytest.raises(ValueError, match="SDE mode requires both"):
            simulate(vf, params, Intervention.none(), y0, time_grid, key=jr.PRNGKey(0))
        with pytest.raises(ValueError, match="SDE mode requires both"):
            simulate(vf, params, Intervention.none(), y0, time_grid, diffusion_cov=jnp.eye(1))

    @pytest.mark.simulation
    def test_sde_config_overrides_step_size(self):
        vf, params, y0, time_grid = _linear_setup()
        trajectories = [
            simulate(
                vf,
                params,
                Intervention.none(),
                y0,
                time_grid,
                config=SimulationConfig(sde_dt=dt),
                key=jr.PRNGKey(0),
                diffusion_cov=jnp.eye(1) * 0.1,
            )
            for dt in (0.02, 0.005)
        ]
        assert all(bool(jnp.isfinite(path).all()) for path in trajectories)
        assert float(jnp.max(jnp.abs(trajectories[0] - trajectories[1]))) > 1e-4
