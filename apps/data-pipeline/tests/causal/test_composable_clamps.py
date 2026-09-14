"""Tests for composable, windowed latent clamps (analysis unified simulation).

Covers ``vmap_simulate_clamps_from_state`` + ``build_segment_bounds``: full-horizon
clamps, finite windows that release to natural dynamics, mid-rollout onset (the
segmentation correctness check), multiple simultaneous clamps, trajectory clamps,
and baseline vs abducted start states.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from nof1_causal_lab.models.ssm.counterfactual import (
    ClampSpec,
    build_segment_bounds,
    vmap_simulate_clamps_from_state,
)
from nof1_causal_lab.models.ssm.dynamics import VectorField
from nof1_causal_lab.models.ssm.dynamics.edges import DenseLinear

# var1 is driven by var0; both stable. Baseline steady state is η* = -A⁻¹c = [1, 1].
_PARAMS = ({"drift": jnp.array([[-1.0, 0.0], [0.5, -1.0]]), "cint": jnp.array([1.0, 0.5])},)
_STEADY_STATE = jnp.array([[1.0, 1.0]])
_TIME_GRID = jnp.linspace(0.0, 12.0, 13)  # daily grid, day == index


def _vf() -> VectorField:
    return VectorField(n_latent=2, components=(DenseLinear(),))


def _run(clamps, initial_states=_STEADY_STATE):
    baseline, action, effect = vmap_simulate_clamps_from_state(
        _vf(), [_PARAMS], initial_states=initial_states, clamps=clamps, time_grid=_TIME_GRID
    )
    return baseline[0], action[0], effect[0]  # single draw → (T, n_latent)


def test_segment_bounds_split_at_window_edges():
    clamps = [
        ClampSpec(index=0, mode="set", value=0.5, from_day=0.0, to_day=14.0),
        ClampSpec(index=1, mode="shift", amount=1.0, from_day=20.0),
    ]
    grid = jnp.arange(31.0)
    assert build_segment_bounds(grid, clamps) == [(0, 14), (14, 20), (20, 30)]
    # No windows → a single segment.
    assert build_segment_bounds(grid, [ClampSpec(index=0, mode="shift", amount=1.0)]) == [(0, 30)]


@pytest.mark.simulation
def test_full_horizon_shift_holds_and_propagates():
    _baseline, action, effect = _run(
        [ClampSpec(index=0, mode="shift", amount=1.0)], initial_states=None
    )
    # var0 held at baseline(1) + 1 = 2 across the whole horizon.
    assert jnp.allclose(action[:, 0], 2.0, atol=0.05)
    # Positive coupling lifts var1 by the end.
    assert float(effect[-1, 1]) > 0.1
    assert jnp.isclose(effect[-1, 0], 1.0, atol=0.05)


@pytest.mark.simulation
def test_window_clamp_pins_at_onset_and_releases_to_natural():
    # The segmentation correctness check: a set clamp opening at day 2 must JUMP to
    # the value at day 2 (not merely hold its slope from the natural value).
    _baseline, action, _effect = _run(
        [ClampSpec(index=0, mode="set", value=3.0, from_day=2.0, to_day=4.0)]
    )
    assert jnp.isclose(action[1, 0], 1.0, atol=0.05)
    assert jnp.allclose(action[2:5, 0], 3.0, atol=0.05)
    # After release, x(t) = 1 + 2 exp(-(t - 4)); this checks the decay,
    # rather than only its eventual equilibrium.
    expected_release = 1.0 + 2.0 * jnp.exp(-(_TIME_GRID[4:] - 4.0))
    assert jnp.allclose(action[4:, 0], expected_release, atol=0.01)


@pytest.mark.simulation
def test_multiple_simultaneous_clamps():
    _baseline, action, _effect = _run(
        [
            ClampSpec(index=0, mode="set", value=3.0),
            ClampSpec(index=1, mode="set", value=5.0),
        ]
    )
    assert jnp.allclose(action[:, 0], 3.0, atol=0.05)
    assert jnp.allclose(action[:, 1], 5.0, atol=0.05)


@pytest.mark.simulation
def test_trajectory_clamp_tracks_values():
    _baseline, action, _effect = _run(
        [ClampSpec(index=0, mode="trajectory", values=(1.0, 3.0), from_day=0.0, to_day=10.0)]
    )
    # Linear interpolation 1 → 3 across [0, 10].
    assert jnp.isclose(action[0, 0], 1.0, atol=0.05)
    assert jnp.isclose(action[5, 0], 2.0, atol=0.1)
    assert jnp.isclose(action[10, 0], 3.0, atol=0.05)


@pytest.mark.simulation
def test_abducted_start_evolves_from_given_state():
    initial_states = jnp.array([[0.0, 0.0]])
    baseline, _action, _effect = _run(
        [ClampSpec(index=0, mode="shift", amount=1.0)], initial_states=initial_states
    )
    # Reference path starts at the abducted state and relaxes toward steady state [1, 1].
    assert jnp.allclose(baseline[0], jnp.array([0.0, 0.0]), atol=0.05)
    assert jnp.allclose(baseline[-1], jnp.array([1.0, 1.0]), atol=0.1)
