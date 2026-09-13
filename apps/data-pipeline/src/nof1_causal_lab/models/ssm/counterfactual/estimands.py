"""Estimand helpers: posterior summaries and temporal extracts.

Pure functions over trajectories and posterior draws. No JAX primitives
specific to the linear case; safe to use unchanged once the non-linear
vector fields land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from nof1_causal_lab.artifacts.effects import (
    EffectSummary,
    EffectTrajectoryPoint,
    TemporalEffect,
    validate_effect_horizons,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from jax import Array


def summarize_draws(draws: Array) -> EffectSummary:
    """Posterior summary statistics for a 1-D array of effect draws."""
    if draws.ndim != 1 or draws.size == 0:
        raise ValueError("An effect summary requires a nonempty vector of posterior draws")
    return EffectSummary(
        mean=float(jnp.mean(draws)),
        median=float(jnp.median(draws)),
        lower_95=float(jnp.quantile(draws, 0.025)),
        upper_95=float(jnp.quantile(draws, 0.975)),
        prob_positive=float(jnp.mean(draws > 0)),
    )


def summarize_temporal_effect(
    effect_trajectory: Array,
    time_grid: Array,
    *,
    horizons_days: Sequence[float],
) -> TemporalEffect:
    """Summarize a trajectory at explicit elapsed-day horizons and its absolute peak.

    Grid points are measured in days. Interpolate within the simulated interval;
    a requested horizon outside that interval is not an observed temporal effect.
    """
    validate_effect_horizons(horizons_days)
    t = jnp.asarray(time_grid)
    traj = jnp.asarray(effect_trajectory)
    if t.ndim != 1 or t.size == 0 or traj.shape != t.shape:
        raise ValueError("Temporal effects require matching nonempty time and effect vectors")
    if not bool(jnp.all(jnp.isfinite(t)) & jnp.all(jnp.isfinite(traj))):
        raise ValueError("Temporal effect times and values must be finite")
    if not bool(jnp.all(jnp.diff(t) > 0)):
        raise ValueError("Temporal effect times must be strictly increasing")
    days = t - t[0]
    if horizons_days[-1] > float(days[-1]):
        raise ValueError("Requested horizons must lie within the simulated interval")
    peak_idx = int(jnp.argmax(jnp.abs(traj)))
    return TemporalEffect(
        horizons=[
            EffectTrajectoryPoint(day=day, effect=float(jnp.interp(day, days, traj)))
            for day in horizons_days
        ],
        peak_effect=float(traj[peak_idx]),
        time_to_peak_days=float(days[peak_idx]),
    )


def build_time_grid(t_start: float, t_end: float, dt: float) -> Array:
    """Inclusive uniform grid from ``t_start`` to ``t_end`` at spacing ``dt``."""
    n_steps = int(jnp.ceil((t_end - t_start) / dt)) + 1
    return jnp.linspace(t_start, t_end, num=n_steps)
