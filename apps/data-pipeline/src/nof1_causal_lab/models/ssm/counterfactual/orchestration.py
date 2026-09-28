"""Paired nonlinear histories with dated state assignments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from nof1_causal_lab.models.ssm.dynamics import (
    Intervention,
    ProcessNoise,
    SimulationConfig,
    simulate,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.scenarios import InterventionSpec
    from nof1_causal_lab.models.ssm.dynamics.draws import DynamicsDraws


@dataclass(frozen=True)
class ResolvedIntervention:
    """A dated intervention bound to its numerical state axis."""

    index: int
    spec: InterventionSpec


def build_segment_bounds(
    time_grid: Array, interventions: list[ResolvedIntervention]
) -> list[tuple[int, int]]:
    """Split at exact event times; the simulation compiler inserts every boundary."""
    grid = np.asarray(time_grid)
    boundaries = {float(grid[0]), float(grid[-1])}
    boundaries.update(event.spec.time for event in interventions)
    coordinates = grid.tolist()
    indices = sorted(coordinates.index(float(np.asarray(t, dtype=grid.dtype))) for t in boundaries)
    return list(pairwise(indices))


def _apply_events(state: Array, events: tuple[tuple[int, float], ...]) -> Array:
    for index, value in events:
        state = state.at[index].set(value)
    return state


def vmap_simulate_interventions_from_state(
    dynamics: DynamicsDraws,
    initial_states: Array,
    interventions: list[ResolvedIntervention],
    *,
    time_grid: Array,
    config: SimulationConfig | None = None,
    noise: ProcessNoise | None = None,
) -> tuple[Array, Array, Array]:
    """Generate paired natural/intervened histories with shared process streams."""
    n_latent = dynamics.vector_field.n_latent
    if initial_states.shape != (dynamics.n_draws, n_latent):
        raise ValueError("Initial states must match the dynamics draw and state axes")
    if noise is not None:
        if noise.key.ndim < 1 or noise.key.shape[0] != dynamics.n_draws:
            raise ValueError("Process keys must match the dynamics draw axis")
        if noise.diffusion_cov.shape != (dynamics.n_draws, n_latent, n_latent):
            raise ValueError("Diffusion covariance must match the dynamics draw and state axes")
    if dynamics.n_draws == 0:
        empty = jnp.zeros((0, time_grid.shape[0], n_latent))
        return empty, empty, empty
    segments = build_segment_bounds(time_grid, interventions)
    grid = np.asarray(time_grid)
    sets = {
        i: tuple(
            (event.index, event.spec.value)
            for event in interventions
            if np.asarray(event.spec.time, dtype=grid.dtype) == time
        )
        for i, time in enumerate(grid)
    }

    def path(params, y0, process_noise: ProcessNoise | None, active):
        pieces = []
        state = y0
        for segment, (i0, i1) in enumerate(segments):
            if active:
                state = _apply_events(state, sets[i0])
            ys = simulate(
                dynamics.vector_field,
                params,
                Intervention.none(),
                state,
                time_grid[i0 : i1 + 1],
                config,
                noise=(
                    ProcessNoise(
                        key=jax.random.fold_in(process_noise.key, segment),
                        diffusion_cov=process_noise.diffusion_cov,
                    )
                    if process_noise is not None
                    else None
                ),
            )
            pieces.append(ys[:-1] if segment < len(segments) - 1 else ys)
            state = ys[-1]
        result = jnp.concatenate(pieces, axis=0)
        if active:
            # A point event at the destination still changes the reported end state.
            result = result.at[-1].set(_apply_events(result[-1], sets[len(grid) - 1]))
        return result

    def per_draw(params, y0, process_noise: ProcessNoise | None):
        reference = path(params, y0, process_noise, False)
        action = path(params, y0, process_noise, True) if interventions else reference
        return reference, action, action - reference

    return jax.vmap(per_draw, axis_size=dynamics.n_draws)(
        dynamics.parameters, initial_states, noise
    )
