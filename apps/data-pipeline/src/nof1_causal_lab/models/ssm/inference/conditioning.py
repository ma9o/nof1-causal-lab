"""Compile exact point measurements into fixed coordinates of the sampled path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.expressions import StateExpression
from nof1_causal_lab.models.ssm import numerics as numeric

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


@dataclass(frozen=True)
class ExactStateConstraints:
    """Observed state values on the model grid; NaN coordinates remain unknown."""

    values: jnp.ndarray

    @property
    def free_mask(self) -> jnp.ndarray:
        return jnp.isnan(self.values)

    def project(self, path: jnp.ndarray) -> jnp.ndarray:
        """Substitute observations in one path or a batch of initialization paths."""
        if path.shape[-2:] != self.values.shape:
            raise ValueError(
                "Exact observations and latent paths must use the same time/state grid"
            )
        return jnp.where(self.free_mask, path, self.values)


def compile_exact_state_constraints(
    spec: ModelSpec, observations: jnp.ndarray
) -> ExactStateConstraints | None:
    """Condition direct state bindings without discarding their dynamics density."""
    exact = [
        (index, indicator)
        for index, indicator in enumerate(numeric.observed_indicators(spec))
        if indicator.likelihood is not None and indicator.likelihood.law.distribution == "Delta"
    ]
    if not exact:
        return None
    observed = np.asarray(observations)
    values = np.full((len(observed), numeric.n_states(spec)), np.nan, dtype=observed.dtype)
    state_indices = {identity: index for index, identity in enumerate(spec.state_order)}
    for column, indicator in exact:
        assert indicator.likelihood is not None
        expression = indicator.likelihood.law.arguments["v"]
        if indicator.support_kind != "point" or not isinstance(expression, StateExpression):
            raise ValueError(
                f"Delta indicator {indicator.name!r} requires a direct point binding "
                "Delta(v=state(...)) for particle inference; affine and interval constraints "
                "are not supported."
            )
        index = state_indices[expression.construct_id]
        readings = observed[:, column]
        if np.isinf(readings).any():
            raise ValueError(f"Exact observations for {indicator.name!r} must be finite or missing")
        present = ~np.isnan(readings)
        conflict = present & ~np.isnan(values[:, index]) & (readings != values[:, index])
        if conflict.any():
            raise ValueError(
                f"Conflicting exact observations for {expression.construct_id!r} "
                f"at model rows {np.flatnonzero(conflict).tolist()}"
            )
        values[present, index] = readings[present]
    return ExactStateConstraints(jnp.asarray(values))
