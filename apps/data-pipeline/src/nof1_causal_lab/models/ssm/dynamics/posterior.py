"""Posterior sample assembly for vector-field dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.ssm import numerics as numeric

from .spec import compile_dynamics, pack_component_params_from_samples

if TYPE_CHECKING:
    from jax import Array

    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    from .vector_field import VectorField


@dataclass(frozen=True)
class PosteriorDynamicsSamples:
    """Vector-field posterior draws reconstructed from canonical sites."""

    vector_field: VectorField
    param_samples: list[tuple[dict[str, Array], ...]]


def _posterior_draw_count(samples: UncheckedJsonObject) -> int:
    if not samples:
        return 0
    for values in samples.values():
        if hasattr(values, "shape") and len(values.shape) > 0:
            return int(values.shape[0])
    return 0


def component_param_samples_from_site_samples(
    spec: ModelSpec,
    samples: UncheckedJsonObject,
    *,
    prefix: str = "vf",
) -> list[tuple[dict[str, Array], ...]]:
    """Pack posterior site samples into per-draw vector-field params."""
    n_draws = _posterior_draw_count(samples)
    param_samples: list[tuple[dict[str, Array], ...]] = []
    for draw_idx in range(n_draws):
        draw = {
            name: jnp.asarray(values)[draw_idx]
            for name, values in samples.items()
            if hasattr(values, "shape") and len(values.shape) > 0
        }
        param_samples.append(
            pack_component_params_from_samples(
                numeric.dynamics_components(spec), draw, prefix=prefix
            )
        )
    return param_samples


def posterior_dynamics_from_samples(
    spec: ModelSpec,
    samples: UncheckedJsonObject,
    *,
    prefix: str = "vf",
) -> PosteriorDynamicsSamples:
    """Rebuild posterior vector-field draws from an ``ModelSpec`` and samples."""
    compiled = compile_dynamics(numeric.dynamics_components(spec), prefix=prefix)
    return PosteriorDynamicsSamples(
        vector_field=compiled.vector_field,
        param_samples=component_param_samples_from_site_samples(
            spec,
            samples,
            prefix=prefix,
        ),
    )
