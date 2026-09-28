"""Batched vector-field parameters reconstructed from canonical numerical sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.models.ssm import numerics as numeric

from .spec import compile_dynamics, pack_component_params_from_samples

if TYPE_CHECKING:
    from jax import Array

    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    from .vector_field import VectorField


@dataclass(frozen=True)
class DynamicsDraws:
    """One vector field and component parameters with a shared leading draw axis.

    The explicit count also describes fully fixed fields, whose parameter pytree
    has no leaves for ``vmap`` to infer a batch size from.
    """

    vector_field: VectorField
    parameters: tuple[dict[str, Array], ...]
    n_draws: int

    def __post_init__(self) -> None:
        if self.n_draws < 0:
            raise ValueError("Dynamics draw count must be non-negative")
        if len(self.parameters) != len(self.vector_field.components):
            raise ValueError("Dynamics parameters must match the vector-field components")
        for component in self.parameters:
            for name, values in component.items():
                if values.ndim < 1 or values.shape[0] != self.n_draws:
                    raise ValueError(f"{name} must share the dynamics draw axis")


def dynamics_from_samples(
    spec: ModelSpec,
    samples: dict[str, Array],
    *,
    n_draws: int,
    prefix: str = "vf",
) -> DynamicsDraws:
    """Pack already batched site arrays without unstacking individual draws."""
    dynamics = numeric.dynamics_components(spec)
    return DynamicsDraws(
        vector_field=compile_dynamics(dynamics, prefix=prefix).vector_field,
        parameters=pack_component_params_from_samples(dynamics, samples, prefix=prefix),
        n_draws=n_draws,
    )
