"""Bind scientific expressions, parameter sites, and node potentials for Dynestyx."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .vector_field import VectorField

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpyro.distributions as dist
    from jax import Array

    from nof1_causal_lab.models.ssm.structure.sites import SemanticBinding, SiteDescriptor

    from .expression import ExpressionComponentSpec

    type PriorFn = Callable[[str], dist.Distribution]


@dataclass(frozen=True, eq=False)
class DynamicsSpec:
    """Declarative SSM dynamics: latent dimension + tuple of component specs."""

    n_latent: int
    components: tuple[ExpressionComponentSpec, ...] = field(default_factory=tuple)


@dataclass(frozen=True, eq=False)
class CompiledDynamics:
    """Output of ``compile_dynamics``: ready-to-fit vector field plus a
    NumPyro-callable that produces the matching ``args.params`` tuple."""

    spec: DynamicsSpec
    vector_field: VectorField
    sample_params: Callable[[PriorFn], tuple[dict[str, Array], ...]]
    site_registry: tuple[SiteDescriptor, ...]
    site_prefix: str = "vf"


def compile_dynamics(spec: DynamicsSpec, *, prefix: str = "vf") -> CompiledDynamics:
    """Compile a ``DynamicsSpec`` into a ``VectorField`` and a
    NumPyro-callable parameter sampler.

    The ``prefix`` is prepended to every NumPyro sample site name so
    nested compositions or multiple SSMs in one model stay disambiguated.
    """
    components = tuple(component_spec.build() for component_spec in spec.components)
    vector_field = VectorField(
        n_latent=spec.n_latent,
        components=components,
        potential_indices=tuple(
            i for i, item in enumerate(spec.components) if item.kind == "potential"
        ),
    )
    component_specs = spec.components
    site_registry = tuple(
        site
        for i, component_spec in enumerate(component_specs)
        for site in component_spec.iter_sites(
            prefix=f"{prefix}_{i}",
            n_latent=spec.n_latent,
        )
    )

    def _sample_all_params(prior_fn: PriorFn) -> tuple[dict[str, Array], ...]:
        return tuple(
            component_spec.sample_params(prefix=f"{prefix}_{i}", prior_fn=prior_fn)
            for i, component_spec in enumerate(component_specs)
        )

    return CompiledDynamics(
        spec=spec,
        vector_field=vector_field,
        sample_params=_sample_all_params,
        site_registry=site_registry,
        site_prefix=prefix,
    )


def iter_dynamics_semantic_bindings(
    spec: DynamicsSpec,
    *,
    latent_names: tuple[str, ...],
    prefix: str = "vf",
) -> Iterator[SemanticBinding]:
    """Yield semantic prior bindings owned by vector-field component specs."""
    for component_index, component_spec in enumerate(spec.components):
        yield from component_spec.iter_semantic_bindings(
            prefix=f"{prefix}_{component_index}",
            latent_names=latent_names,
            component_index=component_index,
        )


def pack_component_params_from_samples(
    spec: DynamicsSpec,
    samples: dict[str, Array],
    *,
    prefix: str = "vf",
) -> tuple[dict[str, Array], ...]:
    """Pack flat sample-site values into the vector-field param tuple.

    This mirrors ``CompiledDynamics.sample_params`` without entering a NumPyro
    sampling context, so post-fit likelihood evaluators can rebuild the same
    runtime dynamics object from constrained parameter draws.
    """
    return tuple(
        component.pack_params(f"{prefix}_{index}", samples)
        for index, component in enumerate(spec.components)
    )
