"""Shared inference helpers that do not own concrete backend implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
)
from nof1_causal_lab.numpyro_json import empirical_atoms

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Set

    from numpyro.primitives import Message

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

import logging
from typing import TYPE_CHECKING

from numpyro import handlers

from nof1_causal_lab.models.ssm.constants import INTERNAL_DIAGNOSTIC_SITES

logger = logging.getLogger(__name__)


def _trace_public_sites(
    model_fn: Callable[[], None],
    *,
    exclude: Set[str] | None = None,
) -> frozenset[str]:
    """Trace the original prior and select its parameter sample sites for reporting.

    Reparameterization may later turn these sites into deterministics; replay
    still restores their original names. Assembled matrices are deterministic
    outputs of this original trace, not additional scientific parameters.
    """
    excluded = set(INTERNAL_DIAGNOSTIC_SITES)
    if exclude is not None:
        excluded.update(exclude)

    with handlers.seed(rng_seed=0):
        trace: dict[str, Message] = handlers.trace(model_fn).get_trace()

    return frozenset(
        name
        for name, site in trace.items()
        if site["type"] == "sample" and not site.get("is_observed", False) and name not in excluded
    )


def _filter_public_samples(
    samples: Mapping[str, jnp.ndarray], public_sites: Set[str]
) -> dict[str, jnp.ndarray]:
    """Drop internal handler sites, keeping only original model outputs."""
    return {name: values for name, values in samples.items() if name in public_sites}


def _scientific_draws(model_spec: CompiledModel) -> JointPosteriorDraws:
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

    if len(model_spec.laws) != 1:
        raise ValueError("Retained particle draws require all random quantities in one joint law")
    law = model_spec.laws[0]
    states = tuple(state.id for state in model_spec.states if not state.is_input)
    if set(law.layout.constructs) != set(states):
        raise ValueError("Retained particle draws require all random quantities in one joint law")
    parameters, paths = law.layout.unpack(jnp.asarray(empirical_atoms(law.distribution)))
    return JointPosteriorDraws(
        parameters=dict(parameters.items()),
        latent_paths=jnp.stack([paths[identity] for identity in law.layout.constructs], axis=-1),
        state_ids=law.layout.constructs,
    )


def assemble_parameter_draws(
    model_spec: CompiledModel,
    parameters: Mapping[str, jnp.ndarray],
    *,
    count: int,
) -> dict[str, jnp.ndarray]:
    """Assemble native tensors from aligned scientific parameter coordinates."""
    bindings, auxiliary = parameter_bindings(model_spec)
    expected = {identity for binding in bindings for identity in binding.coordinates}
    if set(parameters) != expected:
        raise ValueError("Draws do not match the scientific parameters of this ModelSpec")
    registry = build_site_registry(model_spec)
    # Padding has no scientific interpretation and is never read by an emission.
    # Its canonical completion is zero; every active coordinate is filled below.
    samples: dict[str, jnp.ndarray] = {}
    for site in registry:
        values = [
            parameters[identity]
            for binding in bindings
            for identity, coordinate in binding.coordinates.items()
            if coordinate.site_name == site.name
        ]
        dtype = jnp.result_type(*values) if values else jnp.float32
        samples[site.name] = jnp.zeros((count, *site.shape), dtype=dtype)
    covered = set(auxiliary)
    for binding in bindings:
        for identity, coordinate in binding.coordinates.items():
            samples[coordinate.site_name] = (
                samples[coordinate.site_name]
                .at[(slice(None), *coordinate.indices)]
                .set(parameters[identity])
            )
            covered.add(coordinate)
    from itertools import product

    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate

    all_coordinates = {
        ParameterCoordinate(site_name=site.name, indices=indices)  # pyright: ignore[reportUnhashable] - Pydantic generates __hash__ from Value's inherited frozen configuration.
        for site in registry
        for indices in product(*(range(size) for size in site.shape))
    }
    if covered != all_coordinates:
        raise ValueError("Scientific draws must cover every active native coordinate")
    samples.update(assemble_deterministics_from_registry(samples, model_spec, n_draws=count))
    return samples


def model_draws(
    model_spec: CompiledModel, *, input_values: jnp.ndarray | None = None
) -> JointPosteriorDraws:
    """Derive native tensors from the current model's aligned particle distribution."""
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

    retained = _scientific_draws(model_spec)
    samples = assemble_parameter_draws(
        model_spec, retained.parameters, count=retained.describe().n_draws
    )
    paths = retained.latent_paths
    state_ids = [state.id for state in model_spec.states if not state.is_input]
    if paths is not None:
        if set(retained.state_ids) != set(state_ids):
            raise ValueError("Stored trajectories do not match ModelSpec construct identities")
        paths = paths[..., [retained.state_ids.index(identity) for identity in state_ids]]
        if numeric.input_mask(model_spec).any():
            if input_values is None:
                raise ValueError(
                    "Retained histories with exogenous inputs require the replayed panel path"
                )
            full_ids = numeric.state_ids(model_spec)
            full_paths = jnp.broadcast_to(input_values, (paths.shape[0], *input_values.shape))
            paths = full_paths.at[
                :, :, jnp.asarray([full_ids.index(identity) for identity in state_ids])
            ].set(paths)
            state_ids = full_ids
    return JointPosteriorDraws(parameters=samples, latent_paths=paths, state_ids=tuple(state_ids))
