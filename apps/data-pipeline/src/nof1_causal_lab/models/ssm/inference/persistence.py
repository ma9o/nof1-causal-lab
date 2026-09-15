"""Condition one ModelSpec and derive numerical draws from its current joint law."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform
from nof1_causal_lab.models.model_distributions import joint_distribution_id
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws, ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
)
from nof1_causal_lab.numpyro_json import empirical_atoms, empirical_distribution

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.numpyro_json import ArrayLoader


def condition_model(
    model_spec: ModelSpec,
    result: ParticleMCMCPosterior,
    *,
    times,
    array_writer: Callable[[np.ndarray], str] | None = None,
    array_loader: ArrayLoader | None = None,
) -> ModelSpec:
    """Replace the current uncertainty with the engine's full joint empirical law.

    The original distributions survive in the input revision. No run metadata,
    posterior containers, execution coordinates, or independent fitted marginals
    are attached to the scientific model.
    """
    if not isinstance(result, ParticleMCMCPosterior):
        raise TypeError("Conditioning requires production particle-MCMC output")
    if (
        result.evidence.engine != "marginal_particle_gibbs"
        or result.evidence.latent_transition != "euler_maruyama"
    ):
        raise ValueError("Conditioning requires the exact nonlinear particle-MCMC target")
    bindings, _ = parameter_bindings(model_spec)
    samples = result.get_samples()
    paths = result.draws.latent_paths
    state_ids = numeric.state_ids(model_spec)
    grid = np.asarray(times)
    if paths is None or paths.shape[1:] != (len(grid), len(state_ids)):
        raise ValueError("Conditioning must retain the complete aligned latent trajectories")
    if result.draws.state_ids and tuple(result.draws.state_ids) != tuple(state_ids):
        raise ValueError("Engine latent trajectories do not match the model's state identities")
    columns = [
        np.asarray(samples[coordinate.site_name][(slice(None), *coordinate.indices)])[:, None]
        for binding in sorted(bindings, key=lambda item: item.parameter_id)
        for _, coordinate in sorted(binding.coordinates.items())
    ]
    columns.extend(
        np.asarray(paths[:, :, state_ids.index(identity)]) for identity in sorted(state_ids)
    )
    joint = np.concatenate(columns, axis=1)
    law = empirical_distribution(joint, array_writer=array_writer, array_loader=array_loader)
    identity = joint_distribution_id(
        model_spec,
        [p.id for p in model_spec.parameters if p.value is None],
        state_ids,
        grid.tolist(),
    )
    return model_spec.revised(
        edges=replace_constructs(
            model_spec.edges,
            tuple(
                construct.model_copy(
                    update={"distribution": identity if construct.id in state_ids else None}
                )
                for construct in model_spec.constructs
            ),
        ),
        parameters=tuple(
            parameter
            if parameter.value is not None
            else parameter.model_copy(
                update={
                    "distribution": identity,
                    "distribution_transform": PriorAuthoringTransform.IDENTITY,
                    "reference_interval_days": None,
                }
            )
            for parameter in model_spec.parameters
        ),
        distributions={identity: law},
        time_points=tuple(float(value) for value in grid),
    )


def _scientific_draws(model_spec: ModelSpec) -> JointPosteriorDraws:
    bindings, _ = parameter_bindings(model_spec)
    states = sorted(numeric.state_ids(model_spec))
    members = [
        *[model_spec.parameter(binding.parameter_id) for binding in bindings],
        *[model_spec.get_construct(identity) for identity in states],
    ]
    references = {member.distribution for member in members if member.distribution is not None}
    if len(references) != 1 or any(
        member.distribution is None or member.distribution not in references for member in members
    ):
        raise ValueError("Retained particle draws require all random quantities in one joint law")
    atoms = empirical_atoms(model_spec.distributions[next(iter(references))])
    offset = 0
    parameters = {}
    for binding in sorted(bindings, key=lambda item: item.parameter_id):
        for identity in sorted(binding.coordinates):
            parameters[identity] = jnp.asarray(atoms[:, offset])
            offset += 1
    expected = offset + len(states) * len(model_spec.time_points)
    if atoms.shape[1] != expected:
        raise ValueError("Joint event coordinates do not match the model's scientific quantities")
    paths = (
        atoms[:, offset:]
        .reshape(len(atoms), len(states), len(model_spec.time_points))
        .transpose(0, 2, 1)
    )
    return JointPosteriorDraws(
        parameters=parameters, latent_paths=jnp.asarray(paths), state_ids=tuple(states)
    )


def assemble_parameter_draws(
    model_spec: ModelSpec,
    parameters: dict[str, jnp.ndarray],
    *,
    count: int,
) -> dict[str, jnp.ndarray]:
    """Assemble native tensors from aligned scientific parameter coordinates."""
    bindings, auxiliary = parameter_bindings(model_spec)
    expected = {identity for binding in bindings for identity in binding.coordinates}
    if parameters.keys() != expected:
        raise ValueError("Draws do not match the scientific parameters of this ModelSpec")
    registry = build_site_registry(model_spec)
    # Padding has no scientific interpretation and is never read by an emission.
    # Its canonical completion is zero; every active coordinate is filled below.
    samples = {}
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
        ParameterCoordinate(site_name=site.name, indices=indices)
        for site in registry
        for indices in product(*(range(size) for size in site.shape))
    }
    if covered != all_coordinates:
        raise ValueError("Scientific draws must cover every active native coordinate")
    samples.update(assemble_deterministics_from_registry(samples, model_spec, n_draws=count))
    return samples


def model_draws(model_spec: ModelSpec) -> JointPosteriorDraws:
    """Derive native tensors from the current model's aligned particle distribution."""
    retained = _scientific_draws(model_spec)
    samples = assemble_parameter_draws(
        model_spec, retained.parameters, count=retained.describe().n_draws
    )
    paths = retained.latent_paths
    if paths is not None:
        state_ids = numeric.state_ids(model_spec)
        if set(retained.state_ids) != set(state_ids):
            raise ValueError("Stored trajectories do not match ModelSpec construct identities")
        paths = paths[..., [retained.state_ids.index(identity) for identity in state_ids]]
    return JointPosteriorDraws(
        parameters=samples, latent_paths=paths, state_ids=tuple(numeric.state_ids(model_spec))
    )
