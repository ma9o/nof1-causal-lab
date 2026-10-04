"""Condition one ModelSpec and derive numerical draws from its current joint law."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import joint_law_layout
from nof1_causal_lab.numpyro_json import empirical_distribution

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import ArrayLike

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.identity import DistributionId
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.inference.types import ParticleMCMCPosterior
    from nof1_causal_lab.numpyro_json import ArrayLoader


def condition_model(
    model_spec: ModelSpec,
    compiled: CompiledModel,
    result: ParticleMCMCPosterior,
    *,
    times: ArrayLike,
    array_writer: Callable[[np.ndarray], str] | None = None,
    array_loader: ArrayLoader | None = None,
) -> tuple[ModelSpec, DistributionId]:
    """Replace the current uncertainty with the engine's full joint empirical law.

    The original distributions survive in the input revision. No run metadata,
    posterior containers, compiler bindings, or independent fitted marginals
    are attached to the scientific model.
    """
    bindings = compiled.bindings
    samples = result.get_samples()
    paths = result.draws.latent_paths
    state_ids = numeric.state_ids(compiled)
    modeled_ids = [
        identity
        for identity in state_ids
        if model_spec.get_construct(identity).role == "endogenous"
    ]
    grid = np.asarray(times)
    if paths is None or paths.shape[1:] != (len(grid), len(state_ids)):
        raise ValueError("Conditioning must retain the complete aligned latent trajectories")
    if result.draws.state_ids and tuple(result.draws.state_ids) != tuple(state_ids):
        raise ValueError("Engine latent trajectories do not match the model's state identities")
    conditioned_parameters = {binding.parameter_id for binding in bindings}
    layout = joint_law_layout(
        bindings,
        parameters=conditioned_parameters,
        constructs=modeled_ids,
        time_points=grid.tolist(),
        construct_labels={
            identity: model_spec.get_construct(identity).name for identity in modeled_ids
        },
    )
    joint = layout.pack(
        {
            identity: samples[coordinate.site_name][(slice(None), *coordinate.indices)]
            for binding in bindings
            for identity, coordinate in binding.coordinates.items()
        },
        {identity: paths[:, :, index] for index, identity in enumerate(state_ids)},
    )
    law = empirical_distribution(joint, array_writer=array_writer, array_loader=array_loader)
    identity = layout.distribution_id
    edges = replace_constructs(
        model_spec.edges,
        tuple(
            construct.with_distribution(
                identity if construct.id in modeled_ids else construct.distribution
            )
            for construct in model_spec.constructs
        ),
    )
    parameters = tuple(
        parameter if parameter.id not in conditioned_parameters else parameter.conditioned(identity)
        for parameter in model_spec.parameters
    )
    retained_laws = {
        member.distribution
        for member in (
            *parameters,
            *(edge.cause for edge in edges),
            *(edge.effect for edge in edges),
        )
        if member.distribution is not None
    }
    conditioned = model_spec.revised(
        edges=edges,
        parameters=parameters,
        distributions={
            **{
                key: value
                for key, value in model_spec.distributions.items()
                if key in retained_laws
            },
            identity: law,
        },
        law_layouts={
            **{key: value for key, value in model_spec.law_layouts.items() if key in retained_laws},
            identity: layout,
        },
    )
    return conditioned, identity
