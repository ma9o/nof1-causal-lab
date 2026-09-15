"""Draw the current scientific laws without assigning them a prior/posterior role."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_parameter_law
from nof1_causal_lab.models.ssm.compile.prior_indexing import build_semantic_prior_bindings
from nof1_causal_lab.models.ssm.inference.persistence import assemble_parameter_draws
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

if TYPE_CHECKING:
    import jax

    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def sample_model_laws(model: ModelSpec, *, draws: int, key: jax.Array) -> JointPosteriorDraws:
    """Sample each registered law once, preserving its joint event coordinates.

    Scalar laws supply independent elements under ModelSpec's scalar convention.
    Joint laws already describe native scientific coordinates; trajectory entries
    share the sampled atom with parameters even when a replication regenerates paths.
    """
    model.require_priors()
    bindings = {binding.parameter_id: binding for binding in parameter_bindings(model)[0]}
    semantics = build_semantic_prior_bindings(model).by_parameter
    values: dict[str, jnp.ndarray] = {}
    paths: dict[str, jnp.ndarray] = {}
    for index, (identity, law) in enumerate(sorted(model.distributions.items())):
        members = sorted(
            (parameter for parameter in model.parameters if parameter.distribution == identity),
            key=lambda parameter: parameter.id,
        )
        law_key = random.fold_in(key, index)
        if not law.batch_shape and not law.event_shape:
            parameter = members[0]
            coordinates = sorted(bindings[parameter.id].coordinates)
            native_law, _ = compile_parameter_law(
                model, parameter, semantics[parameter.id], numeric.edge_lag_days(model)
            )
            sampled = native_law.sample(law_key, sample_shape=(draws, len(coordinates)))
            values.update((coordinate, sampled[:, i]) for i, coordinate in enumerate(coordinates))
            continue

        if any(
            parameter.distribution_transform != PriorAuthoringTransform.IDENTITY
            for parameter in members
        ):
            raise ValueError("Joint probability laws must use native scientific coordinates")
        sampled = law.sample(law_key, sample_shape=(draws,))
        offset = 0
        for parameter in members:
            coordinates = sorted(bindings[parameter.id].coordinates)
            for coordinate in coordinates:
                values[coordinate] = sampled[:, offset]
                offset += 1
        for construct in sorted(
            (c for c in model.constructs if c.distribution == identity), key=lambda c: c.id
        ):
            paths[construct.id] = sampled[:, offset : offset + len(model.time_points)]
            offset += len(model.time_points)
    state_ids = tuple(numeric.state_ids(model))
    if paths and set(paths) != set(state_ids):
        raise ValueError("Conditional simulation requires a joint draw for every state")
    return JointPosteriorDraws(
        parameters=assemble_parameter_draws(model, values, count=draws),
        latent_paths=jnp.stack([paths[identity] for identity in state_ids], axis=-1)
        if paths
        else None,
        state_ids=state_ids,
    )
