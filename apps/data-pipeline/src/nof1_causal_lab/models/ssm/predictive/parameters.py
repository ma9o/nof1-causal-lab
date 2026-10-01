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
from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
from nof1_causal_lab.numpyro_json import distribution_shape

if TYPE_CHECKING:
    import jax

    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def validate_simulation_laws(model: ModelSpec) -> None:
    """Check the general simulator's law capabilities without sampling or fitting."""
    model.require_priors()
    semantics = build_semantic_prior_bindings(model).by_parameter
    for parameter in model.execution_parameters:
        assert parameter.distribution is not None
        law = model.distributions[parameter.distribution]
        if not any(distribution_shape(law)):
            compile_parameter_law(model, parameter, semantics[parameter.id])
        elif parameter.distribution_transform != PriorAuthoringTransform.IDENTITY:
            raise ValueError("Joint probability laws must use native scientific coordinates")
    retained = {c.id for c in model.constructs if c.distribution is not None}
    state_ids = {
        identity
        for identity in numeric.state_ids(model)
        if model.get_construct(identity).role == "endogenous"
    }
    if retained & state_ids and not state_ids <= retained:
        raise ValueError("Conditional simulation requires a joint draw for every state")


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
    state_ids = tuple(
        identity
        for identity in numeric.state_ids(model)
        if model.get_construct(identity).role == "endogenous"
    )
    active_laws = {
        member.distribution
        for member in (
            *model.execution_parameters,
            *(model.get_construct(identity) for identity in state_ids),
        )
        if member.distribution is not None
    }
    for index, (identity, law) in enumerate(sorted(model.distributions.items())):
        if identity not in active_laws:
            continue
        members = [
            parameter
            for parameter in model.execution_parameters
            if parameter.distribution == identity
        ]
        law_key = random.fold_in(key, index)
        if not law.batch_shape and not law.event_shape:
            parameter = members[0]
            coordinates = sorted(bindings[parameter.id].coordinates)
            native_law, _ = compile_parameter_law(model, parameter, semantics[parameter.id])
            sampled = native_law.sample(law_key, sample_shape=(draws, len(coordinates)))
            values.update((coordinate, sampled[:, i]) for i, coordinate in enumerate(coordinates))
            continue

        if any(
            parameter.distribution_transform != PriorAuthoringTransform.IDENTITY
            for parameter in members
        ):
            raise ValueError("Joint probability laws must use native scientific coordinates")
        sampled = law.sample(law_key, sample_shape=(draws,))
        layout = JointLawLayout.from_bindings(
            bindings.values(),
            parameters=[parameter.id for parameter in members],
            constructs=[c.id for c in model.constructs if c.distribution == identity],
            time_points=model.time_points,
        )
        parameter_draws, trajectories = layout.unpack(sampled)
        values.update(parameter_draws.items())
        paths.update(trajectories.items())
    if paths and set(paths) != set(state_ids):
        raise ValueError("Conditional simulation requires a joint draw for every state")
    return JointPosteriorDraws(
        parameters=assemble_parameter_draws(model, values, count=draws),
        latent_paths=jnp.stack([paths[identity] for identity in state_ids], axis=-1)
        if paths
        else None,
        state_ids=state_ids,
    )
