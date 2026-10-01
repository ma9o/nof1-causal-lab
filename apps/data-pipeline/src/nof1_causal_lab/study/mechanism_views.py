"""Evaluate conditional mechanism curves with the production expression interpreter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from nof1_causal_lab.artifacts.expressions import expression_coefficients, expression_states
from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponent
from nof1_causal_lab.study.visual_models import (
    MechanismCurves,
    MechanismViewRequest,
    ResponseCurve,
)
from nof1_causal_lab.study.visuals import finite_values

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ParameterId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.study.snapshots import ModelReader


def mechanism_parameters(model: ModelSpec, identities: set[ParameterId]):
    """Preserve joint atoms, or sample native current laws with a reproducible plot seed."""
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.compile.prior_compilation import quantity_parameter_law
    from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
    from nof1_causal_lab.numpyro_json import empirical_atoms, materialize_distribution

    parameters = [model.parameter(identity) for identity in sorted(identities)]
    if any(p.distribution is None for p in parameters):
        raise ValueError("Assign probability laws to this mechanism's parameters first")
    if not parameters:
        return {}, "fixed", 1
    laws = {p.distribution for p in parameters if p.distribution is not None}
    native = {
        identity: materialize_distribution(model.distributions[identity]) for identity in laws
    }
    only = next(iter(native.values())) if len(native) == 1 else None
    retained = (
        isinstance(only, dist.MixtureSameFamily)
        and isinstance(only.component_distribution, dist.Delta)
        and np.all(
            np.asarray(only.mixing_distribution.probs)
            == np.asarray(only.mixing_distribution.probs)[0]
        )
    )
    total = len(empirical_atoms(only)) if retained else 128
    values = {}
    for index, identity in enumerate(sorted(laws)):
        law = native[identity]
        members = [p for p in parameters if p.distribution == identity]
        key = jax.random.fold_in(jax.random.PRNGKey(0), index)
        if not law.batch_shape and not law.event_shape:
            for member_index, parameter in enumerate(members):
                compiled, _ = quantity_parameter_law(
                    model, parameter, parameter.distribution_transform
                )
                values[parameter.id] = np.asarray(
                    compiled.sample(jax.random.fold_in(key, member_index), (total,))
                )
        else:
            bindings, _ = parameter_bindings(model)
            by_id = {b.parameter_id: b for b in bindings}
            layout = JointLawLayout.from_bindings(
                bindings,
                parameters=[p.id for p in model.execution_parameters if p.distribution == identity],
                constructs=[c.id for c in model.constructs if c.distribution == identity],
                time_points=model.time_points,
            )
            draws = empirical_atoms(law) if retained else np.asarray(law.sample(key, (total,)))
            for parameter in members:
                coordinates = by_id[parameter.id].coordinates
                if len(coordinates) != 1:
                    raise ValueError(
                        "A scalar drift coefficient requires one scientific coordinate"
                    )
                column = layout.parameter_columns[next(iter(coordinates))]
                values[parameter.id] = draws[:, column]
    return values, "retained" if retained else "sampled", total


def mechanism_curves(reader: ModelReader, request: MechanismViewRequest) -> MechanismCurves:
    model = reader.model
    if model is None:
        raise ValueError("No model at this revision")
    edge = next((item for item in model.edges if item.id == request.owner_id), None)
    construct = next((item for item in model.constructs if item.id == request.owner_id), None)
    if edge is not None:
        mechanisms, target, default_axis = edge.mechanisms, edge.effect, edge.cause.id
    elif construct is not None:
        mechanisms, target, default_axis = construct.dynamics, construct, construct.id
    else:
        raise ValueError("Unknown mechanism owner")
    if not mechanisms:
        raise ValueError("No dynamics are declared for this entity")
    dependencies = {identity for m in mechanisms for identity in expression_states(m.expression)}
    dependencies.add(default_axis)
    axis = request.axis or default_axis
    if axis not in dependencies:
        raise ValueError("The response axis must be a state in this mechanism")
    if request.moderator is not None and (
        request.moderator not in dependencies or request.moderator == axis
    ):
        raise ValueError("The moderator must be another state in this mechanism")
    if request.held.keys() - dependencies or axis in request.held:
        raise ValueError("Held values must name other states in this mechanism")
    held = {
        identity: request.held.get(identity, 0.0)
        for identity in sorted(dependencies - {axis, request.moderator})
    }
    ids = tuple(item.id for item in model.constructs)
    parameters, law, total = mechanism_parameters(
        model,
        {
            operand.value
            for mechanism in mechanisms
            for operand in expression_coefficients(mechanism.expression)
            if isinstance(operand.value, str)
        },
    )
    if request.start >= total:
        raise ValueError("Draw page starts past this law")
    stop = min(request.start + request.count, total)
    components = [
        ExpressionComponent(
            target=ids.index(target.id),
            edge_owned=edge is not None,
            expression=mechanism.expression,
            state_ids=ids,
        )
        for mechanism in mechanisms
    ]

    def response(state, params):
        result = jnp.asarray(0.0)
        for mechanism, component in zip(mechanisms, components, strict=True):
            if mechanism.kind == "potential":
                result = result - jax.grad(component.evaluate)(state, params)[ids.index(target.id)]
            else:
                result = result + component.evaluate(state, params)
        return result

    x = np.linspace(request.lower, request.upper, request.points)
    curves, nonfinite = [], 0
    for level in request.levels if request.moderator is not None else (None,):
        states = np.zeros((request.points, len(ids)))
        for identity, value in held.items():
            states[:, ids.index(identity)] = value
        states[:, ids.index(axis)] = x
        if request.moderator is not None:
            states[:, ids.index(request.moderator)] = level
        evaluate = jax.vmap(jax.vmap(response, in_axes=(0, None)), in_axes=(None, 0))
        draws = {
            identity: jnp.asarray(values[request.start : stop])
            for identity, values in parameters.items()
        }
        if draws:
            values = np.asarray(evaluate(jnp.asarray(states), draws))
        else:
            values = np.asarray(jax.vmap(response, in_axes=(0, None))(jnp.asarray(states), {}))[
                None, :
            ]
        for index, row in enumerate(values):
            nonfinite += int((~np.isfinite(row)).sum())
            curves.append(
                ResponseCurve(draw=request.start + index, level=level, values=finite_values(row))
            )
    return MechanismCurves(
        axis=axis,
        axis_label=model.get_construct(axis).name,
        target_label=target.name,
        states={identity: model.get_construct(identity).name for identity in sorted(dependencies)},
        held=held,
        moderator=request.moderator,
        x=tuple(float(v) for v in x),
        curves=tuple(curves),
        law=law,
        total_draws=total,
        start=request.start,
        count=stop - request.start,
        nonfinite=nonfinite,
    )
