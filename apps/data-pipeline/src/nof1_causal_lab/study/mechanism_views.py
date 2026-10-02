"""Render exact conditional response curves from resolved view inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.study.visual_models import MechanismCurves, MechanismViewRequest, ResponseCurve
from nof1_causal_lab.study.visuals import finite_values

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from nof1_causal_lab.artifacts.identity import ConstructId, ParameterId
    from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponent


def mechanism_curves(
    request: MechanismViewRequest,
    components: tuple[ExpressionComponent, ...],
    kinds: tuple[Literal["drift", "potential"], ...],
    ids: tuple[ConstructId, ...],
    target: ConstructId,
    target_label: str,
    axis: ConstructId,
    labels: Mapping[ConstructId, str],
    held: Mapping[ConstructId, float],
    parameters: Mapping[ParameterId, np.ndarray],
    law: Literal["retained", "sampled", "fixed"],
    total: int,
) -> MechanismCurves:
    stop = min(request.start + request.count, total)

    def response(state: jax.Array, params: Mapping[str, jax.Array]) -> jax.Array:
        result = jnp.asarray(0.0)
        for kind, component in zip(kinds, components, strict=True):
            if kind == "potential":
                gradient: Callable[[jax.Array, Mapping[str, jax.Array]], jax.Array] = jax.grad(
                    component.evaluate
                )
                result = result - gradient(state, params)[ids.index(target)]
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
        draws: dict[str, jax.Array] = {
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
        axis_label=labels[axis],
        target_label=target_label,
        states=labels,
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
