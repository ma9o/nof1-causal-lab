"""Bind native observation operands to compiled coordinates and the cached predictor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from nof1_causal_lab.artifacts.expressions import Expression, fold_expression
from nof1_causal_lab.artifacts.likelihood import Law, LinkFunction, map_law
from nof1_causal_lab.models.ssm.dynamics.expression import (
    SCALAR_OPERATIONS,
    BoundExpression,
    apply_expression_function,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import jax

    from nof1_causal_lab.artifacts.expressions import (
        BinaryOperator,
        CoefficientExpression,
        ExpressionFunction,
    )
    from nof1_causal_lab.artifacts.identity import ParameterId
    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
    from nof1_causal_lab.models.likelihoods import LikelihoodAnalysis
    from nof1_causal_lab.models.ssm.compile.bindings import CompiledParameterBinding
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.dynamics.expression import OperandEvaluator


def _inverse(eta: jax.Array) -> jax.Array:
    valid = jnp.isfinite(eta) & (eta > 0.0)
    return jnp.where(valid, 1.0 / jnp.where(valid, eta, 1.0), jnp.nan)


def _identity(eta: jax.Array) -> jax.Array:
    return eta


_RESPONSES: dict[LinkFunction, Callable[[jax.Array], jax.Array]] = {
    LinkFunction.IDENTITY: _identity,
    LinkFunction.LOG: lambda eta: apply_expression_function("exp", (eta,)),
    LinkFunction.INVERSE: _inverse,
    LinkFunction.LOGIT: lambda eta: apply_expression_function("sigmoid", (eta,)),
    LinkFunction.PROBIT: lambda eta: apply_expression_function("normal_cdf", (eta,)),
    LinkFunction.CUMULATIVE_LOGIT: _identity,
    LinkFunction.SOFTMAX: _identity,
}


def _positive_domain(raw: OperandEvaluator) -> OperandEvaluator:
    """The inverse response's positive predictor domain is a scientific boundary."""

    def evaluate(eta: jax.Array, scale: jax.Array, values: tuple[jax.Array, ...]) -> jax.Array:
        valid = jnp.isfinite(eta) & (eta > 0.0)
        result = raw(jnp.where(valid, eta, 1.0), scale, values)
        return jnp.where(valid, result, jnp.nan)

    return evaluate


def bind_observation_law(
    law: Law[Expression],
    terms: LikelihoodAnalysis,
    bindings: Mapping[ParameterId, CompiledParameterBinding],
    *,
    channel: int,
    category_count: int,
    sampling_count: int,
    categorical_anchor: bool,
) -> Law[BoundExpression]:
    """Bind all IDs once; execution evaluates only resolved gathers and arithmetic."""
    response = _RESPONSES[terms.link]

    def bind(expression: Expression) -> BoundExpression:
        coordinates: list[ParameterCoordinate] = []

        def coefficient(operand: CoefficientExpression) -> OperandEvaluator:
            if operand.role == "observation_scale":
                return lambda _eta, scale, _values: scale
            value = operand.value
            if value is None:
                raise ValueError("Compilation requires assigned observation operands")
            if isinstance(value, (int, float)):
                return lambda _eta, _scale, _values: jnp.asarray(value)
            binding = bindings[value]
            selected = sorted(
                binding.coordinates.values(), key=lambda coordinate: coordinate.indices
            )
            vector = operand.role in {"cutpoint_gaps", "category_intercepts", "category_slopes"}
            width = category_count - (2 if operand.role == "cutpoint_gaps" else 1)
            if operand.role in {
                "cutpoint_base",
                "cutpoint_gaps",
                "category_intercepts",
                "category_slopes",
            }:
                selected = [
                    coordinate for coordinate in selected if coordinate.indices[0] == channel
                ]
            # An anchored category's first slope is pinned to one; its sampled slot is unread.
            pinned = categorical_anchor and operand.role == "category_slopes"
            if vector:
                selected = [
                    coordinate
                    for coordinate in selected
                    if int(pinned) <= coordinate.indices[-1] < width
                ]
            start = len(coordinates)
            coordinates.extend(selected)
            stop = len(coordinates)
            if not vector:
                if stop - start != 1:
                    raise ValueError(f"{operand.role} must bind to one native coordinate")
                return lambda _eta, _scale, values: values[start]
            if stop - start != width - int(pinned):
                raise ValueError(f"{operand.role} must bind every declared category coordinate")

            def gather(
                _eta: jax.Array, _scale: jax.Array, values: tuple[jax.Array, ...]
            ) -> jax.Array:
                result = jnp.stack(values[start:stop]) if stop > start else jnp.empty((0,))
                return (
                    jnp.concatenate((jnp.ones((1,), dtype=result.dtype), result))
                    if pinned
                    else result
                )

            return gather

        def call(
            name: ExpressionFunction, arguments: tuple[OperandEvaluator, ...]
        ) -> OperandEvaluator:
            if name == "ordered_cutpoints":

                def cutpoints(
                    eta: jax.Array, scale: jax.Array, values: tuple[jax.Array, ...]
                ) -> jax.Array:
                    base = jnp.atleast_1d(arguments[0](eta, scale, values))
                    if category_count == 2:
                        return base
                    return jnp.concatenate(
                        (base, base + jnp.cumsum(arguments[1](eta, scale, values)))
                    )

                return cutpoints
            if name == "category_logits":

                def logits(
                    eta: jax.Array, scale: jax.Array, values: tuple[jax.Array, ...]
                ) -> jax.Array:
                    predictor, intercepts, slopes = (
                        argument(eta, scale, values) for argument in arguments
                    )
                    return jnp.concatenate(
                        (jnp.zeros((1,), dtype=predictor.dtype), intercepts + slopes * predictor)
                    )

                return logits
            return lambda eta, scale, values: apply_expression_function(
                name, tuple(argument(eta, scale, values) for argument in arguments)
            )

        def binary(
            name: BinaryOperator, left: OperandEvaluator, right: OperandEvaluator
        ) -> OperandEvaluator:
            return lambda eta, scale, values: jnp.asarray(
                SCALAR_OPERATIONS[name](left(eta, scale, values), right(eta, scale, values))
            )

        evaluate: OperandEvaluator = fold_expression(
            expression,
            literal=lambda value: lambda _eta, _scale, _values: jnp.asarray(value),
            state_value=lambda _identity: lambda eta, _scale, _values: eta,
            coefficient_value=coefficient,
            binary=binary,
            call=call,
            substitution=(terms.predictor, lambda eta, _scale, _values: eta),
        )
        if terms.link == LinkFunction.INVERSE:
            evaluate = _positive_domain(evaluate)
        event_size = (
            category_count if law.distribution == "Categorical" and expression == law.logits else 0
        )
        if law.distribution == "OrderedLogistic" and expression == law.cutpoints:
            event_size = category_count - 1
        return BoundExpression(
            expression,
            evaluate,
            tuple(coordinates),
            (),
            event_size,
            sampling_count,
            response,
            terms.link,
        )

    return map_law(law, bind)


def materialize_observation_laws(
    model: CompiledModel, samples: Mapping[str, jax.Array]
) -> tuple[Law[BoundExpression], ...]:
    """Gather each law's dynamic leaves once from canonical parameter-site draws."""

    def bind(operand: BoundExpression) -> BoundExpression:
        return operand.bind(samples)

    return tuple(map_law(observation.law, bind) for observation in model.observations)
