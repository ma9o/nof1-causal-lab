"""Bind scientific expression references to numerical state and parameter coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient, ParameterCoefficient
from nof1_causal_lab.artifacts.construct import CausalEdge, KnownInput
from nof1_causal_lab.artifacts.expressions import (
    CoefficientExpression,
    expression_states,
    linear_coefficient,
    map_expression,
)
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponentSpec

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Sequence

    from nof1_causal_lab.artifacts.construct import Construct
    from nof1_causal_lab.artifacts.mechanism import DynamicsMechanism
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def _is_projected_loading(
    owner: Construct | CausalEdge,
    mechanism: DynamicsMechanism,
    retained_states: Collection[str],
) -> bool:
    if (
        not isinstance(owner, CausalEdge)
        or owner.cause.id in retained_states
        or isinstance(owner.cause.usage, KnownInput)
    ):
        return False
    weight = linear_coefficient(mechanism.expression, owner.cause.id)
    if not isinstance(weight, FixedCoefficient):
        raise ValueError("Marginalized confounders support fixed linear loadings")
    return True


def lower_mechanisms(model: ModelSpec) -> tuple[ExpressionComponentSpec, ...]:
    """Require executable coverage, then bind every scalar expression without kind dispatch."""
    states = set(model.state_order)
    retained_edges = {edge.id for edge in model.execution_edges}
    modeled_edges: set[str] = set()
    modeled_nodes: set[str] = set()
    for owner, mechanism in model.iter_mechanisms():
        if _is_projected_loading(owner, mechanism, states):
            continue
        if isinstance(owner, CausalEdge):
            if owner.id not in retained_edges:
                raise ValueError(f"Mechanism references unknown retained edge {owner.id!r}")
            dependencies = expression_states(mechanism.expression)
            modeled_edges.update(
                edge.id
                for edge in model.edges
                if edge.effect.id == owner.effect.id and edge.cause.id in dependencies
            )
        else:
            if owner.id not in states:
                raise ValueError(f"Mechanism references unknown retained state {owner.id!r}")
            if owner.temporal_status == "time_invariant":
                raise ValueError("Time-invariant states cannot have drift mechanisms")
            modeled_nodes.add(owner.id)
    expected_nodes = {
        key for key in states if model.get_construct(key).temporal_status != "time_invariant"
    }
    if modeled_nodes != expected_nodes or modeled_edges != retained_edges:
        raise IncompleteModelError(
            "Mechanisms must cover the retained dynamic states and edges: "
            f"missing states={sorted(expected_nodes - modeled_nodes)}, "
            f"missing edges={sorted(retained_edges - modeled_edges)}"
        )
    return tuple(component for _, component in iter_mechanism_components(model, model.state_order))


def iter_mechanism_components(
    model: ModelSpec, state_order: Sequence[str]
) -> Iterator[tuple[DynamicsMechanism, ExpressionComponentSpec]]:
    """Emit a bound expression alongside the exact scientific term that produced it."""
    state_index = {key: index for index, key in enumerate(state_order)}
    input_edges: set[str] = set()

    def resolve_constant(node):
        if isinstance(node, CoefficientExpression) and isinstance(
            node.coefficient, ParameterCoefficient
        ):
            value = model.parameter(node.coefficient.parameter_id).value
            if value is not None:
                return CoefficientExpression(
                    role=node.role, coefficient=FixedCoefficient(value=value)
                )
        return node

    for owner, mechanism in model.iter_mechanisms():
        if _is_projected_loading(owner, mechanism, state_index):
            continue
        if isinstance(owner, CausalEdge):
            if isinstance(owner.cause.usage, KnownInput):
                linear_coefficient(mechanism.expression, owner.cause.id)
                if owner.id in input_edges:
                    raise ValueError(
                        "Known-input edges support one linear expression per input matrix cell"
                    )
                input_edges.add(owner.id)
                continue
            target = state_index[owner.effect.id]
        else:
            target = state_index[owner.id]
        yield (
            mechanism,
            ExpressionComponentSpec(
                expression=map_expression(mechanism.expression, resolve_constant),
                target=target,
                state_ids=tuple(state_order),
                source=state_index[owner.cause.id] if isinstance(owner, CausalEdge) else None,
                kind=mechanism.kind,
            ),
        )
