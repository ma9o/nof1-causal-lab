"""Read parameter meaning from component slots; no parallel inventory is stored."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
from nof1_causal_lab.artifacts.expressions import (
    CoefficientExpression,
    expression_coefficients,
    expression_states,
)
from nof1_causal_lab.artifacts.identity import (
    ConstructRef,
    EdgeRef,
    IndicatorRef,
    MechanismRef,
)
from nof1_causal_lab.artifacts.indicator import IndicatorSpec
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
from nof1_causal_lab.artifacts.parameter import SiteKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nof1_causal_lab.artifacts.identity import EntityRef, ParameterId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


@dataclass(frozen=True)
class CoefficientUse:
    """One derived use of a literal or parameter within a scientific component."""

    quantity: SiteKind
    owners: tuple[EntityRef, ...]
    value: float | ParameterId
    slot: str


@dataclass(frozen=True)
class ParameterContext:
    """A derived reverse index of the slots referencing one parameter."""

    uses: tuple[CoefficientUse, ...]

    @property
    def quantity(self) -> SiteKind:
        kinds = {use.quantity for use in self.uses}
        if len(kinds) != 1:
            raise ValueError("A shared parameter must have compatible coefficient meanings")
        return self.uses[0].quantity

    @property
    def owners(self) -> tuple[EntityRef, ...]:
        return tuple({owner.id: owner for use in self.uses for owner in use.owners}.values())


def iter_coefficient_uses(model: ModelSpec) -> Iterator[CoefficientUse]:
    for owner, mechanism in model.iter_mechanisms():
        refs: list[EntityRef] = [MechanismRef(id=mechanism.id)]
        target = owner.effect.id if isinstance(owner, CausalEdgeSpec) else owner.id
        refs.append(ConstructRef(id=target))
        if isinstance(owner, CausalEdgeSpec):
            refs.append(EdgeRef(id=owner.id))
        refs.extend(
            ConstructRef(id=identity)
            for identity in sorted(expression_states(mechanism.expression) - {target})
        )
        for index, operand in enumerate(expression_coefficients(mechanism.expression)):
            if operand.value is None:
                continue
            quantity = operand.meaning.quantity
            yield CoefficientUse(
                quantity, tuple(refs), operand.value, f"{mechanism.id}.expression.{index}"
            )
    for construct in model.constructs:
        ref = ConstructRef(id=construct.id)
        for operand in construct.coefficients:
            if operand.value is None:
                continue
            quantity = operand.meaning.quantity
            if (
                quantity == SiteKind.T0_VAR_DIAG
                and not construct.indicators
                and construct.temporal_status == "time_invariant"
            ):
                quantity = SiteKind.STATIC_STATE_SD
            yield CoefficientUse(
                quantity,
                (ref, *(ConstructRef(id=identity) for identity in operand.construct_ids)),
                operand.value,
                ".".join((construct.id, "coefficients", operand.role, *operand.construct_ids)),
            )
        for indicator in construct.indicators:
            if indicator.likelihood is None:
                continue
            terms = indicator.likelihood.terms
            for identity, operand in terms.loadings.items():
                if operand.value is not None:
                    yield CoefficientUse(
                        operand.meaning.quantity,
                        (ConstructRef(id=identity), IndicatorRef(id=indicator.id)),
                        operand.value,
                        f"{indicator.id}.likelihood.loading.{identity}",
                    )
            observation_refs = (ref, IndicatorRef(id=indicator.id))
            for operand in (terms.intercept, *terms.auxiliary):
                if operand.value is not None:
                    yield CoefficientUse(
                        operand.meaning.quantity,
                        observation_refs,
                        operand.value,
                        f"{indicator.id}.likelihood.{operand.role}",
                    )


def parameter_contexts(model: ModelSpec) -> dict[ParameterId, ParameterContext]:
    grouped: dict[ParameterId, list[CoefficientUse]] = {}
    for use in iter_coefficient_uses(model):
        if isinstance(use.value, str):
            grouped.setdefault(use.value, []).append(use)
    return {identity: ParameterContext(tuple(uses)) for identity, uses in grouped.items()}


def execution_coefficient_uses(model: ModelSpec) -> Iterator[CoefficientUse]:
    """Exclude coefficients owned only by structure outside the numerical selection."""
    states = set(model.state_order)
    roots = {
        edge.cause.id
        for edge in model.edges
        if edge.cause.id in model.marginalized_construct_ids and edge.effect.id in states
    }
    edges = {
        edge.id
        for edge in model.edges
        if edge.cause.id in states | roots and edge.effect.id in states
    }
    active = {
        "construct": states | roots,
        "edge": edges,
        "indicator": set(model.manifest_indicator_order),
    }
    for use in iter_coefficient_uses(model):
        if all(owner.kind == "mechanism" or owner.id in active[owner.kind] for owner in use.owners):
            yield use


type CoefficientOwner = (
    ConstructSpec | CausalEdgeSpec | IndicatorSpec | DynamicsMechanismSpec | CoefficientExpression
)


def _owned_coefficients(component: CoefficientOwner) -> Iterator[CoefficientExpression]:
    if isinstance(component, CoefficientExpression):
        yield component
    elif isinstance(component, DynamicsMechanismSpec):
        yield from expression_coefficients(component.expression)
    elif isinstance(component, IndicatorSpec):
        if component.likelihood is not None:
            for argument in component.likelihood.law.arguments.values():
                yield from expression_coefficients(argument)
    else:
        mechanisms = (
            component.dynamics if isinstance(component, ConstructSpec) else component.mechanisms
        )
        for mechanism in mechanisms:
            yield from expression_coefficients(mechanism.expression)
        if isinstance(component, ConstructSpec):
            yield from component.coefficients
            for indicator in component.indicators:
                yield from _owned_coefficients(indicator)


def referenced_parameter_ids(*components: CoefficientOwner) -> frozenset[ParameterId]:
    """Follow coefficient references through owned components, without an execution plan."""
    return frozenset(
        operand.value
        for component in components
        for operand in _owned_coefficients(component)
        if isinstance(operand.value, str)
    )


def coefficient_value(coefficient: float | ParameterId) -> float | None:
    """Resolve a literal; named parameters always represent uncertain quantities."""
    return None if isinstance(coefficient, str) else coefficient


def baseline_factor_groups(model: ModelSpec):
    """A shared scale denotes one identifiable factor for marginalized baseline roots."""
    grouped = {}
    retained_parents = {
        edge.cause.id for edge in model.edges if edge.effect.id in model.state_order
    }
    for construct in model.constructs:
        if (
            construct.indicators
            or construct.id not in retained_parents
            or construct.temporal_status != "time_invariant"
            or construct.coefficient("initial_scale") is None
        ):
            continue
        coefficient = construct.coefficient("initial_scale")
        key = coefficient if isinstance(coefficient, str) else construct.id
        grouped.setdefault(key, []).append(construct)
    return tuple(tuple(group) for group in grouped.values())
