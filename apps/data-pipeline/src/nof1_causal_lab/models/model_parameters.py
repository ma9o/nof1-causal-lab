"""Read parameter meaning from component slots; no parallel inventory is stored."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from nof1_causal_lab.artifacts.construct import CausalEdgeSpec
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


def referenced_parameter_ids(*components: BaseModel) -> frozenset[ParameterId]:
    """Follow coefficient references through owned components, without an execution plan."""
    identities = set()

    def visit(value: object) -> None:
        if isinstance(value, CoefficientExpression):
            if isinstance(value.value, str):
                identities.add(value.value)
        elif isinstance(value, CausalEdgeSpec):
            visit(value.mechanisms)
        elif isinstance(value, BaseModel):
            for name in type(value).model_fields:
                visit(getattr(value, name))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, tuple):
            for item in value:
                visit(item)

    for component in components:
        visit(component)
    return frozenset(identities)


def coefficient_value(model: ModelSpec, coefficient: float | ParameterId) -> float | None:
    if isinstance(coefficient, str):
        return model.parameter(coefficient).value
    return coefficient


def baseline_factor_groups(model: ModelSpec):
    """A shared scale denotes one identifiable factor for marginalized baseline roots."""
    grouped = {}
    for construct in model.constructs:
        if (
            construct.indicators
            or construct.temporal_status != "time_invariant"
            or construct.coefficient("initial_scale") is None
        ):
            continue
        coefficient = construct.coefficient("initial_scale")
        key = coefficient if isinstance(coefficient, str) else construct.id
        grouped.setdefault(key, []).append(construct)
    return tuple(tuple(group) for group in grouped.values())
