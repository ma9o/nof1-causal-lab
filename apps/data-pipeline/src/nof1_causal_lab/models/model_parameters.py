"""Read parameter meaning from component slots; no parallel inventory is stored."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from nof1_causal_lab.artifacts.coefficient import ParameterCoefficient
from nof1_causal_lab.artifacts.construct import CausalEdge, KnownInput
from nof1_causal_lab.artifacts.expressions import expression_coefficients, expression_states
from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, IndicatorRef, MechanismRef
from nof1_causal_lab.artifacts.parameter import SiteKind

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nof1_causal_lab.artifacts.coefficient import Coefficient
    from nof1_causal_lab.artifacts.identity import EntityRef, ParameterId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


@dataclass(frozen=True)
class CoefficientUse:
    """One derived use of a literal or parameter within a scientific component."""

    quantity: SiteKind
    owners: tuple[EntityRef, ...]
    coefficient: Coefficient
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
        target = owner.effect.id if isinstance(owner, CausalEdge) else owner.id
        refs.append(ConstructRef(id=target))
        if isinstance(owner, CausalEdge):
            refs.append(EdgeRef(id=owner.id))
        refs.extend(
            ConstructRef(id=identity)
            for identity in sorted(expression_states(mechanism.expression) - {target})
        )
        input_effect = isinstance(owner, CausalEdge) and isinstance(owner.cause.usage, KnownInput)
        for index, operand in enumerate(expression_coefficients(mechanism.expression)):
            if operand.coefficient is None:
                continue
            quantity = operand.meaning.quantity
            if input_effect and quantity == SiteKind.DYNAMICS_WEIGHT:
                quantity = SiteKind.INPUT_EFFECT
            yield CoefficientUse(
                quantity, tuple(refs), operand.coefficient, f"{mechanism.id}.expression.{index}"
            )
    for construct in model.constructs:
        ref = ConstructRef(id=construct.id)
        if construct.innovation is not None:
            noise = construct.innovation
            yield CoefficientUse(
                SiteKind.DIFFUSION_DIAG, (ref,), noise.scale, f"{construct.id}.innovation.scale"
            )
            if noise.degrees_of_freedom is not None:
                yield CoefficientUse(
                    SiteKind.PROC_DF,
                    (ref,),
                    noise.degrees_of_freedom,
                    f"{construct.id}.innovation.degrees_of_freedom",
                )
            for loading in noise.loadings:
                yield CoefficientUse(
                    SiteKind.DIFFUSION_LOWER,
                    (ref, ConstructRef(id=loading.other_id)),
                    loading.coefficient,
                    f"{construct.id}.innovation.loadings.{loading.other_id}",
                )
        if construct.initial_state is not None:
            initial = construct.initial_state
            baseline = not construct.indicators and construct.temporal_status == "time_invariant"
            yield CoefficientUse(
                SiteKind.T0_MEANS, (ref,), initial.mean, f"{construct.id}.initial_state.mean"
            )
            yield CoefficientUse(
                SiteKind.STATIC_STATE_SD if baseline else SiteKind.T0_VAR_DIAG,
                (ref,),
                initial.scale,
                f"{construct.id}.initial_state.scale",
            )
            for correlation in initial.correlations:
                yield CoefficientUse(
                    SiteKind.T0_VAR_LOWER,
                    (ref, ConstructRef(id=correlation.other_id)),
                    correlation.coefficient,
                    f"{construct.id}.initial_state.correlations.{correlation.other_id}",
                )
        for indicator in construct.indicators:
            if indicator.likelihood is None:
                continue
            terms = indicator.likelihood.terms
            for identity, operand in terms.loadings.items():
                if operand.coefficient is not None:
                    yield CoefficientUse(
                        operand.meaning.quantity,
                        (ConstructRef(id=identity), IndicatorRef(id=indicator.id)),
                        operand.coefficient,
                        f"{indicator.id}.likelihood.loading.{identity}",
                    )
            observation_refs = (ref, IndicatorRef(id=indicator.id))
            for operand in (terms.intercept, *terms.auxiliary):
                if operand.coefficient is not None:
                    yield CoefficientUse(
                        operand.meaning.quantity,
                        observation_refs,
                        operand.coefficient,
                        f"{indicator.id}.likelihood.{operand.role}",
                    )


def parameter_contexts(model: ModelSpec) -> dict[ParameterId, ParameterContext]:
    grouped: dict[ParameterId, list[CoefficientUse]] = {}
    for use in iter_coefficient_uses(model):
        if isinstance(use.coefficient, ParameterCoefficient):
            grouped.setdefault(use.coefficient.parameter_id, []).append(use)
    return {identity: ParameterContext(tuple(uses)) for identity, uses in grouped.items()}


def referenced_parameter_ids(*components: BaseModel) -> frozenset[ParameterId]:
    """Follow coefficient references through owned components, without an execution plan."""
    identities = set()

    def visit(value: object) -> None:
        if isinstance(value, ParameterCoefficient):
            identities.add(value.parameter_id)
        elif isinstance(value, CausalEdge):
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


def coefficient_value(model: ModelSpec, coefficient: Coefficient) -> float | None:
    if isinstance(coefficient, ParameterCoefficient):
        return model.parameter(coefficient.parameter_id).value
    return coefficient.value


def baseline_factor_groups(model: ModelSpec):
    """A shared scale denotes one identifiable factor for marginalized baseline roots."""
    grouped = {}
    for construct in model.constructs:
        if (
            construct.indicators
            or construct.temporal_status != "time_invariant"
            or construct.initial_state is None
        ):
            continue
        coefficient = construct.initial_state.scale
        key = (
            coefficient.parameter_id
            if isinstance(coefficient, ParameterCoefficient)
            else construct.id
        )
        grouped.setdefault(key, []).append(construct)
    return tuple(tuple(group) for group in grouped.values())
