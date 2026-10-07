"""Read parameter meaning from component slots; no parallel inventory is stored."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.construct import CausalEdgeSpec
from nof1_causal_lab.artifacts.expressions import (
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
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.models.model_structure import selected_indicators, selected_state_ids

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nof1_causal_lab.artifacts.construct import ConstructSpec
    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec, _ModelEntities
    from nof1_causal_lab.artifacts.identity import EntityRef, ParameterId
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.models.model_structure import StructuralSelection


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
        """Shared coefficient quantity, rejecting uses with incompatible scientific meanings."""
        kinds = {use.quantity for use in self.uses}
        if len(kinds) != 1:
            raise ValueError("A shared parameter must have compatible coefficient meanings")
        return self.uses[0].quantity

    @property
    def owners(self) -> tuple[EntityRef, ...]:
        """Distinct scientific owners of all coefficient uses, preserving first occurrence."""
        return tuple({owner.id: owner for use in self.uses for owner in use.owners}.values())


def iter_coefficient_uses(
    dynamical_model_spec: DynamicalModelSpec | _ModelEntities,
) -> Iterator[CoefficientUse]:
    """Yield model coefficient uses with their quantity meanings, owners, and authored locations."""
    for owner, mechanism in dynamical_model_spec.iter_mechanisms():
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
    for construct in dynamical_model_spec.constructs:
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
            terms = indicator.likelihood.parsed
            for identity, operand in terms.loadings.items():
                if operand.value is not None:
                    yield CoefficientUse(
                        operand.meaning.quantity,
                        (ConstructRef(id=identity), IndicatorRef(id=indicator.observation.id)),
                        operand.value,
                        f"{indicator.observation.id}.likelihood.loading.{identity}",
                    )
            observation_refs = (ref, IndicatorRef(id=indicator.observation.id))
            for operand in (terms.intercept, *terms.auxiliary):
                if operand.value is not None:
                    yield CoefficientUse(
                        operand.meaning.quantity,
                        observation_refs,
                        operand.value,
                        f"{indicator.observation.id}.likelihood.{operand.role}",
                    )


def parameter_contexts(
    dynamical_model_spec: DynamicalModelSpec | _ModelEntities,
) -> dict[ParameterId, ParameterContext]:
    """Group symbolic coefficient uses by parameter identity to establish each parameter's context."""
    grouped: dict[ParameterId, list[CoefficientUse]] = {}
    for use in iter_coefficient_uses(dynamical_model_spec):
        if isinstance(use.value, str):
            grouped.setdefault(use.value, []).append(use)
    return {identity: ParameterContext(tuple(uses)) for identity, uses in grouped.items()}


def execution_coefficient_uses(selection: StructuralSelection) -> Iterator[CoefficientUse]:
    """Exclude coefficients owned only by structure outside the numerical selection."""
    dynamical_model_spec = selection.dynamical_model_spec
    states = set(selected_state_ids(selection))
    roots = {
        edge.cause.id
        for edge in dynamical_model_spec.edges
        if edge.cause.id in selection.marginalized_construct_ids and edge.effect.id in states
    }
    edges = {
        edge.id
        for edge in dynamical_model_spec.edges
        if edge.cause.id in states | roots and edge.effect.id in states
    }
    active = {
        "construct": states | roots,
        "edge": edges,
        "indicator": {indicator.observation.id for indicator in selected_indicators(selection)},
    }
    for use in iter_coefficient_uses(dynamical_model_spec):
        if all(owner.kind == "mechanism" or owner.id in active[owner.kind] for owner in use.owners):
            yield use


def execution_parameters(selection: StructuralSelection) -> tuple[ParameterSpec, ...]:
    """Select the scientific coefficients used by the current retained structure."""
    referenced = {use.value for use in execution_coefficient_uses(selection)}
    return tuple(
        parameter
        for parameter in selection.dynamical_model_spec.parameters
        if parameter.id in referenced
    )


def require_priors(selection: StructuralSelection) -> None:
    """Only the coefficients the selection executes need prior laws."""
    missing = [
        parameter.id
        for parameter in execution_parameters(selection)
        if parameter.distribution is None
    ]
    if missing:
        raise IncompleteModelError(f"Compilation requires declared prior laws for {missing}")


def coefficient_value(coefficient: float | ParameterId) -> float | None:
    """Resolve a literal; named parameters always represent uncertain quantities."""
    return None if isinstance(coefficient, str) else coefficient


def baseline_factor_groups(
    selection: StructuralSelection,
) -> tuple[tuple[ConstructSpec, ...], ...]:
    """A shared scale denotes one identifiable factor for marginalized baseline roots."""
    dynamical_model_spec = selection.dynamical_model_spec
    grouped: dict[str, list[ConstructSpec]] = {}
    states = set(selected_state_ids(selection))
    retained_parents = {
        edge.cause.id for edge in dynamical_model_spec.edges if edge.effect.id in states
    }
    for construct in dynamical_model_spec.constructs:
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
