"""Derive execution selections and projection findings from canonical model entities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.artifacts.construct import (
    CausalEdgeSpec,
    TemporalStatus,
)
from nof1_causal_lab.compilation_errors import AggregatedCompileError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.artifacts.identification import IdentificationReport
    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.question import QuestionSpec

type DependencyKey = tuple[
    ConstructId, ConstructId, Literal["innovation_correlation", "initial_state_correlation"]
]


class StructuralCompilationError(AggregatedCompileError):
    """Unsupported scientific structure encountered when preparing execution."""

    header = "Structural compilation failed"


class StructuralSelectionError(ValueError):
    """The model breaks a check its outcome scope decides: parameter anchors."""


@dataclass(frozen=True, eq=False)
class StructuralSelection:
    """One model's executable structure, scoped by the question's outcome.

    The outcome keeps its connected component as states; without one, every
    measured construct is a state. The rest of the DAG stays for identification.
    Construction checks what the scope decides: the retained constructs' anchors
    only.
    """

    model: ModelSpec
    outcome: ConstructId | None

    def __post_init__(self) -> None:
        """Require a model-owned outcome and valid parameter anchors for the structural selection."""
        from nof1_causal_lab.models.model_checks import validate_parameter_anchors

        if self.outcome is not None and self.outcome not in self.model._constructs:
            raise ValueError("A selection's outcome must be a construct of its model")
        validate_parameter_anchors(self)

    @classmethod
    def for_question(cls, model: ModelSpec, question: QuestionSpec) -> StructuralSelection:
        """The question's outcome scopes the model once the model defines it."""
        defined = question.outcome in model._constructs
        return cls(model, question.outcome if defined else None)

    @cached_property
    def identification(self) -> IdentificationReport:
        """Causal identification findings for this model and selected outcome."""
        from nof1_causal_lab.models.identification import identify_model

        return identify_model(self)

    @cached_property
    def marginalized_construct_ids(self) -> frozenset[ConstructId]:
        """Construct identities removed from the execution state by marginalization."""
        return marginalized_construct_ids(self)

    @cached_property
    def retained_construct_ids(self) -> frozenset[ConstructId]:
        """Construct identities retained in the execution representation."""
        return retained_construct_ids(self)

    @cached_property
    def induced_dependencies(self) -> Mapping[DependencyKey, tuple[ConstructId, ...]]:
        """Dependencies induced by marginalization, paired with the constructs that induce them."""
        return induced_dependencies(self)



def marginalized_construct_ids(selection: StructuralSelection) -> frozenset[ConstructId]:
    """Identify eligible unmeasured roots without changing the scientific DAG."""
    model = selection.model
    observed = {
        construct.id
        for construct in model.constructs
        if construct.indicators or construct.role == "exogenous"
    }
    blocked = {
        confounder
        for treatment, finding in selection.identification.non_identifiable.items()
        if treatment in observed
        for confounder in finding.confounders
    }
    children = {edge.effect.id for edge in model.edges}
    return frozenset(
        construct.id
        for construct in model.constructs
        if construct.id not in observed | blocked and construct.id not in children
    )


def selected_state_ids(selection: StructuralSelection) -> tuple[ConstructId, ...]:
    """A structural selection is meaningful even while numerical choices are drafts."""
    selected = selection.retained_construct_ids
    return tuple(
        item.id
        for static in (False, True)
        for item in selection.model.constructs
        if item.id in selected and (item.temporal_status == TemporalStatus.TIME_INVARIANT) == static
    )


def selected_edges(selection: StructuralSelection) -> tuple[CausalEdgeSpec, ...]:
    """Select authored edges whose cause and effect both belong to the retained state set."""
    states = set(selected_state_ids(selection))
    return tuple(
        edge
        for edge in selection.model.edges
        if edge.cause.id in states and edge.effect.id in states
    )


def selected_indicators(selection: StructuralSelection) -> tuple[IndicatorSpec, ...]:
    """Select indicators owned by retained states, preserving authored model order."""
    states = set(selected_state_ids(selection))
    return tuple(
        indicator for owner, indicator in selection.model.iter_indicators() if owner.id in states
    )


def reference_indicators(selection: StructuralSelection) -> Mapping[ConstructId, IndicatorId]:
    """Choose the measurement anchor for each selected state and return its observation ID."""
    from types import MappingProxyType

    from nof1_causal_lab.utils.causal_design import choose_reference_indicator

    return MappingProxyType(
        {
            identity: choose_reference_indicator(
                selection.model.get_construct(identity).indicators
            ).observation.id
            for identity in selected_state_ids(selection)
            if selection.model.get_construct(identity).indicators
        }
    )


def induced_dependencies(
    selection: StructuralSelection,
) -> dict[DependencyKey, tuple[ConstructId, ...]]:
    """Pairs of retained states sharing projected roots, with their scientific sources."""
    from nof1_causal_lab.utils.identifiability import dag_to_admg, get_observed_constructs

    model = selection.model
    observed = get_observed_constructs(model.constructs)
    _, confounders = dag_to_admg(model.constructs, model.edges, observed)
    retained = set(selected_state_ids(selection))
    sources: dict[DependencyKey, list[ConstructId]] = defaultdict(list)
    for identity in sorted(
        selection.marginalized_construct_ids, key=lambda cid: model.get_construct(cid).name
    ):
        construct = model.get_construct(identity)
        if construct.name not in confounders:
            continue
        children = sorted(
            {
                edge.effect.id
                for edge in model.edges
                if edge.cause.id == identity and edge.effect.id in retained
            },
            key=lambda cid: model.get_construct(cid).name,
        )
        kind: Literal["innovation_correlation", "initial_state_correlation"] = (
            "initial_state_correlation"
            if construct.temporal_status == TemporalStatus.TIME_INVARIANT
            else "innovation_correlation"
        )
        for first, second in combinations(children, 2):
            sources[first, second, kind].append(identity)
    return {
        key: tuple(sorted(identities, key=lambda cid: model.get_construct(cid).name))
        for key, identities in sorted(
            sources.items(),
            key=lambda item: (
                model.get_construct(item[0][0]).name,
                model.get_construct(item[0][1]).name,
                item[0][2],
            ),
        )
    }


def retained_construct_ids(selection: StructuralSelection) -> frozenset[ConstructId]:
    """Keep the outcome's component after projection, including statistical dependencies."""
    import networkx as nx

    model = selection.model
    measured = {
        construct.id
        for construct in model.constructs
        if construct.indicators or construct.role == "exogenous"
    }
    if selection.outcome is None:
        # A model-wide operation has no outcome against which to discard a component.
        return frozenset(measured)

    graph = nx.Graph()
    graph.add_nodes_from((*measured, selection.outcome))
    marginalized = selection.marginalized_construct_ids
    graph.add_edges_from(
        (edge.cause.id, edge.effect.id)
        for edge in model.edges
        if edge.effect.id in measured
        and (
            edge.cause.id in marginalized
            or (
                edge.cause.id in measured
                and edge.effect.temporal_status != TemporalStatus.TIME_INVARIANT
            )
        )
    )
    for construct in model.constructs:
        if construct.id not in measured:
            continue
        dependencies = {
            identity for operand in construct.coefficients for identity in operand.construct_ids
        }
        for indicator in construct.indicators:
            if indicator.likelihood is not None:
                dependencies.update(indicator.likelihood.parsed.loadings)
        graph.add_edges_from((construct.id, identity) for identity in dependencies & measured)

    # Shared parameters and joint laws can connect components without a direct causal edge.
    law_members: dict[str, set[ConstructId]] = defaultdict(set)
    for construct in model.constructs:
        if construct.distribution is not None and construct.id in measured:
            law_members[construct.distribution].add(construct.id)
    for parameter in model.parameters:
        owners = {
            owner.id
            for owner in model.parameter_context(parameter.id).owners
            if owner.kind == "construct" and owner.id in measured
        }
        nx.add_path(graph, sorted(owners))
        if parameter.distribution is not None:
            law_members[parameter.distribution].update(owners)
    for members in law_members.values():
        nx.add_path(graph, sorted(members))
    component = nx.node_connected_component(graph, selection.outcome)
    return frozenset(identity for identity in measured if identity in component)


def unsupported_construct_ids(selection: StructuralSelection) -> frozenset[ConstructId]:
    """Required parents outside the selection without an executable marginalization rule."""
    states = set(selected_state_ids(selection))
    selected = selection.retained_construct_ids
    return frozenset(
        {
            edge.cause.id
            for edge in selection.model.edges
            if edge.effect.id in states
            and edge.cause.id not in states
            and (not edge.cause.indicators or edge.cause.id not in selected)
            and edge.cause.id not in selection.marginalized_construct_ids
        }
    )


def validate_execution_structure(selection: StructuralSelection) -> None:
    """Check executable capabilities after the model's intrinsic/reference validation."""
    selection.model.require_measurements()
    states = set(selected_state_ids(selection))
    errors = []
    if selection.outcome is not None and selection.outcome not in states:
        errors.append("The question's outcome requires retained measurement indicators.")
    for edge in selected_edges(selection):
        if edge.effect.temporal_status == TemporalStatus.TIME_INVARIANT:
            errors.append(
                f"Unsupported retained static-target edge {edge.id} "
                f"({edge.cause.name!r} -> {edge.effect.name!r}). "
                "The executable SSM has no baseline structural-equation semantics. "
                "Supply supported baseline structural semantics before executing this relation."
            )
    unsupported = unsupported_construct_ids(selection)
    if unsupported:
        errors.append(
            "Required constructs outside the retained states cannot be projected: "
            f"{sorted(unsupported)}. Supply measurements or supported marginalization semantics."
        )
    if errors:
        raise StructuralCompilationError(errors)

