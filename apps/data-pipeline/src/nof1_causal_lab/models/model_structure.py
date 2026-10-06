"""Derive execution selections and projection findings from canonical model entities."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.artifacts.construct import (
    CausalEdgeSpec,
    ConstructSpec,
    TemporalStatus,
)
from nof1_causal_lab.artifacts.execution import StructuralDisposition, StructuralItemDisposition
from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, IndicatorRef
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.study.view_models import (
    Added,
    Change,
    Removed,
    Revised,
    Unchanged,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from nof1_causal_lab.artifacts.identification import IdentificationReport
    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
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

    @cached_property
    def structural_dispositions(self) -> tuple[StructuralItemDisposition, ...]:
        """Execution treatment of the selection's constructs, edges, and observation indicators."""
        return structural_dispositions(self)


def marginalized_construct_ids(selection: StructuralSelection) -> frozenset[ConstructId]:
    """Identify eligible unmeasured roots without changing the scientific DAG."""
    model = selection.model
    observed = {construct.id for construct in model.constructs if construct.indicators}
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
    measured = {construct.id for construct in model.constructs if construct.indicators}
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


def structural_dispositions(
    selection: StructuralSelection,
) -> tuple[StructuralItemDisposition, ...]:
    """Explain the computational treatment of every scientific entity at this revision."""
    model = selection.model
    model.require_measurements()
    unsupported = unsupported_construct_ids(selection)
    states = set(selected_state_ids(selection))
    edge_ids = {edge.id for edge in selected_edges(selection)}
    manifests = {indicator.observation.id for indicator in selected_indicators(selection)}
    findings = []
    for construct in model.constructs:
        if construct.id in states:
            disposition = StructuralDisposition.RETAINED_STATE
            reason = "Measured construct selected as a state; execution requirements are checked separately."
        elif construct.id in unsupported:
            disposition = StructuralDisposition.UNSUPPORTED
            reason = "Required cause outside the retained states has no supported marginalization semantics."
        elif construct.id in selection.marginalized_construct_ids:
            disposition = StructuralDisposition.MARGINALIZED
            reason = "Safe unmeasured root projected from the executable state vector."
        else:
            disposition = StructuralDisposition.IDENTIFICATION_ONLY
            reason = (
                f"Disconnected from outcome '{model.get_construct(selection.outcome).name}' "
                "after structural projection."
                if construct.indicators and selection.outcome is not None
                else "Scientific-DAG construct not retained in the executable state."
            )
        findings.append(
            StructuralItemDisposition(
                target=ConstructRef(id=construct.id),
                disposition=disposition,
                reason=reason,
            )
        )
    for edge in model.edges:
        retained = edge.id in edge_ids
        unsupported_edge = edge.cause.id in unsupported or (
            retained and edge.effect.temporal_status == TemporalStatus.TIME_INVARIANT
        )
        findings.append(
            StructuralItemDisposition(
                target=EdgeRef(id=edge.id),
                disposition=StructuralDisposition.UNSUPPORTED
                if unsupported_edge
                else StructuralDisposition.RETAINED_EDGE
                if retained
                else StructuralDisposition.PROJECTED_EDGE,
                reason="Required relation lacks supported state or baseline semantics."
                if unsupported_edge
                else "Both endpoints survive the executable projection."
                if retained
                else "At least one endpoint is not an executable retained state.",
            )
        )
    for indicator in model.indicators:
        if indicator.observation.id in manifests:
            disposition = StructuralDisposition.MANIFEST
            reason = "Indicator retained as a manifest likelihood channel."
        else:
            disposition = StructuralDisposition.EXCLUDED_INDICATOR
            reason = "Indicator belongs to a construct outside the executable state vector."
        findings.append(
            StructuralItemDisposition(
                target=IndicatorRef(id=indicator.observation.id),
                disposition=disposition,
                reason=reason,
            )
        )
    return tuple(findings)


def model_graph_entities(
    selection: StructuralSelection,
) -> tuple[tuple[ConstructSpec, ...], tuple[CausalEdgeSpec, ...]]:
    """Show authored structure until execution dispositions establish the retained graph."""
    model = selection.model
    if model.measurement_clock is None or not model.indicators:
        return model.constructs, model.edges
    dispositions = {item.target.id: item.disposition for item in selection.structural_dispositions}
    return (
        tuple(
            item
            for item in model.constructs
            if dispositions[item.id] == StructuralDisposition.RETAINED_STATE
        ),
        tuple(
            item
            for item in model.edges
            if dispositions[item.id] == StructuralDisposition.RETAINED_EDGE
        ),
    )


def compare_parameters(
    left: ModelSpec | None, right: ModelSpec | None
) -> tuple[Change[ParameterSpec], ...]:
    """Compare native parameter decisions and law contents by persistent identity."""
    old = {p.id: p for p in left.parameters} if left is not None else {}
    new = {p.id: p for p in right.parameters} if right is not None else {}
    old_laws = left.model_dump(mode="json")["distributions"] if left is not None else {}
    new_laws = right.model_dump(mode="json")["distributions"] if right is not None else {}
    changes: list[Change[ParameterSpec]] = []
    for identity in sorted(old.keys() | new.keys()):
        a, b = old.get(identity), new.get(identity)
        if a == b and (
            a is None
            or a.distribution is None
            or json.dumps(old_laws[a.distribution], sort_keys=True)
            == json.dumps(new_laws[a.distribution], sort_keys=True)
        ):
            continue
        change = (
            Added(after=new[identity])
            if identity not in old
            else Removed(before=old[identity])
            if identity not in new
            else Revised(before=old[identity], after=new[identity])
        )
        changes.append(change)
    return tuple(changes)


def compare_model_graph(
    left: StructuralSelection | None, right: StructuralSelection | None
) -> tuple[
    tuple[Change[ConstructRef] | Unchanged[ConstructRef], ...],
    tuple[Change[EdgeRef] | Unchanged[EdgeRef], ...],
]:
    """Compare entity presence and time-slice topology against the pinned models."""
    graphs = tuple(
        model_graph_entities(selection) if selection is not None else ((), ())
        for selection in (left, right)
    )

    def topology(
        entity: ConstructSpec | CausalEdgeSpec,
    ) -> bool | tuple[ConstructId, bool, ConstructId]:
        if isinstance(entity, CausalEdgeSpec):
            return entity.cause.id, entity.cause.is_dynamic, entity.effect.id
        return entity.is_dynamic

    def entities[T: (ConstructSpec, CausalEdgeSpec), RefT: (ConstructRef, EdgeRef)](
        before: Sequence[T], after: Sequence[T], project: Callable[[T], RefT]
    ) -> Iterator[Change[RefT] | Unchanged[RefT]]:
        old, new = {item.id: item for item in before}, {item.id: item for item in after}
        for identity in sorted(old.keys() | new.keys()):
            if identity not in old:
                yield Added(after=project(new[identity]))
            elif identity not in new:
                yield Removed(before=project(old[identity]))
            elif topology(old[identity]) == topology(new[identity]):
                yield Unchanged(before=project(old[identity]), after=project(new[identity]))
            else:
                yield Revised(before=project(old[identity]), after=project(new[identity]))

    return (
        tuple(entities(graphs[0][0], graphs[1][0], lambda item: ConstructRef(id=item.id))),
        tuple(entities(graphs[0][1], graphs[1][1], lambda item: EdgeRef(id=item.id))),
    )
