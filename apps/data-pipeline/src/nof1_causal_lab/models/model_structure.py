"""Derive execution selections and projection findings from canonical model entities."""

from __future__ import annotations

import json
from collections import defaultdict
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
    ComparisonConnection,
    ConstructComparison,
    EdgeComparison,
    ModelDefinitionChange,
    ModelGraphComparison,
    ParameterChange,
    Removed,
    Revised,
    Unchanged,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.json_types import JsonValue

type DependencyKey = tuple[
    ConstructId, ConstructId, Literal["innovation_correlation", "initial_state_correlation"]
]


class StructuralCompilationError(AggregatedCompileError):
    """Unsupported scientific structure encountered when preparing execution."""

    header = "Structural compilation failed"


def marginalized_construct_ids(model: ModelSpec) -> frozenset[ConstructId]:
    """Identify eligible unmeasured roots without changing the scientific DAG."""
    from nof1_causal_lab.models.identification import identify_model

    observed = {construct.id for construct in model.constructs if construct.indicators}
    blocked = {
        confounder
        for treatment, finding in identify_model(model).non_identifiable.items()
        if treatment in observed
        for confounder in finding.confounders
    }
    children = {edge.effect.id for edge in model.edges}
    return frozenset(
        construct.id
        for construct in model.constructs
        if construct.id not in observed | blocked and construct.id not in children
    )


def selected_state_ids(model: ModelSpec) -> tuple[ConstructId, ...]:
    """A structural selection is meaningful even while numerical choices are drafts."""
    selected = retained_construct_ids(model)
    return tuple(
        item.id
        for static in (False, True)
        for item in model.constructs
        if item.id in selected and (item.temporal_status == TemporalStatus.TIME_INVARIANT) == static
    )


def selected_edges(model: ModelSpec) -> tuple[CausalEdgeSpec, ...]:
    states = set(selected_state_ids(model))
    return tuple(
        edge for edge in model.edges if edge.cause.id in states and edge.effect.id in states
    )


def selected_indicators(model: ModelSpec) -> tuple[IndicatorSpec, ...]:
    states = set(selected_state_ids(model))
    return tuple(indicator for owner, indicator in model.iter_indicators() if owner.id in states)


def reference_indicators(model: ModelSpec) -> Mapping[ConstructId, IndicatorId]:
    from types import MappingProxyType

    from nof1_causal_lab.utils.causal_design import choose_reference_indicator

    return MappingProxyType(
        {
            identity: choose_reference_indicator(model.get_construct(identity).indicators).id
            for identity in selected_state_ids(model)
        }
    )


def induced_dependencies(model: ModelSpec) -> dict[DependencyKey, tuple[ConstructId, ...]]:
    """Pairs of retained states sharing projected roots, with their scientific sources."""
    from nof1_causal_lab.utils.identifiability import dag_to_admg, get_observed_constructs

    observed = get_observed_constructs(model.constructs)
    _, confounders = dag_to_admg(model.constructs, model.edges, observed)
    retained = set(selected_state_ids(model))
    sources: dict[DependencyKey, list[ConstructId]] = defaultdict(list)
    for identity in sorted(
        model.marginalized_construct_ids, key=lambda cid: model.get_construct(cid).name
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


def retained_construct_ids(model: ModelSpec) -> frozenset[ConstructId]:
    """Keep the outcome's component after projection, including statistical dependencies."""
    import networkx as nx

    measured = {construct.id for construct in model.constructs if construct.indicators}
    if model.default_outcome is None:
        # A model-wide operation has no outcome against which to discard a component.
        return frozenset(measured)

    graph = nx.Graph()
    graph.add_nodes_from((*measured, model.default_outcome))
    marginalized = model.marginalized_construct_ids
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
    component = nx.node_connected_component(graph, model.default_outcome)
    return frozenset(identity for identity in measured if identity in component)


def unsupported_construct_ids(model: ModelSpec) -> frozenset[ConstructId]:
    """Required parents outside the selection without an executable marginalization rule."""
    states = set(selected_state_ids(model))
    selected = retained_construct_ids(model)
    return frozenset(
        {
            edge.cause.id
            for edge in model.edges
            if edge.effect.id in states
            and edge.cause.id not in states
            and (not edge.cause.indicators or edge.cause.id not in selected)
            and edge.cause.id not in model.marginalized_construct_ids
        }
    )


def validate_execution_structure(model: ModelSpec) -> None:
    """Check executable capabilities after the model's intrinsic/reference validation."""
    model.require_measurements()
    states = set(selected_state_ids(model))
    errors = []
    if model.default_outcome is not None and model.default_outcome not in states:
        errors.append("The default outcome requires retained measurement indicators.")
    for edge in selected_edges(model):
        if edge.effect.temporal_status == TemporalStatus.TIME_INVARIANT:
            errors.append(
                f"Unsupported retained static-target edge {edge.id} "
                f"({edge.cause.name!r} -> {edge.effect.name!r}). "
                "The executable SSM has no baseline structural-equation semantics. "
                "Supply supported baseline structural semantics before executing this relation."
            )
    unsupported = unsupported_construct_ids(model)
    if unsupported:
        errors.append(
            "Required constructs outside the retained states cannot be projected: "
            f"{sorted(unsupported)}. Supply measurements or supported marginalization semantics."
        )
    if errors:
        raise StructuralCompilationError(errors)


def structural_dispositions(model: ModelSpec) -> tuple[StructuralItemDisposition, ...]:
    """Explain the computational treatment of every scientific entity at this revision."""
    model.require_measurements()
    unsupported = unsupported_construct_ids(model)
    states = set(selected_state_ids(model))
    edge_ids = {edge.id for edge in selected_edges(model)}
    manifests = {indicator.id for indicator in selected_indicators(model)}
    findings = []
    for construct in model.constructs:
        if construct.id in states:
            disposition = StructuralDisposition.RETAINED_STATE
            reason = "Measured construct selected as a state; execution requirements are checked separately."
        elif construct.id in unsupported:
            disposition = StructuralDisposition.UNSUPPORTED
            reason = "Required cause outside the retained states has no supported marginalization semantics."
        elif construct.id in model.marginalized_construct_ids:
            disposition = StructuralDisposition.MARGINALIZED
            reason = "Safe unmeasured root projected from the executable state vector."
        else:
            disposition = StructuralDisposition.IDENTIFICATION_ONLY
            reason = (
                f"Disconnected from outcome '{model.get_construct(model.default_outcome).name}' "
                "after structural projection."
                if construct.indicators and model.default_outcome is not None
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
        if indicator.id in manifests:
            disposition = StructuralDisposition.MANIFEST
            reason = "Indicator retained as a manifest likelihood channel."
        else:
            disposition = StructuralDisposition.EXCLUDED_INDICATOR
            reason = "Indicator belongs to a construct outside the executable state vector."
        findings.append(
            StructuralItemDisposition(
                target=IndicatorRef(id=indicator.id),
                disposition=disposition,
                reason=reason,
            )
        )
    return tuple(findings)


def model_graph_entities(
    model: ModelSpec,
) -> tuple[tuple[ConstructSpec, ...], tuple[CausalEdgeSpec, ...]]:
    """Show authored structure until execution dispositions establish the retained graph."""
    if model.measurement_clock is None or not model.indicators:
        return model.constructs, model.edges
    dispositions = {item.target.id: item.disposition for item in model.structural_dispositions}
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


def compare_parameters(left: ModelSpec, right: ModelSpec) -> list[ParameterChange]:
    """Compare native parameter decisions and law contents by persistent identity."""
    old, new = {p.id: p for p in left.parameters}, {p.id: p for p in right.parameters}
    old_laws = left.model_dump(mode="json")["distributions"]
    new_laws = right.model_dump(mode="json")["distributions"]
    changes = []
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
        changes.append(ParameterChange(parameter_id=identity, change=change))
    return changes


def compare_model_definitions(left: ModelSpec, right: ModelSpec) -> list[ModelDefinitionChange]:
    """Compare every authored field, aligning entities by ID rather than list position."""

    def definition(model: ModelSpec) -> JsonValue:
        value: dict[str, JsonValue] = model.model_dump(mode="json", exclude={"edges", "parameters"})
        constructs: dict[str, JsonValue] = {}
        for item in model.constructs:
            construct: dict[str, JsonValue] = item.model_dump(mode="json", exclude={"indicators"})
            indicators: dict[str, JsonValue] = {
                indicator.id: indicator.model_dump(mode="json") for indicator in item.indicators
            }
            construct["indicators"] = indicators
            constructs[item.id] = construct
        value["constructs"] = constructs
        value["parameters"] = {item.id: item.model_dump(mode="json") for item in model.parameters}
        value["edges"] = {
            item.id: {
                **item.model_dump(mode="json", exclude={"cause", "effect"}),
                "cause": item.cause.id,
                "effect": item.effect.id,
            }
            for item in model.edges
        }
        return value

    changes = []

    def walk(before: JsonValue, after: JsonValue, path: str) -> None:
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(before.keys() | after.keys()):
                pointer = path + "/" + key.replace("~", "~0").replace("/", "~1")
                if key not in before or key not in after:
                    changes.append(
                        ModelDefinitionChange(
                            path=pointer,
                            change=Added(after=after[key])
                            if key in after
                            else Removed(before=before[key]),
                        )
                    )
                else:
                    walk(before[key], after[key], pointer)
        elif before != after:
            changes.append(
                ModelDefinitionChange(path=path, change=Revised(before=before, after=after))
            )

    walk(definition(left), definition(right), "")
    return changes


def compare_model_graph(left: ModelSpec, right: ModelSpec) -> ModelGraphComparison:
    """Compare displayed nodes and connections, including their time-slice topology."""
    from nof1_causal_lab.models.model_structure import model_graph_entities

    graphs = [model_graph_entities(model) for model in (left, right)]
    dispositions = [
        {item.target.id: item for item in model.structural_dispositions}
        if model.measurement_clock is not None and model.indicators
        else {}
        for model in (left, right)
    ]

    def topology(
        entity: ConstructSpec | CausalEdgeSpec,
    ) -> bool | tuple[ConstructId, bool, ConstructId]:
        if isinstance(entity, CausalEdgeSpec):
            return entity.cause.id, entity.cause.is_dynamic, entity.effect.id
        return entity.is_dynamic

    def entities[T: (ConstructSpec, CausalEdgeSpec), DefinitionT](
        before: Sequence[T],
        after: Sequence[T],
        project: Callable[[T], DefinitionT],
    ) -> Iterator[tuple[T, Change[DefinitionT] | Unchanged[DefinitionT]]]:
        old, new = {item.id: item for item in before}, {item.id: item for item in after}

        for identity in sorted(old.keys() | new.keys()):
            if identity not in old:
                yield new[identity], Added(after=project(new[identity]))
            elif identity not in new:
                yield old[identity], Removed(before=project(old[identity]))
            else:
                a, b = old[identity], new[identity]
                yield (
                    b,
                    (
                        Unchanged(before=project(a), after=project(b))
                        if topology(a) == topology(b)
                        else Revised(before=project(a), after=project(b))
                    ),
                )

    def connection(edge: CausalEdgeSpec) -> ComparisonConnection:
        return ComparisonConnection(
            cause=ConstructRef(id=edge.cause.id),
            effect=ConstructRef(id=edge.effect.id),
            description=edge.description,
        )

    return ModelGraphComparison(
        before_dynamic_construct_ids=tuple(item.id for item in graphs[0][0] if item.is_dynamic),
        after_dynamic_construct_ids=tuple(item.id for item in graphs[1][0] if item.is_dynamic),
        constructs=tuple(
            ConstructComparison(
                construct_id=item.id,
                change=change,
                before_disposition=dispositions[0].get(item.id),
                after_disposition=dispositions[1].get(item.id),
            )
            for item, change in entities(graphs[0][0], graphs[1][0], lambda item: item)
        ),
        edges=tuple(
            EdgeComparison(
                edge_id=item.id,
                change=change,
                before_disposition=dispositions[0].get(item.id),
                after_disposition=dispositions[1].get(item.id),
            )
            for item, change in entities(graphs[0][1], graphs[1][1], connection)
        ),
    )
