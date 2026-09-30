"""Derive execution selections and projection findings from canonical model entities."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.artifacts.construct import (
    CausalEdgeSpec,
    ConstructSpec,
    Role,
    TemporalStatus,
)
from nof1_causal_lab.artifacts.execution import StructuralDisposition, StructuralItemDisposition
from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, IndicatorRef
from nof1_causal_lab.compilation_errors import AggregatedCompileError

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec

type DependencyKey = tuple[
    ConstructId, ConstructId, Literal["innovation_correlation", "initial_state_correlation"]
]


class StructuralCompilationError(AggregatedCompileError):
    """Unsupported scientific structure encountered when preparing execution."""

    header = "Structural compilation failed"


def marginalized_construct_ids(model: ModelSpec) -> frozenset[ConstructId]:
    """Identify eligible unobserved exogenous roots without changing the scientific DAG."""
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
        if construct.id not in observed | blocked
        and construct.role == Role.EXOGENOUS
        and construct.id not in children
    )


def induced_dependencies(model: ModelSpec) -> dict[DependencyKey, tuple[ConstructId, ...]]:
    """Pairs of retained states sharing projected roots, with their scientific sources."""
    from nof1_causal_lab.utils.identifiability import dag_to_admg, get_observed_constructs

    observed = get_observed_constructs(model.constructs)
    _, confounders = dag_to_admg(model.constructs, model.edges, observed)
    retained = set(model.state_order)
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


def dependency_id(key: DependencyKey, sources: tuple[ConstructId, ...]) -> str:
    """Preserve the semantic identity of a projected dependence across axis reorderings."""
    first, second, kind = key
    identity = "\0".join((kind, *sorted((first, second)), *sorted(sources)))
    digest = sha256(f"dependency\0{identity}".encode()).hexdigest()[:20]
    return f"dependency:{digest}"


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
                dependencies.update(indicator.likelihood.terms.loadings)
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
    return frozenset(measured & nx.node_connected_component(graph, model.default_outcome))


def unsupported_construct_ids(model: ModelSpec) -> frozenset[ConstructId]:
    """Required parents outside the selection without an executable marginalization rule."""
    states = set(model.state_order)
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
    states = set(model.state_order)
    errors = []
    if model.default_outcome is not None and model.default_outcome not in states:
        errors.append("The default outcome requires retained measurement indicators.")
    for edge in model.execution_edges:
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
    states = set(model.state_order)
    edge_ids = {edge.id for edge in model.execution_edges}
    manifests = set(model.manifest_indicator_order)
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
            reason = "Safe unobserved exogenous root projected from the executable state vector."
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
