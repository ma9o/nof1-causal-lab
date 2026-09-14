"""Derive execution selections and projection findings from canonical model entities."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.artifacts.construct import (
    Role,
    ScientificOnlyConstruct,
    TemporalStatus,
    replace_constructs,
)
from nof1_causal_lab.artifacts.execution import StructuralDisposition, StructuralItemDisposition
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
    from nof1_causal_lab.models.model_inputs import identification_input
    from nof1_causal_lab.utils.identifiability import analyze_unobserved_constructs

    inputs = identification_input(model)
    analysis = analyze_unobserved_constructs(
        inputs["graph"],
        inputs["observations"],
        identify_model(model).status.model_dump(mode="json"),
    )
    candidates = analysis["can_marginalize"]
    children = {edge.effect.id for edge in model.edges}
    return frozenset(
        construct.id
        for construct in model.constructs
        if construct.name in candidates
        and construct.role == Role.EXOGENOUS
        and construct.id not in children
    )


def induced_dependencies(model: ModelSpec) -> dict[DependencyKey, tuple[ConstructId, ...]]:
    """Pairs of retained states sharing projected roots, with their scientific sources."""
    from nof1_causal_lab.models.model_inputs import graph_input, observation_input
    from nof1_causal_lab.utils.identifiability import dag_to_admg, get_observed_constructs

    observed = get_observed_constructs(
        {construct.id: construct.name for construct in model.constructs}, observation_input(model)
    )
    _, confounders = dag_to_admg(graph_input(model), observed)
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


def validate_execution_structure(model: ModelSpec) -> None:
    """Check executable capabilities after the model's intrinsic/reference validation."""
    model.require_measurements()
    states = set(model.state_order)
    inputs = set(model.known_inputs)
    errors = []
    for edge in model.execution_edges:
        if edge.effect.temporal_status == TemporalStatus.TIME_INVARIANT:
            errors.append(
                f"Unsupported retained static-target edge {edge.id} "
                f"({edge.cause.name!r} -> {edge.effect.name!r}). "
                "The executable SSM has no baseline structural-equation semantics. "
                "Convert observed baseline quantities to known inputs, reduce the static chain "
                "before compilation, or retain the relation only in the scientific DAG."
            )
    unused_inputs = inputs - {edge.cause.id for edge in model.execution_edges}
    if unused_inputs:
        errors.append(
            "Known inputs have no outgoing edge into a retained state: "
            f"{sorted(model.get_construct(identity).name for identity in unused_inputs)}. "
            "Mark scientific-context-only constructs explicitly instead of compiling unused transition inputs."
        )
    manifests = model.manifest_indicator_order
    uncovered = states - {model.indicator_owner(identity).id for identity in manifests}
    if uncovered:
        errors.append(f"Retained states have no manifest indicators: {sorted(uncovered)}")
    if len(manifests) < len(states):
        errors.append(
            "Loading matrix is rank-deficient at structural compilation: "
            f"n_manifest ({len(manifests)}) < n_latent ({len(states)})."
        )
    identities = [item.id for item in (*model.constructs, *model.edges, *model.indicators)]
    if len(identities) != len(set(identities)):
        errors.append("Construct, edge, and indicator identities must be distinct")
    if errors:
        raise StructuralCompilationError(errors)


def structural_dispositions(model: ModelSpec) -> tuple[StructuralItemDisposition, ...]:
    """Explain the computational treatment of every scientific entity at this revision."""
    validate_execution_structure(model)
    states = set(model.state_order)
    inputs = set(model.known_inputs)
    edge_ids = {edge.id for edge in model.execution_edges}
    manifests = set(model.manifest_indicator_order)
    input_indicators = {usage.source_indicator_id for usage in model.known_inputs.values()}
    findings = []
    for construct in model.constructs:
        if construct.id in states:
            disposition = StructuralDisposition.RETAINED_STATE
            reason = "Measured construct retained as an executable latent state."
        elif construct.id in inputs:
            disposition = StructuralDisposition.KNOWN_INPUT
            reason = "Observed construct compiled as a deterministic transition input."
        elif construct.id in model.marginalized_construct_ids:
            disposition = StructuralDisposition.MARGINALIZED
            reason = "Safe unobserved exogenous root projected from the executable state vector."
        elif isinstance(construct.usage, ScientificOnlyConstruct):
            disposition = StructuralDisposition.IDENTIFICATION_ONLY
            reason = (
                "Author explicitly retained this measured construct for scientific context only."
            )
        else:
            disposition = StructuralDisposition.IDENTIFICATION_ONLY
            reason = "Scientific-DAG construct not retained in the executable state."
        findings.append(
            StructuralItemDisposition(
                source_id=construct.id,
                source_kind="construct",
                disposition=disposition,
                reason=reason,
            )
        )
    for edge in model.edges:
        retained = edge.id in edge_ids
        findings.append(
            StructuralItemDisposition(
                source_id=edge.id,
                source_kind="edge",
                disposition=StructuralDisposition.RETAINED_EDGE
                if retained
                else StructuralDisposition.PROJECTED_EDGE,
                reason="Both endpoints survive the executable projection."
                if retained
                else "At least one endpoint is not an executable retained state or known input.",
            )
        )
    for indicator in model.indicators:
        if indicator.id in manifests:
            disposition = StructuralDisposition.MANIFEST
            reason = "Indicator retained as a manifest likelihood channel."
        elif indicator.id in input_indicators:
            disposition = StructuralDisposition.KNOWN_INPUT_SOURCE
            reason = "Indicator supplies a deterministic known-input trajectory."
        else:
            disposition = StructuralDisposition.EXCLUDED_INDICATOR
            reason = "Indicator belongs to a construct outside the executable state vector."
        findings.append(
            StructuralItemDisposition(
                source_id=indicator.id,
                source_kind="indicator",
                disposition=disposition,
                reason=reason,
            )
        )
    return tuple(findings)


def model_for_constructs(model: ModelSpec, keep_names: set[str]) -> ModelSpec:
    """Select the scientific components exercised by one cumulative admission check."""
    from nof1_causal_lab.models.model_parameters import referenced_parameter_ids

    states = {
        identity
        for identity in model.state_order
        if model.get_construct(identity).name in keep_names
    }
    edges = {
        edge.id
        for edge in model.execution_edges
        if edge.effect.id in states
        and (edge.cause.id in states or edge.cause.id in model.known_inputs)
    }
    inputs = {
        edge.cause.id
        for edge in model.edges
        if edge.id in edges and edge.cause.id in model.known_inputs
    }
    manifests = {
        identity
        for identity in model.manifest_indicator_order
        if model.indicator_owner(identity).id in states
    }
    confounders = {
        cid
        for (first, second, _), sources in model.induced_dependencies.items()
        if first in states and second in states
        for cid in sources
    }
    constructs = []
    for construct in model.constructs:
        noise = construct.innovation if construct.id in states else None
        initial = construct.initial_state if construct.id in states | confounders else None
        if noise is not None:
            noise = noise.model_copy(
                update={
                    "loadings": tuple(item for item in noise.loadings if item.other_id in states)
                }
            )
        if initial is not None:
            initial = initial.model_copy(
                update={
                    "correlations": tuple(
                        item for item in initial.correlations if item.other_id in states
                    )
                }
            )
        constructs.append(
            construct.model_copy(
                update={
                    "usage": ScientificOnlyConstruct(
                        reason="Outside this cumulative admission scope."
                    )
                    if construct.indicators and construct.id not in states | inputs
                    else construct.usage,
                    "dynamics": construct.dynamics if construct.id in states else (),
                    "innovation": noise,
                    "initial_state": initial,
                    "indicators": tuple(
                        indicator.model_copy(
                            update={
                                "likelihood": indicator.likelihood
                                if indicator.id in manifests
                                else None
                            }
                        )
                        for indicator in construct.indicators
                    ),
                }
            )
        )
    selected_edges = tuple(
        edge.model_copy(update={"mechanisms": edge.mechanisms if edge.id in edges else ()})
        for edge in model.edges
    )
    selected_edges = replace_constructs(selected_edges, constructs)
    referenced = referenced_parameter_ids(*constructs, *selected_edges)
    return model.revised(
        edges=selected_edges,
        parameters=tuple(parameter for parameter in model.parameters if parameter.id in referenced),
    )
