"""Derive execution selections and projection findings from canonical model entities."""

from __future__ import annotations

from collections import defaultdict
from functools import cached_property
from hashlib import sha256
from itertools import combinations
from typing import TYPE_CHECKING, Literal, override

from pydantic import PrivateAttr

from nof1_causal_lab.artifacts.construct import (
    Role,
    TemporalStatus,
    replace_constructs,
)
from nof1_causal_lab.artifacts.execution import StructuralDisposition, StructuralItemDisposition
from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, IndicatorRef
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.compilation_errors import AggregatedCompileError

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ConstructId

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


def unsupported_construct_ids(model: ModelSpec) -> frozenset[ConstructId]:
    """Required unmeasured parents without an executable marginalization rule."""
    return frozenset(
        {
            edge.cause.id
            for edge in model.edges
            if edge.effect.id in model.state_order
            and not edge.cause.indicators
            and edge.cause.id not in model.marginalized_construct_ids
        }
    )


def validate_execution_structure(model: ModelSpec) -> None:
    """Check executable capabilities after the model's intrinsic/reference validation."""
    model.require_measurements()
    states = set(model.state_order)
    errors = []
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
            "Required unmeasured constructs cannot be projected: "
            f"{sorted(unsupported)}. Supply measurements or supported marginalization semantics."
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
            reason = "Required unmeasured cause has no supported marginalization semantics."
        elif construct.id in model.marginalized_construct_ids:
            disposition = StructuralDisposition.MARGINALIZED
            reason = "Safe unobserved exogenous root projected from the executable state vector."
        else:
            disposition = StructuralDisposition.IDENTIFICATION_ONLY
            reason = "Scientific-DAG construct not retained in the executable state."
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
        if edge.effect.id in states and edge.cause.id in states
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
        coefficients = tuple(
            operand
            for operand in construct.coefficients
            if construct.id
            in (states | confounders if operand.role.startswith("initial_") else states)
            and set(operand.construct_ids) <= states
        )
        constructs.append(
            construct.model_copy(
                update={
                    "dynamics": construct.dynamics if construct.id in states else (),
                    "coefficients": coefficients,
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
    parameters = tuple(parameter for parameter in model.parameters if parameter.id in referenced)
    law_references = {item.distribution for item in (*constructs, *parameters)}
    selected = model.revised(
        edges=selected_edges,
        parameters=parameters,
        distributions={
            identity: law
            for identity, law in model.distributions.items()
            if identity in law_references
        },
    )
    return _AdmissionModel.from_selection(
        selected, tuple(key for key in model.state_order if key in states)
    )


class _AdmissionModel(ModelSpec):
    """An operation-local projection; its scope is never authored or serialized."""

    _selected_states: tuple[ConstructId, ...] = PrivateAttr()

    @classmethod
    def from_selection(cls, model: ModelSpec, states: tuple[ConstructId, ...]) -> _AdmissionModel:
        selected = cls.model_validate(model.model_dump(mode="python"))
        object.__setattr__(selected, "_selected_states", states)
        return selected

    @cached_property
    @override
    def state_order(self) -> tuple[ConstructId, ...]:
        return self._selected_states

    @override
    def revised(self, **changes: object) -> ModelSpec:
        return type(self).from_selection(super().revised(**changes), self._selected_states)
