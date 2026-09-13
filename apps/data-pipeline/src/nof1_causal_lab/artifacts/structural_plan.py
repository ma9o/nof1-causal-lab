"""Typed executable structural plan derived from a causal design."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .identity import ConstructId, EdgeId, IndicatorId  # noqa: TC001
from .latent_structure import CausalEdge, Construct, TemporalStatus
from .measurement_structure import Indicator, KnownInput


class PersistedPlanModel(BaseModel):
    """Strict immutable base for persisted structural-plan contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class StructuralDisposition(StrEnum):
    """A structural disposition classifies how compilation uses or excludes an authored model
    entity.
    """

    RETAINED_STATE = "retained_state"
    KNOWN_INPUT = "known_input"
    MARGINALIZED = "marginalized"
    IDENTIFICATION_ONLY = "identification_only"
    RETAINED_EDGE = "retained_edge"
    PROJECTED_EDGE = "projected_edge"
    MANIFEST = "manifest"
    KNOWN_INPUT_SOURCE = "known_input_source"
    EXCLUDED_INDICATOR = "excluded_indicator"


class StructuralItemDisposition(PersistedPlanModel):
    """An item disposition explains the compilation decision for one identified authored
    entity.
    """

    source_id: str
    source_kind: Literal["construct", "edge", "indicator"]
    disposition: StructuralDisposition
    reason: str

    @model_validator(mode="after")
    def validate_owner_kind(self) -> StructuralItemDisposition:
        allowed = {
            "construct": {
                StructuralDisposition.RETAINED_STATE,
                StructuralDisposition.KNOWN_INPUT,
                StructuralDisposition.MARGINALIZED,
                StructuralDisposition.IDENTIFICATION_ONLY,
            },
            "edge": {StructuralDisposition.RETAINED_EDGE, StructuralDisposition.PROJECTED_EDGE},
            "indicator": {
                StructuralDisposition.MANIFEST,
                StructuralDisposition.KNOWN_INPUT_SOURCE,
                StructuralDisposition.EXCLUDED_INDICATOR,
            },
        }
        if self.disposition not in allowed[self.source_kind]:
            raise ValueError("Structural disposition does not apply to its owner kind")
        return self


class StructuralSemanticCatalog(PersistedPlanModel):
    """The semantic catalog preserves authored definitions under the IDs used by the executable
    plan.
    """

    constructs: dict[ConstructId, Construct]
    edges: dict[EdgeId, CausalEdge]
    indicators: dict[IndicatorId, Indicator]
    model_clock: str

    @model_validator(mode="after")
    def validate_identities_and_references(self) -> StructuralSemanticCatalog:
        for catalog in (self.constructs, self.edges, self.indicators):
            for source_id, item in catalog.items():
                if source_id != item.id:
                    raise ValueError(f"Semantic catalog key {source_id!r} differs from entity ID")
        for edge in self.edges.values():
            for source_id in (edge.cause_id, edge.effect_id):
                construct = self.constructs.get(source_id)
                if construct is None:
                    raise ValueError(f"Semantic edge {edge.id!r} has an inconsistent endpoint")
        for indicator in self.indicators.values():
            construct = self.constructs.get(indicator.construct_id)
            if construct is None:
                raise ValueError(f"Semantic indicator {indicator.id!r} has an inconsistent owner")
        return self


class StructuralEdge(PersistedPlanModel):
    """A structural edge connects retained states or known inputs in the executable model."""

    source_id: EdgeId
    cause_id: ConstructId
    effect_id: ConstructId
    lagged: bool


class StructuralKnownInput(KnownInput, PersistedPlanModel):
    """A structural known input binds an observed indicator to a driver of the executable
    dynamics.
    """

    source_id: str


class StructuralInducedDependency(PersistedPlanModel):
    """An induced dependency records dependence created by projecting explicit latent root
    confounders.
    """

    source_id: str
    between: tuple[ConstructId, ConstructId]
    kind: Literal["innovation_correlation", "initial_state_correlation"]
    source_confounder_ids: tuple[ConstructId, ...]


class StructuralPlan(PersistedPlanModel):
    """A structural plan translates the scientific causal design into the topology used by
    model compilation.
    """

    schema_version: Literal[1] = 1
    semantics: StructuralSemanticCatalog
    state_order: tuple[ConstructId, ...]
    edges: tuple[StructuralEdge, ...]
    manifest_indicator_order: tuple[IndicatorId, ...]
    reference_indicator_ids: dict[ConstructId, IndicatorId]
    known_inputs: tuple[StructuralKnownInput, ...]
    induced_dependencies: tuple[StructuralInducedDependency, ...]
    dispositions: tuple[StructuralItemDisposition, ...]

    @model_validator(mode="after")
    def validate_references(self) -> StructuralPlan:
        """Validate ID uniqueness and all executable-plan references."""
        construct_ids = set(self.semantics.constructs)
        indicator_ids = set(self.semantics.indicators)
        edge_ids = set(self.semantics.edges)
        semantic_id_overlap = (
            (construct_ids & indicator_ids)
            | (construct_ids & edge_ids)
            | (indicator_ids & edge_ids)
        )
        if semantic_id_overlap:
            raise ValueError(
                "StructuralPlan semantic source IDs must be globally unique: "
                f"{sorted(semantic_id_overlap)}"
            )
        construct_names = [item.name for item in self.semantics.constructs.values()]
        if len(construct_names) != len(set(construct_names)):
            raise ValueError("StructuralPlan semantic constructs contain duplicate names")
        indicator_names = [item.name for item in self.semantics.indicators.values()]
        if len(indicator_names) != len(set(indicator_names)):
            raise ValueError("StructuralPlan semantic indicators contain duplicate names")
        source_edge_identities = [
            (item.cause_id, item.effect_id) for item in self.semantics.edges.values()
        ]
        if len(source_edge_identities) != len(set(source_edge_identities)):
            raise ValueError(
                "StructuralPlan semantic edges contain duplicate cause/effect identities"
            )

        if len(self.state_order) != len(set(self.state_order)):
            raise ValueError("StructuralPlan state_order contains duplicate source IDs")
        unknown_states = set(self.state_order) - construct_ids
        if unknown_states:
            raise ValueError(
                f"StructuralPlan state_order references unknown constructs: {sorted(unknown_states)}"
            )

        input_ids = {item.construct_id for item in self.known_inputs}
        if len(input_ids) != len(self.known_inputs):
            raise ValueError("StructuralPlan known_inputs contains duplicate constructs")
        known_input_source_ids = [item.source_id for item in self.known_inputs]
        if len(known_input_source_ids) != len(set(known_input_source_ids)):
            raise ValueError("StructuralPlan known_inputs contains duplicate source IDs")
        overlap = set(self.state_order) & input_ids
        if overlap:
            raise ValueError(
                f"StructuralPlan constructs cannot be both states and known inputs: {sorted(overlap)}"
            )
        for item in self.known_inputs:
            if item.construct_id not in construct_ids:
                raise ValueError(
                    f"StructuralPlan known input references unknown construct {item.construct_id!r}"
                )
            if item.source_indicator_id not in indicator_ids:
                raise ValueError(
                    "StructuralPlan known input references unknown source indicator "
                    f"{item.source_indicator_id!r}"
                )
            indicator = self.semantics.indicators.get(item.source_indicator_id)
            construct = self.semantics.constructs.get(item.construct_id)
            if (
                indicator is not None
                and construct is not None
                and indicator.construct_id != construct.id
            ):
                raise ValueError(
                    "StructuralPlan known-input indicator does not measure its construct: "
                    f"{indicator.name!r} measures {indicator.construct_id!r}, "
                    f"expected {construct.name!r}"
                )

        state_ids = set(self.state_order)
        permitted_causes = state_ids | input_ids
        retained_edge_ids: set[str] = set()
        for edge in self.edges:
            if edge.source_id not in edge_ids:
                raise ValueError(
                    f"StructuralPlan edge references unknown source edge {edge.source_id!r}"
                )
            if edge.source_id in retained_edge_ids:
                raise ValueError(f"StructuralPlan repeats retained source edge {edge.source_id!r}")
            retained_edge_ids.add(edge.source_id)
            if edge.effect_id not in state_ids or edge.cause_id not in permitted_causes:
                raise ValueError(
                    "StructuralPlan edge must point into a retained state and originate "
                    "from a retained state or known input"
                )
            source_edge = self.semantics.edges[edge.source_id]
            cause = self.semantics.constructs[edge.cause_id]
            effect = self.semantics.constructs[edge.effect_id]
            if (
                source_edge.cause_id != cause.id
                or source_edge.effect_id != effect.id
                or source_edge.lagged != edge.lagged
            ):
                raise ValueError(
                    "StructuralPlan retained edge does not match its source edge semantics: "
                    f"{edge.source_id!r}"
                )
            if effect.temporal_status == TemporalStatus.TIME_INVARIANT:
                raise ValueError(
                    "StructuralPlan cannot retain an edge targeting a time-invariant state: "
                    f"{source_edge.cause_id!r} -> {source_edge.effect_id!r}"
                )
        used_input_ids = {edge.cause_id for edge in self.edges if edge.cause_id in input_ids}
        unused_input_ids = input_ids - used_input_ids
        if unused_input_ids:
            raise ValueError(
                "StructuralPlan known inputs must drive at least one retained state: "
                f"{sorted(unused_input_ids)}"
            )

        if len(self.manifest_indicator_order) != len(set(self.manifest_indicator_order)):
            raise ValueError(
                "StructuralPlan manifest_indicator_order contains duplicate source IDs"
            )
        unknown_manifests = set(self.manifest_indicator_order) - indicator_ids
        if unknown_manifests:
            raise ValueError(
                "StructuralPlan manifest order references unknown indicators: "
                f"{sorted(unknown_manifests)}"
            )
        known_input_indicator_ids = {item.source_indicator_id for item in self.known_inputs}
        manifest_input_overlap = set(self.manifest_indicator_order) & known_input_indicator_ids
        if manifest_input_overlap:
            raise ValueError(
                "StructuralPlan indicators cannot be both manifests and known-input sources: "
                f"{sorted(manifest_input_overlap)}"
            )
        manifested_state_ids: set[str] = set()
        for indicator_id in self.manifest_indicator_order:
            construct_id = self.semantics.indicators[indicator_id].construct_id
            if construct_id not in state_ids:
                raise ValueError(
                    "StructuralPlan manifest indicator must measure a retained state: "
                    f"{indicator_id!r}"
                )
            manifested_state_ids.add(construct_id)
        unmanifested_states = state_ids - manifested_state_ids
        if unmanifested_states:
            raise ValueError(
                "StructuralPlan retained states lack manifest indicators: "
                f"{sorted(unmanifested_states)}"
            )
        reference_state_ids = set(self.reference_indicator_ids)
        if reference_state_ids != state_ids:
            raise ValueError(
                "StructuralPlan reference indicators must be total for retained states: "
                f"missing={sorted(state_ids - reference_state_ids)}, "
                f"unknown={sorted(reference_state_ids - state_ids)}"
            )
        manifest_indicator_ids = set(self.manifest_indicator_order)
        for construct_id, indicator_id in self.reference_indicator_ids.items():
            if indicator_id not in manifest_indicator_ids:
                raise ValueError(
                    "StructuralPlan reference indicator must be a retained manifest: "
                    f"{construct_id!r} -> {indicator_id!r}"
                )
            construct = self.semantics.constructs[construct_id]
            indicator = self.semantics.indicators[indicator_id]
            if indicator.construct_id != construct.id:
                raise ValueError(
                    "StructuralPlan reference indicator must measure its retained state: "
                    f"{indicator.name!r} measures {indicator.construct_id!r}, "
                    f"expected {construct.name!r}"
                )

        dependency_source_ids = [item.source_id for item in self.induced_dependencies]
        if len(dependency_source_ids) != len(set(dependency_source_ids)):
            raise ValueError("StructuralPlan induced_dependencies contains duplicate source IDs")
        dependency_targets: set[tuple[str, frozenset[str]]] = set()
        for dependency in self.induced_dependencies:
            if dependency.between[0] == dependency.between[1]:
                raise ValueError(
                    "StructuralPlan induced dependency cannot target one state twice: "
                    f"{dependency.source_id!r}"
                )
            if not set(dependency.between) <= state_ids:
                raise ValueError(
                    "StructuralPlan induced dependency references non-retained states: "
                    f"{dependency.between!r}"
                )
            unknown_sources = set(dependency.source_confounder_ids) - construct_ids
            if unknown_sources:
                raise ValueError(
                    "StructuralPlan induced dependency references unknown confounders: "
                    f"{sorted(unknown_sources)}"
                )
            if not dependency.source_confounder_ids:
                raise ValueError(
                    "StructuralPlan induced dependency must name at least one source confounder: "
                    f"{dependency.source_id!r}"
                )
            if len(dependency.source_confounder_ids) != len(set(dependency.source_confounder_ids)):
                raise ValueError(
                    "StructuralPlan induced dependency repeats a source confounder: "
                    f"{dependency.source_id!r}"
                )
            dependency_target = (dependency.kind, frozenset(dependency.between))
            if dependency_target in dependency_targets:
                raise ValueError(
                    "StructuralPlan repeats an induced dependency target and kind: "
                    f"{dependency.kind!r}, {dependency.between!r}"
                )
            dependency_targets.add(dependency_target)

        disposition_ids = [item.source_id for item in self.dispositions]
        if len(disposition_ids) != len(set(disposition_ids)):
            raise ValueError("StructuralPlan dispositions contain duplicate source IDs")
        expected_dispositions = construct_ids | indicator_ids | edge_ids
        missing_dispositions = expected_dispositions - set(disposition_ids)
        unknown_dispositions = set(disposition_ids) - expected_dispositions
        if missing_dispositions or unknown_dispositions:
            raise ValueError(
                "StructuralPlan source dispositions are not total and exact: "
                f"missing={sorted(missing_dispositions)}, "
                f"unknown={sorted(unknown_dispositions)}"
            )
        disposition_by_id = {item.source_id: item for item in self.dispositions}
        valid_dispositions = {
            "construct": {
                StructuralDisposition.RETAINED_STATE,
                StructuralDisposition.KNOWN_INPUT,
                StructuralDisposition.MARGINALIZED,
                StructuralDisposition.IDENTIFICATION_ONLY,
            },
            "edge": {
                StructuralDisposition.RETAINED_EDGE,
                StructuralDisposition.PROJECTED_EDGE,
            },
            "indicator": {
                StructuralDisposition.MANIFEST,
                StructuralDisposition.KNOWN_INPUT_SOURCE,
                StructuralDisposition.EXCLUDED_INDICATOR,
            },
        }
        expected_kind_by_id = {
            **dict.fromkeys(construct_ids, "construct"),
            **dict.fromkeys(edge_ids, "edge"),
            **dict.fromkeys(indicator_ids, "indicator"),
        }
        for source_id, disposition in disposition_by_id.items():
            expected_kind = expected_kind_by_id[source_id]
            if disposition.source_kind != expected_kind:
                raise ValueError(
                    f"StructuralPlan disposition {source_id!r} has source_kind "
                    f"{disposition.source_kind!r}, expected {expected_kind!r}"
                )
            if disposition.disposition not in valid_dispositions[expected_kind]:
                raise ValueError(
                    f"StructuralPlan disposition {source_id!r} is invalid for "
                    f"{expected_kind}: {disposition.disposition!r}"
                )

        expected_executable_dispositions = {
            **dict.fromkeys(state_ids, StructuralDisposition.RETAINED_STATE),
            **dict.fromkeys(input_ids, StructuralDisposition.KNOWN_INPUT),
            **dict.fromkeys(retained_edge_ids, StructuralDisposition.RETAINED_EDGE),
            **dict.fromkeys(
                edge_ids - retained_edge_ids,
                StructuralDisposition.PROJECTED_EDGE,
            ),
            **dict.fromkeys(
                set(self.manifest_indicator_order),
                StructuralDisposition.MANIFEST,
            ),
            **dict.fromkeys(
                known_input_indicator_ids,
                StructuralDisposition.KNOWN_INPUT_SOURCE,
            ),
            **dict.fromkeys(
                indicator_ids - set(self.manifest_indicator_order) - known_input_indicator_ids,
                StructuralDisposition.EXCLUDED_INDICATOR,
            ),
        }
        for source_id, expected in expected_executable_dispositions.items():
            actual = disposition_by_id[source_id].disposition
            if actual != expected:
                raise ValueError(
                    f"StructuralPlan disposition {source_id!r} is {actual!r}, expected {expected!r}"
                )
        for dependency in self.induced_dependencies:
            invalid_sources = [
                source_id
                for source_id in dependency.source_confounder_ids
                if disposition_by_id[source_id].disposition != StructuralDisposition.MARGINALIZED
            ]
            if invalid_sources:
                raise ValueError(
                    "StructuralPlan induced dependencies must originate from marginalized "
                    f"constructs: {invalid_sources}"
                )

        operational_source_ids = known_input_source_ids + dependency_source_ids
        if len(operational_source_ids) != len(set(operational_source_ids)):
            raise ValueError(
                "StructuralPlan operational source IDs must be unique across known inputs "
                "and induced dependencies"
            )
        operational_semantic_overlap = set(operational_source_ids) & expected_dispositions
        if operational_semantic_overlap:
            raise ValueError(
                "StructuralPlan operational source IDs collide with semantic source IDs: "
                f"{sorted(operational_semantic_overlap)}"
            )
        return self


__all__ = [
    "StructuralDisposition",
    "StructuralEdge",
    "StructuralInducedDependency",
    "StructuralItemDisposition",
    "StructuralKnownInput",
    "StructuralPlan",
    "StructuralSemanticCatalog",
]


class StructuralPlanArtifact(PersistedPlanModel):
    """The structural-plan file stores the self-contained executable plan."""

    structural_plan: StructuralPlan
