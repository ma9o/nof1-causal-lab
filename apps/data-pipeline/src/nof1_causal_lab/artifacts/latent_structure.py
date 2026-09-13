"""Latent causal-structure artifact models and validation."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001

from .base import ArtifactPayload
from .evidence import LiteratureSource  # noqa: TC001
from .identity import ConstructId, ConstructRef, EdgeId  # noqa: TC001


class Role(StrEnum):
    """A construct role states whether the variable is modeled as endogenous or treated as
    exogenous.
    """

    ENDOGENOUS = "endogenous"
    EXOGENOUS = "exogenous"


class TemporalStatus(StrEnum):
    """Temporal status states whether a construct varies within the individual over time."""

    TIME_VARYING = "time_varying"
    TIME_INVARIANT = "time_invariant"


class Construct(BaseModel):
    """A construct represents a theoretical entity in the scientific causal model."""

    model_config = ConfigDict(extra="forbid")

    id: ConstructId = Field(description="Persistent identity. Preserve when revising or renaming.")
    name: str = Field(description="Construct name (e.g., 'stress', 'sleep_quality')")
    description: str = Field(description="What this theoretical construct represents")
    role: Role = Field(description="'endogenous' (modeled) or 'exogenous' (given)")
    temporal_status: TemporalStatus = Field(
        description="'time_varying' (changes over time) or 'time_invariant' (fixed)"
    )


class CausalEdge(BaseModel):
    """A causal edge declares a directed causal relationship between two constructs."""

    model_config = ConfigDict(extra="forbid")

    id: EdgeId = Field(description="Persistent identity. Preserve when revising the same edge.")
    cause_id: ConstructId = Field(description="Persistent ID of the cause construct.")
    effect_id: ConstructId = Field(description="Persistent ID of the effect construct.")
    description: str = Field(description="Theoretical justification for this causal link")
    lagged: bool = Field(
        default=True,
        description=(
            "If True, effect at t is caused by cause at t-1 (one model_clock tick delay). "
            "If False (contemporaneous), effect at t is caused by cause at t."
        ),
    )
    sources: list[LiteratureSource] = Field(
        default_factory=list,
        description="Literature sources supporting this causal link",
    )


def _check_edge_constraint(
    edge: CausalEdge,
    construct_map: dict[str, Construct],
) -> str | None:
    """Check a single edge against shared latent-structure constraints."""
    cause_construct = construct_map[edge.cause_id]
    effect_construct = construct_map[edge.effect_id]

    if effect_construct.role == Role.EXOGENOUS:
        return f"Exogenous construct '{effect_construct.name}' cannot be an effect"

    if (
        cause_construct.temporal_status == TemporalStatus.TIME_VARYING
        and effect_construct.temporal_status == TemporalStatus.TIME_INVARIANT
    ):
        return (
            f"Time-varying construct '{cause_construct.name}' cannot be a cause of "
            f"time-invariant construct '{effect_construct.name}'. Time-invariant constructs "
            "are fixed within person and cannot have time-varying parents."
        )

    both_time_varying = (
        cause_construct.temporal_status == TemporalStatus.TIME_VARYING
        and effect_construct.temporal_status == TemporalStatus.TIME_VARYING
    )
    both_endogenous = (
        cause_construct.role == Role.ENDOGENOUS and effect_construct.role == Role.ENDOGENOUS
    )
    if not edge.lagged and both_time_varying and both_endogenous:
        return (
            f"Directed contemporaneous edge '{cause_construct.name}' -> '{effect_construct.name}' "
            "between endogenous time-varying latent constructs is excluded by the "
            "latent-structure contract. Represent directed effects between evolving "
            "latent states with lagged=True; reserve same-time dependence for "
            "explicit confounding or diffusion covariance."
        )

    return None


def _check_global_constraints(
    edges: list[CausalEdge],
) -> list[str]:
    """Check latent-structure global constraints."""
    errors: list[str] = []

    contemporaneous_edges = [(edge.cause_id, edge.effect_id) for edge in edges if not edge.lagged]
    if contemporaneous_edges:
        import networkx as nx

        graph = nx.DiGraph(contemporaneous_edges)
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            errors.append(
                f"Contemporaneous edges form cycle(s) within time slice: {cycles}. "
                "Use lagged=true for feedback loops across time."
            )

    return errors


class LatentStructure(BaseModel):
    """A latent structure defines the scientific causal graph of constructs and directed
    relationships.
    """

    model_config = ConfigDict(extra="forbid")

    default_outcome: ConstructRef | None = Field(
        default=None,
        description="Default query target for this workspace; outcome selection is not a construct property.",
    )
    constructs: list[Construct] = Field(
        min_length=1, description="Theoretical constructs in the model"
    )
    edges: list[CausalEdge] = Field(description="Causal edges between constructs")

    @model_validator(mode="after")
    def validate_latent_structure(self) -> LatentStructure:
        """Validate latent structure constraints."""
        for label, items in (("construct", self.constructs), ("edge", self.edges)):
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {label} IDs")
        construct_map = {construct.id: construct for construct in self.constructs}
        if self.default_outcome is not None:
            target = construct_map.get(self.default_outcome.id)
            if target is None:
                raise ValueError("Default outcome references an unknown construct")
            if target.role != Role.ENDOGENOUS:
                raise ValueError("Default outcome must reference an endogenous construct")
        if len({construct.name for construct in self.constructs}) != len(self.constructs):
            raise ValueError("Duplicate construct names")

        for edge in self.edges:
            if edge.cause_id not in construct_map:
                raise ValueError(f"Edge cause '{edge.cause_id}' not in constructs")
            if edge.effect_id not in construct_map:
                raise ValueError(f"Edge effect '{edge.effect_id}' not in constructs")

            error = _check_edge_constraint(edge, construct_map)
            if error:
                raise ValueError(error)

        global_errors = _check_global_constraints(self.edges)
        if global_errors:
            raise ValueError(global_errors[0])

        return self


def validate_latent_structure(
    data: UncheckedJsonObject,
) -> tuple[LatentStructure | None, list[str]]:
    """Validate a latent structure dict, collecting all errors."""
    errors: list[str] = []

    if not isinstance(data, dict):
        return None, ["Input must be a dictionary"]

    constructs = data.get("constructs", [])
    edges = data.get("edges", [])

    if not isinstance(constructs, list):
        errors.append("'constructs' must be a list")
        constructs = []
    if not isinstance(edges, list):
        errors.append("'edges' must be a list")
        edges = []

    valid_constructs: list[Construct] = []
    construct_names: set[str] = set()

    for index, construct_data in enumerate(constructs):
        if not isinstance(construct_data, dict):
            errors.append(f"constructs[{index}]: must be a dictionary")
            continue

        name = construct_data.get("name", f"<unnamed_{index}>")
        if name in construct_names:
            errors.append(f"Duplicate construct name: '{name}'")
        construct_names.add(name)

        try:
            construct = Construct.model_validate(construct_data)
            valid_constructs.append(construct)
        except ValidationError as exc:
            error_msg = str(exc)
            if "validation error" in error_msg.lower():
                for line in error_msg.split("\n")[1:]:
                    line = line.strip()
                    if line and not line.startswith("For further"):
                        errors.append(f"constructs[{index}] ({name}): {line}")
            else:
                errors.append(f"constructs[{index}] ({name}): {error_msg}")

    construct_map = {construct.id: construct for construct in valid_constructs}
    valid_edges: list[CausalEdge] = []
    for index, edge_data in enumerate(edges):
        if not isinstance(edge_data, dict):
            errors.append(f"edges[{index}]: must be a dictionary")
            continue

        cause = edge_data.get("cause_id", "<missing>")
        effect = edge_data.get("effect_id", "<missing>")
        edge_label = f"edges[{index}] ({cause} -> {effect})"

        try:
            edge = CausalEdge.model_validate(edge_data)
        except ValidationError as exc:
            errors.append(f"{edge_label}: {exc}")
            continue

        if edge.cause_id not in construct_map:
            errors.append(f"{edge_label}: cause '{edge.cause_id}' not in constructs")
            continue
        if edge.effect_id not in construct_map:
            errors.append(f"{edge_label}: effect '{edge.effect_id}' not in constructs")
            continue

        constraint_error = _check_edge_constraint(edge, construct_map)
        if constraint_error:
            errors.append(f"{edge_label}: {constraint_error}")
            continue

        valid_edges.append(edge)

    errors.extend(_check_global_constraints(valid_edges))

    if not errors:
        try:
            model = LatentStructure.model_validate(
                {
                    "constructs": valid_constructs,
                    "edges": valid_edges,
                    "default_outcome": data.get("default_outcome"),
                }
            )
            return model, []
        except ValidationError as exc:
            errors.append(f"Final validation failed: {exc}")

    return None, errors


__all__ = [
    "CausalEdge",
    "Construct",
    "LatentStructure",
    "Role",
    "TemporalStatus",
    "validate_latent_structure",
]


class LatentStructureArtifact(ArtifactPayload):
    """This artifact stores the authored causal structure proposed for the research question."""

    latent_structure: LatentStructure
