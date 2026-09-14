"""Scientific constructs own observations, intrinsic dynamics, and authored usage."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)

from nof1_causal_lab.numpyro_json import NumPyroDistribution  # noqa: TC001

from .evidence import LiteratureSource  # noqa: TC001
from .identity import ConstructId, ConstructRef, DistributionId, EdgeId, IndicatorId
from .indicator import Indicator  # noqa: TC001
from .mechanism import DynamicsMechanism  # noqa: TC001
from .state_distribution import InitialStateSpec, InnovationSpec  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from nof1_causal_lab.json_types import JsonObject


class KnownInput(BaseModel):
    """An observed-input declaration binds a construct to its measured driver trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    kind: Literal["known_input"] = "known_input"
    source_indicator_id: IndicatorId
    scale: float = Field(
        default=1.0, gt=0.0, description="Positive divisor applied before inference"
    )
    missing_policy: Literal["zero", "forward_fill"] = "zero"


class ScientificOnlyConstruct(BaseModel):
    """A scientific-only declaration excludes an identified construct from executable states."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    kind: Literal["scientific_only"] = "scientific_only"
    reason: str = Field(min_length=1)


type ConstructUsage = Annotated[KnownInput | ScientificOnlyConstruct, Field(discriminator="kind")]


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

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    id: ConstructId = Field(description="Persistent identity. Preserve when revising or renaming.")
    name: str = Field(description="Construct name (e.g., 'stress', 'sleep_quality')")
    description: str = Field(description="What this theoretical construct represents")
    indicators: tuple[Indicator, ...] = ()
    dynamics: tuple[DynamicsMechanism, ...] = ()
    innovation: InnovationSpec | None = None
    initial_state: InitialStateSpec | None = None
    distribution: NumPyroDistribution | DistributionId | None = Field(
        default=None,
        description="Law of this construct's trajectory on ModelSpec.time_points; may be joint.",
    )
    usage: ConstructUsage | None = None
    role: Role = Field(description="'endogenous' (modeled) or 'exogenous' (given)")
    temporal_status: TemporalStatus = Field(
        description="'time_varying' (changes over time) or 'time_invariant' (fixed)"
    )


@dataclass
class _EndpointScope:
    definitions: dict[str, object] = field(default_factory=dict)
    constructs: dict[ConstructId, Construct] = field(default_factory=dict)


_endpoint_scope: ContextVar[_EndpointScope | None] = ContextVar("endpoint_scope", default=None)
_serialized_endpoints: ContextVar[set[ConstructId] | None] = ContextVar(
    "serialized_endpoints", default=None
)


def _validate_endpoint(value: object, handler: ValidatorFunctionWrapHandler) -> Construct:
    scope = _endpoint_scope.get()
    if isinstance(value, ConstructRef) or (
        isinstance(value, dict) and value.get("kind") == "construct"
    ):
        reference = ConstructRef.model_validate(value)
        if scope is None or reference.id not in scope.definitions:
            raise ValueError(f"Undefined construct endpoint {reference.id!r}")
        if reference.id in scope.constructs:
            return scope.constructs[reference.id]
        value = scope.definitions[reference.id]
    construct = handler(value)
    if scope is None:
        return construct
    if construct.id in scope.constructs:
        canonical = scope.constructs[construct.id]
        if canonical.model_dump(mode="json") != construct.model_dump(mode="json"):
            raise ValueError(f"Conflicting definitions of construct {construct.id!r}")
        return canonical
    scope.constructs[construct.id] = construct
    return construct


def _serialize_endpoint(value: Construct) -> Construct | ConstructRef:
    seen = _serialized_endpoints.get()
    if seen is not None:
        if value.id in seen:
            return ConstructRef(id=value.id)
        seen.add(value.id)
    return value


ConstructEndpoint = Annotated[
    Construct,
    WrapValidator(_validate_endpoint, json_schema_input_type=Construct | ConstructRef),
    PlainSerializer(_serialize_endpoint, return_type=Construct | ConstructRef, when_used="json"),
]


class CausalEdge(BaseModel):
    """A causal edge declares a directed causal relationship between two constructs."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    id: EdgeId = Field(description="Persistent identity. Preserve when revising the same edge.")
    mechanisms: tuple[DynamicsMechanism, ...] = ()
    cause: ConstructEndpoint = Field(
        description="Cause construct; shared endpoints have one identity."
    )
    effect: ConstructEndpoint = Field(
        description="Effect construct; shared endpoints have one identity."
    )
    description: str = Field(description="Theoretical justification for this causal link")
    lagged: bool = Field(
        default=True,
        description=(
            "If True, effect at t is caused by cause at t-1 (one model_clock tick delay). "
            "If False (contemporaneous), effect at t is caused by cause at t."
        ),
    )
    sources: tuple[LiteratureSource, ...] = Field(
        default_factory=tuple,
        description="Literature sources supporting this causal link",
    )


@contextmanager
def endpoint_validation_scope(
    values: object, *, endpoints: Iterable[Construct] = ()
) -> Iterator[None]:
    """Resolve forward and shared JSON references within exactly one causal graph."""
    scope = _EndpointScope()
    for endpoint in endpoints:
        scope.definitions[endpoint.id] = endpoint
        scope.constructs[endpoint.id] = endpoint
    if isinstance(values, (list, tuple)):
        for edge in values:
            if isinstance(edge, CausalEdge):
                edge_endpoints: tuple[object, object] = (edge.cause, edge.effect)
            elif isinstance(edge, dict):
                edge_endpoints = (edge.get("cause"), edge.get("effect"))
            else:
                continue  # The edge validator reports malformed edge values.
            for endpoint in edge_endpoints:
                if isinstance(endpoint, Construct):
                    scope.definitions.setdefault(endpoint.id, endpoint)
                elif (
                    isinstance(endpoint, dict)
                    and "name" in endpoint
                    and isinstance(endpoint.get("id"), str)
                ):
                    identity = endpoint.get("id")
                    assert isinstance(identity, str)
                    scope.definitions.setdefault(identity, endpoint)
    token = _endpoint_scope.set(scope)
    try:
        yield
    finally:
        _endpoint_scope.reset(token)


@contextmanager
def endpoint_serialization_scope(referenced: Iterable[ConstructId] = ()) -> Iterator[None]:
    """Write each endpoint definition once, followed by references to its identity."""
    token = _serialized_endpoints.set(set(referenced))
    try:
        yield
    finally:
        _serialized_endpoints.reset(token)


def serialize_edge_references(edges: Iterable[CausalEdge]) -> list[JsonObject]:
    """An authoring view of edge updates whose endpoint definitions belong to the base graph."""
    edges = tuple(edges)
    with endpoint_serialization_scope(
        endpoint.id for edge in edges for endpoint in (edge.cause, edge.effect)
    ):
        return [edge.model_dump(mode="json") for edge in edges]


def replace_constructs(
    edges: Iterable[CausalEdge], replacements: Iterable[Construct]
) -> tuple[CausalEdge, ...]:
    """Replace endpoint values throughout a graph without changing its membership."""
    edges = tuple(edges)
    by_id = {construct.id: construct for construct in replacements}
    identities = {endpoint.id for edge in edges for endpoint in (edge.cause, edge.effect)}
    if unknown := by_id.keys() - identities:
        raise ValueError(f"Cannot replace constructs absent from the graph: {sorted(unknown)}")
    return tuple(
        edge.model_copy(
            update={
                "cause": by_id.get(edge.cause.id, edge.cause),
                "effect": by_id.get(edge.effect.id, edge.effect),
            }
        )
        for edge in edges
    )


def _check_edge_constraint(edge: CausalEdge) -> str | None:
    """Check a single edge against shared latent-structure constraints."""
    cause_construct = edge.cause
    effect_construct = edge.effect

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
            "scientific model contract. Represent directed effects between evolving "
            "latent states with lagged=True; reserve same-time dependence for "
            "explicit confounding or diffusion covariance."
        )

    return None


def _check_global_constraints(
    edges: tuple[CausalEdge, ...],
) -> list[str]:
    """Check latent-structure global constraints."""
    errors: list[str] = []

    import networkx as nx

    graph = nx.Graph((edge.cause.id, edge.effect.id) for edge in edges)
    if not nx.is_connected(graph):
        errors.append("The scientific model must be one connected causal graph")

    contemporaneous_edges = [(edge.cause.id, edge.effect.id) for edge in edges if not edge.lagged]
    if contemporaneous_edges:
        graph = nx.DiGraph(contemporaneous_edges)
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            errors.append(
                f"Contemporaneous edges form cycle(s) within time slice: {cycles}. "
                "Use lagged=true for feedback loops across time."
            )

    return errors
