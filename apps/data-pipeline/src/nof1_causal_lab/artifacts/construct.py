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
    model_validator,
)

from nof1_causal_lab.distributions import DistributionFamily

from .evidence import LiteratureSource  # noqa: TC001
from .expressions import CONSTRUCT_COEFFICIENT_ROLES, CoefficientExpression, CoefficientRole
from .identity import (
    ConstructId,
    ConstructRef,
    DistributionId,
    EdgeId,
    ParameterId,
)
from .indicator import IndicatorSpec  # noqa: TC001
from .mechanism import DynamicsMechanismSpec  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from nof1_causal_lab.json_types import JsonObject


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


class ConstructSpec(BaseModel):
    """A specification of a theoretical entity in the scientific causal model."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    id: ConstructId = Field(description="Persistent identity. Preserve when revising or renaming.")
    name: str = Field(description="Construct name (e.g., 'stress', 'sleep_quality')")
    description: str = Field(description="What this theoretical construct represents")
    indicators: tuple[IndicatorSpec, ...] = ()
    dynamics: tuple[DynamicsMechanismSpec, ...] = ()
    coefficients: tuple[CoefficientExpression, ...] = ()
    innovation_family: Literal[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T] = (
        DistributionFamily.GAUSSIAN
    )
    distribution: DistributionId | None = Field(
        default=None,
        description="Membership in a trajectory law in ModelSpec.distributions on ModelSpec.time_points.",
    )
    role: Role = Field(
        description="'endogenous' or 'exogenous' (no modeled causal parents; may still be uncertain)"
    )
    temporal_status: TemporalStatus = Field(
        description="'time_varying' (changes over time) or 'time_invariant' (fixed)"
    )

    def coefficient(
        self, role: CoefficientRole, *, construct_ids: tuple[ConstructId, ...] = ()
    ) -> float | ParameterId | None:
        """Read one scalar use by its role and scientific references."""
        return next(
            (
                operand.value
                for operand in self.coefficients
                if operand.role == role and operand.construct_ids == construct_ids
            ),
            None,
        )

    def with_coefficients(self, *operands: CoefficientExpression) -> ConstructSpec:
        """Replace the specified uses while preserving other authored coefficients."""
        values = {(operand.role, operand.construct_ids): operand for operand in self.coefficients}
        values.update({(operand.role, operand.construct_ids): operand for operand in operands})
        return self.model_copy(update={"coefficients": tuple(values.values())})

    @model_validator(mode="after")
    def validate_coefficients(self) -> ConstructSpec:
        seen = set()
        for operand in self.coefficients:
            if operand.role not in CONSTRUCT_COEFFICIENT_ROLES:
                raise ValueError(f"{operand.role} belongs in a dynamics or likelihood expression")
            key = (operand.role, operand.construct_ids)
            if key in seen:
                raise ValueError(f"Duplicate construct coefficient {key}")
            seen.add(key)
            coupled = operand.role in {"diffusion_loading", "initial_correlation"}
            if len(operand.construct_ids) != int(coupled) or self.id in operand.construct_ids:
                raise ValueError(
                    "Joint coefficients require one other construct; scalar uses require none"
                )
            if (
                operand.role == "process_degrees_of_freedom"
                and self.innovation_family != DistributionFamily.STUDENT_T
            ):
                raise ValueError("Only Student-t innovations have a degrees-of-freedom coefficient")
        return self


@dataclass
class _EndpointScope:
    definitions: dict[str, object] = field(default_factory=dict)
    constructs: dict[ConstructId, ConstructSpec] = field(default_factory=dict)


_endpoint_scope: ContextVar[_EndpointScope | None] = ContextVar("endpoint_scope", default=None)
_serialized_endpoints: ContextVar[set[ConstructId] | None] = ContextVar(
    "serialized_endpoints", default=None
)


def _validate_endpoint(value: object, handler: ValidatorFunctionWrapHandler) -> ConstructSpec:
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


def _serialize_endpoint(value: ConstructSpec) -> ConstructSpec | ConstructRef:
    seen = _serialized_endpoints.get()
    if seen is not None:
        if value.id in seen:
            return ConstructRef(id=value.id)
        seen.add(value.id)
    return value


ConstructEndpoint = Annotated[
    ConstructSpec,
    WrapValidator(_validate_endpoint, json_schema_input_type=ConstructSpec | ConstructRef),
    PlainSerializer(
        _serialize_endpoint, return_type=ConstructSpec | ConstructRef, when_used="json"
    ),
]


class CausalEdgeSpec(BaseModel):
    """A specification of a directed causal relationship between two constructs."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    id: EdgeId = Field(description="Persistent identity. Preserve when revising the same edge.")
    mechanisms: tuple[DynamicsMechanismSpec, ...] = ()
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
    values: object, *, endpoints: Iterable[ConstructSpec] = ()
) -> Iterator[None]:
    """Resolve forward and shared JSON references within exactly one causal graph."""
    scope = _EndpointScope()
    for endpoint in endpoints:
        scope.definitions[endpoint.id] = endpoint
        scope.constructs[endpoint.id] = endpoint
    if isinstance(values, (list, tuple)):
        for edge in values:
            if isinstance(edge, CausalEdgeSpec):
                edge_endpoints: tuple[object, object] = (edge.cause, edge.effect)
            elif isinstance(edge, dict):
                edge_endpoints = (edge.get("cause"), edge.get("effect"))
            else:
                continue  # The edge validator reports malformed edge values.
            for endpoint in edge_endpoints:
                if isinstance(endpoint, ConstructSpec):
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


def serialize_edge_references(edges: Iterable[CausalEdgeSpec]) -> list[JsonObject]:
    """An authoring view of edge updates whose endpoint definitions belong to the base graph."""
    edges = tuple(edges)
    with endpoint_serialization_scope(
        endpoint.id for edge in edges for endpoint in (edge.cause, edge.effect)
    ):
        return [edge.model_dump(mode="json") for edge in edges]


def replace_constructs(
    edges: Iterable[CausalEdgeSpec], replacements: Iterable[ConstructSpec]
) -> tuple[CausalEdgeSpec, ...]:
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


def _check_edge_constraint(edge: CausalEdgeSpec) -> str | None:
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
    edges: tuple[CausalEdgeSpec, ...],
) -> list[str]:
    """Check latent-structure global constraints."""
    errors: list[str] = []

    import networkx as nx

    graph = nx.Graph((edge.cause.id, edge.effect.id) for edge in edges)
    if edges and not nx.is_connected(graph):
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
