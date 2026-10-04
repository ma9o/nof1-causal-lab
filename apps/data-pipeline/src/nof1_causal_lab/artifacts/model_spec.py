"""The scientific model: stable entities enriched by successive validated revisions."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, override

from pydantic import (
    Field,
    FiniteFloat,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
from nof1_causal_lab.numpyro_json import NumPyroDistribution
from .construct import (
    CausalEdgeSpec,
    ConstructSpec,
    Role,
    TemporalStatus,
    _check_edge_constraint,
    _check_global_constraints,
    endpoint_serialization_scope,
    endpoint_validation_scope,
)
from .duration import Duration
from .expressions import expression_states
from .identity import ConstructId, DistributionId, EdgeId, IndicatorId, MechanismId, ParameterId
from .parameter_spec import ParameterSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nof1_causal_lab.models.model_parameters import ParameterContext

    from .indicator import IndicatorSpec
    from .likelihood import LikelihoodSpec
    from .mechanism import DynamicsMechanismSpec


class ModelSpec(Value):
    """A connected causal graph with owned scientific detail, built to answer the study question."""

    edges: tuple[CausalEdgeSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    distributions: Mapping[DistributionId, NumPyroDistribution] = Field(
        default_factory=dict,
        description=(
            "All explicit probability laws. Members are the parameters and constructs referring to each ID. "
            "Event coordinates are parameters by ID and element ID, then constructs by ID "
            "and time point. A scalar law belongs to one parameter and applies independently to its elements."
        ),
    )
    law_layouts: Mapping[DistributionId, JointLawLayout] = Field(
        default_factory=dict,
        description="Scientific coordinates and production labels of each joint law, beside its native atoms.",
    )

    @computed_field
    @property
    def time_points(self) -> tuple[FiniteFloat, ...]:
        """Joint trajectory coordinates own the model's retained time grid."""
        return next(
            (layout.time_points for layout in self.law_layouts.values() if layout.constructs), ()
        )

    measurement_clock: Duration | None = None

    @field_validator("edges", mode="wrap")
    @classmethod
    def resolve_endpoints(
        cls, value: object, handler: ValidatorFunctionWrapHandler
    ) -> tuple[CausalEdgeSpec, ...]:
        with endpoint_validation_scope(value):
            edges: tuple[CausalEdgeSpec, ...] = handler(value)
            return edges

    @field_serializer("edges", mode="wrap")
    def serialize_edges(  # noqa: ANN201 -- Pydantic wrap serialization must retain the edge field schema; a return annotation replaces it.
        self, value: tuple[CausalEdgeSpec, ...], handler: SerializerFunctionWrapHandler
    ):
        with endpoint_serialization_scope():
            edges: tuple[CausalEdgeSpec, ...] = handler(value)
            return edges

    @property
    def constructs(self) -> tuple[ConstructSpec, ...]:
        """Enumerate the graph's unique endpoints in first-occurrence order."""
        return tuple(self._constructs.values())

    @override
    def __eq__(self, other: object) -> bool:
        """Scientific equality excludes derived caches and their references back to this model."""
        return isinstance(other, ModelSpec) and bool(
            self.model_dump(mode="json") == other.model_dump(mode="json")
        )

    @cached_property
    def _constructs(self) -> Mapping[ConstructId, ConstructSpec]:
        return MappingProxyType(
            {endpoint.id: endpoint for edge in self.edges for endpoint in (edge.cause, edge.effect)}
        )

    @cached_property
    def _edges(self) -> Mapping[EdgeId, CausalEdgeSpec]:
        return MappingProxyType({item.id: item for item in self.edges})

    @cached_property
    def _indicators(self) -> Mapping[IndicatorId, IndicatorSpec]:
        return MappingProxyType({item.observation.id: item for _, item in self.iter_indicators()})

    @cached_property
    def _indicator_owners(self) -> Mapping[IndicatorId, ConstructSpec]:
        return MappingProxyType(
            {item.observation.id: owner for owner, item in self.iter_indicators()}
        )

    @cached_property
    def _parameters(self) -> Mapping[ParameterId, ParameterSpec]:
        return MappingProxyType({item.id: item for item in self.parameters})

    @cached_property
    def _mechanisms(self) -> Mapping[MechanismId, DynamicsMechanismSpec]:
        return MappingProxyType({item.id: item for _, item in self.iter_mechanisms()})

    @cached_property
    def _parameter_contexts(self) -> Mapping[ParameterId, ParameterContext]:
        from nof1_causal_lab.models.model_parameters import parameter_contexts

        return MappingProxyType(dict(parameter_contexts(self)))

    def parameter_context(self, identity: ParameterId) -> ParameterContext:
        return self._parameter_contexts[identity]

    def get_construct(self, identity: ConstructId) -> ConstructSpec:
        return self._constructs[identity]

    def distribution_for(self, identity: ParameterId | ConstructId) -> NumPyroDistribution | None:
        """Resolve the law a quantity participates in, retaining its full joint dependence."""
        quantity = (
            self.parameter(cast("ParameterId", identity))
            if identity.startswith("parameter:")
            else self.get_construct(cast("ConstructId", identity))
        )
        return (
            self.distributions[quantity.distribution] if quantity.distribution is not None else None
        )

    def edge(self, identity: EdgeId) -> CausalEdgeSpec:
        return self._edges[identity]

    def indicator(self, identity: IndicatorId) -> IndicatorSpec:
        return self._indicators[identity]

    def indicator_owner(self, identity: IndicatorId) -> ConstructSpec:
        return self._indicator_owners[identity]

    def parameter(self, identity: ParameterId) -> ParameterSpec:
        return self._parameters[identity]

    def mechanism(self, identity: MechanismId) -> DynamicsMechanismSpec:
        return self._mechanisms[identity]

    def iter_indicators(self) -> Iterator[tuple[ConstructSpec, IndicatorSpec]]:
        for construct in self.constructs:
            for indicator in construct.indicators:
                yield construct, indicator

    def iter_likelihoods(self) -> Iterator[tuple[IndicatorSpec, LikelihoodSpec]]:
        for _, indicator in self.iter_indicators():
            if indicator.likelihood is not None:
                yield indicator, indicator.likelihood

    def iter_mechanisms(
        self,
    ) -> Iterator[tuple[ConstructSpec | CausalEdgeSpec, DynamicsMechanismSpec]]:
        for construct in self.constructs:
            for mechanism in construct.dynamics:
                yield construct, mechanism
        for edge in self.edges:
            for mechanism in edge.mechanisms:
                yield edge, mechanism

    def parameters_for(
        self, identity: ConstructId | EdgeId | IndicatorId | MechanismId
    ) -> tuple[ParameterSpec, ...]:
        return tuple(
            parameter
            for parameter in self.parameters
            if any(owner.id == identity for owner in self.parameter_context(parameter.id).owners)
        )

    @property
    def indicators(self) -> tuple[IndicatorSpec, ...]:
        return tuple(indicator for _, indicator in self.iter_indicators())

    @model_validator(mode="after")
    def validate_references(self) -> ModelSpec:
        references = {
            entity.distribution
            for entity in (*self.parameters, *self.constructs)
            if entity.distribution is not None
        }
        if references != set(self.distributions):
            raise ValueError("Distributions must be referenced and every reference must exist")
        observations = tuple(item.observation for item in self.indicators)
        for label, items in (
            ("edge", self.edges),
            ("indicator", observations),
            ("parameter", self.parameters),
            ("mechanism", tuple(item for _, item in self.iter_mechanisms())),
        ):
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {label} IDs")
        for label, items in (("construct", self.constructs), ("indicator", observations)):
            if len({item.name for item in items}) != len(items):
                raise ValueError(f"Duplicate {label} names")
        endpoint_pairs = [(edge.cause.id, edge.effect.id) for edge in self.edges]
        if len(endpoint_pairs) != len(set(endpoint_pairs)):
            raise ValueError("Declare one causal edge per endpoint pair and compose its mechanisms")
        for edge in self.edges:
            if error := _check_edge_constraint(edge):
                raise ValueError(error)
        if errors := _check_global_constraints(self.edges):
            raise ValueError(errors[0])
        for construct in self.constructs:
            if construct.temporal_status == TemporalStatus.TIME_INVARIANT and construct.dynamics:
                raise ValueError("Time-invariant constructs cannot have intrinsic drift")
        for owner, mechanism in self.iter_mechanisms():
            dependencies = expression_states(mechanism.expression)
            if unknown := dependencies - self._constructs.keys():
                raise ValueError(f"Expression references unknown constructs: {sorted(unknown)}")
            if isinstance(owner, CausalEdgeSpec):
                if owner.cause.id not in dependencies:
                    raise ValueError("An edge expression must reference its causal source")
                allowed = {owner.effect.id} | {
                    edge.cause.id for edge in self.edges if edge.effect.id == owner.effect.id
                }
                if dependencies - allowed:
                    raise ValueError(
                        "Expression dependencies require explicit causal edges to their effect"
                    )
            elif dependencies - {owner.id}:
                raise ValueError("Intrinsic dynamics may reference only their owning construct")
        for indicator, likelihood in self.iter_likelihoods():
            owners = set(likelihood.parsed.loadings)
            unknown = owners - self._constructs.keys()
            if unknown:
                raise ValueError(f"Likelihood references unknown constructs: {sorted(unknown)}")
            if self.indicator_owner(indicator.observation.id).id not in owners:
                raise ValueError(
                    f"Likelihood {indicator.observation.id!r} must include its measured construct"
                )

        from nof1_causal_lab.models.model_parameters import iter_coefficient_uses

        for construct in self.constructs:
            unknown = {
                identity for operand in construct.coefficients for identity in operand.construct_ids
            } - self._constructs.keys()
            if unknown:
                raise ValueError(
                    f"Construct coefficients reference unknown constructs: {sorted(unknown)}"
                )
            if any(
                operand.role in {"diffusion_loading", "initial_correlation"}
                and any(
                    self.get_construct(identity).role == Role.EXOGENOUS
                    for identity in operand.construct_ids
                )
                for operand in construct.coefficients
            ):
                raise ValueError("Noise and initial correlations cannot reference exogenous inputs")
        for use in iter_coefficient_uses(self):
            if isinstance(use.value, str) and use.value not in self._parameters:
                raise ValueError(f"Coefficient {use.slot!r} references an undeclared parameter")
        if unused := self._parameters.keys() - self._parameter_contexts.keys():
            raise ValueError(f"Parameters are not referenced by component slots: {sorted(unused)}")
        for identity, context in self._parameter_contexts.items():
            quantity = context.quantity
            parameter = self.parameter(identity)
            if (
                parameter.transform.kind == "dt_persistence_to_ct_decay"
                and quantity.value != "dynamics_decay"
            ):
                raise ValueError("Persistence coordinates describe a dynamics decay quantity")
        if self.distributions or self.law_layouts:
            from nof1_causal_lab.models.model_distributions import validate_distribution_memberships

            validate_distribution_memberships(self)
        from nof1_causal_lab.models.coefficient_redundancy import validate_coefficient_redundancy

        validate_coefficient_redundancy(self)
        return self

    def require_measurements(self) -> None:
        if self.measurement_clock is None or not self.indicators:
            raise IncompleteModelError("Measurements require a measurement clock and indicators")
