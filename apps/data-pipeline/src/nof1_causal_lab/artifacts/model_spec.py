"""The scientific model: stable entities enriched by successive validated revisions."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, cast, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_serializer,
    field_validator,
    model_validator,
)

from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.numpyro_json import NumPyroDistribution  # noqa: TC001

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
from .duration import parse_duration_to_hours
from .expressions import expression_coefficients, expression_states
from .identity import (  # noqa: TC001
    ConstructId,
    DistributionId,
    EdgeId,
    IndicatorId,
    MechanismId,
    ParameterId,
)
from .indicator import IndicatorSpec  # noqa: TC001
from .parameter_spec import ParameterSpec  # noqa: TC001

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nof1_causal_lab.models.model_structure import DependencyKey

    from .execution import AnchorCertificate, StructuralItemDisposition
    from .likelihood import LikelihoodSpec
    from .mechanism import DynamicsMechanismSpec


class ModelSpec(BaseModel):
    """An evolving research question and connected causal graph with owned scientific detail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str | None = Field(default=None, min_length=1)
    edges: tuple[CausalEdgeSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    distributions: dict[DistributionId, NumPyroDistribution] = Field(
        default_factory=dict,
        description=(
            "All explicit probability laws. Members are the parameters and constructs referring to each ID. "
            "Event coordinates are parameters by ID and element ID, then constructs by ID "
            "and time point. A scalar law belongs to one parameter and applies independently to its elements."
        ),
    )
    time_points: tuple[FiniteFloat, ...] = ()
    measurement_clock: str | None = None
    default_outcome: ConstructId | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("question text must be non-empty")
        return value

    @field_validator("edges", mode="wrap")
    @classmethod
    def resolve_endpoints(cls, value, handler):
        with endpoint_validation_scope(value):
            return handler(value)

    @field_serializer("edges", mode="wrap")
    def serialize_edges(self, value, handler):
        with endpoint_serialization_scope():
            return handler(value)

    @property
    def constructs(self) -> tuple[ConstructSpec, ...]:
        """Enumerate the graph's unique endpoints in first-occurrence order."""
        return tuple(self._constructs.values())

    @override
    def __eq__(self, other: object) -> bool:
        """Scientific equality excludes derived caches and their references back to this model."""
        return isinstance(other, ModelSpec) and self.model_dump(mode="json") == other.model_dump(
            mode="json"
        )

    @field_validator("measurement_clock")
    @classmethod
    def validate_clock(cls, value: str | None) -> str | None:
        if value is not None:
            parse_duration_to_hours(value)
        return value

    @cached_property
    def _constructs(self) -> dict[ConstructId, ConstructSpec]:
        return {
            endpoint.id: endpoint for edge in self.edges for endpoint in (edge.cause, edge.effect)
        }

    @cached_property
    def _edges(self) -> dict[EdgeId, CausalEdgeSpec]:
        return {item.id: item for item in self.edges}

    @cached_property
    def _indicators(self) -> dict[IndicatorId, IndicatorSpec]:
        return {item.id: item for _, item in self.iter_indicators()}

    @cached_property
    def _indicator_owners(self) -> dict[IndicatorId, ConstructSpec]:
        return {item.id: owner for owner, item in self.iter_indicators()}

    @cached_property
    def _parameters(self) -> dict[ParameterId, ParameterSpec]:
        return {item.id: item for item in self.parameters}

    @cached_property
    def _mechanisms(self) -> dict[MechanismId, DynamicsMechanismSpec]:
        return {item.id: item for _, item in self.iter_mechanisms()}

    @cached_property
    def state_order(self) -> tuple[ConstructId, ...]:
        """Execution axes reference the existing measured constructs."""
        self.require_measurements()
        retained = [item for item in self.constructs if item.indicators]
        return tuple(
            item.id
            for static in (False, True)
            for item in retained
            if (item.temporal_status == TemporalStatus.TIME_INVARIANT) == static
        )

    @cached_property
    def execution_edges(self) -> tuple[CausalEdgeSpec, ...]:
        states = set(self.state_order)
        return tuple(
            edge for edge in self.edges if edge.cause.id in states and edge.effect.id in states
        )

    @cached_property
    def manifest_indicator_order(self) -> tuple[IndicatorId, ...]:
        states = set(self.state_order)
        return tuple(
            indicator.id for owner, indicator in self.iter_indicators() if owner.id in states
        )

    @cached_property
    def reference_indicator_ids(self) -> dict[ConstructId, IndicatorId]:
        from nof1_causal_lab.utils.causal_design import choose_reference_indicator

        references = {}
        for identity in self.state_order:
            indicator = choose_reference_indicator(
                [item.model_dump(mode="json") for item in self.get_construct(identity).indicators]
            )
            assert indicator is not None
            references[identity] = indicator["id"]
        return references

    @cached_property
    def marginalized_construct_ids(self) -> frozenset[ConstructId]:
        from nof1_causal_lab.models.model_structure import marginalized_construct_ids

        return marginalized_construct_ids(self)

    @cached_property
    def induced_dependencies(self) -> dict[DependencyKey, tuple[ConstructId, ...]]:
        from nof1_causal_lab.models.model_structure import induced_dependencies

        return induced_dependencies(self)

    @cached_property
    def structural_dispositions(self) -> tuple[StructuralItemDisposition, ...]:
        from nof1_causal_lab.models.model_structure import structural_dispositions

        return structural_dispositions(self)

    def require_execution_structure(self) -> None:
        from nof1_causal_lab.models.model_structure import validate_execution_structure

        validate_execution_structure(self)

    @cached_property
    def _parameter_contexts(self):
        from nof1_causal_lab.models.model_parameters import parameter_contexts

        return parameter_contexts(self)

    def parameter_context(self, identity: ParameterId):
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

    def parameters_for(  # noqa: V105 - public canonical ownership accessor
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

    @property
    def model_clock_days(self) -> float:
        if self.measurement_clock is None:
            raise ValueError("The model has no measurement clock")
        return parse_duration_to_hours(self.measurement_clock) / 24.0

    def revised(self, **changes: object) -> ModelSpec:
        """Validate a whole candidate; cached indexes belong only to their original value."""
        return ModelSpec.model_validate(
            {**{name: getattr(self, name) for name in type(self).model_fields}, **changes}
        )

    @model_validator(mode="after")
    def validate_references(self) -> ModelSpec:
        references = {
            entity.distribution
            for entity in (*self.parameters, *self.constructs)
            if entity.distribution is not None
        }
        if references != self.distributions.keys():
            raise ValueError("Distributions must be referenced and every reference must exist")
        if any(b <= a for a, b in zip(self.time_points, self.time_points[1:], strict=False)):
            raise ValueError("Trajectory time points must be strictly increasing")
        if any(item.distribution is not None for item in self.constructs) and not self.time_points:
            raise ValueError("Construct trajectory distributions require time points")
        for label, items in (
            ("construct", self.constructs),
            ("edge", self.edges),
            ("indicator", self.indicators),
            ("parameter", self.parameters),
            ("mechanism", tuple(item for _, item in self.iter_mechanisms())),
        ):
            ids = [item.id for item in items]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate {label} IDs")
        for label, items in (("construct", self.constructs), ("indicator", self.indicators)):
            if len({item.name for item in items}) != len(items):
                raise ValueError(f"Duplicate {label} names")
        if self.default_outcome is not None:
            target = self._constructs.get(self.default_outcome)
            if target is None:
                raise ValueError("Default outcome references an unknown construct")
            if target.role != Role.ENDOGENOUS:
                raise ValueError("Default outcome must reference an endogenous construct")
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
                if mechanism.kind == "potential":
                    raise ValueError(
                        "Potentials belong to nodes; directed edges require drift terms"
                    )
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
            for operand in expression_coefficients(mechanism.expression):
                reference = operand.value
                if isinstance(reference, str) and reference in self._parameters:
                    value = self.parameter(reference).value
                    if value is not None:
                        operand.validate_value(value)
        for indicator, likelihood in self.iter_likelihoods():
            owners = set(likelihood.terms.loadings)
            unknown = owners - self._constructs.keys()
            if unknown:
                raise ValueError(f"Likelihood references unknown constructs: {sorted(unknown)}")
            if self.indicator_owner(indicator.id).id not in owners:
                raise ValueError(f"Likelihood {indicator.id!r} must include its measured construct")

        from nof1_causal_lab.models.model_parameters import iter_coefficient_uses

        for construct in self.constructs:
            unknown = {
                identity for operand in construct.coefficients for identity in operand.construct_ids
            } - self._constructs.keys()
            if unknown:
                raise ValueError(
                    f"Construct coefficients reference unknown constructs: {sorted(unknown)}"
                )
        for use in iter_coefficient_uses(self):
            for owner in use.owners:
                if owner.id not in {
                    *self._constructs,
                    *self._edges,
                    *self._indicators,
                    *self._mechanisms,
                }:
                    raise ValueError(f"Coefficient {use.slot!r} references an unknown entity")
            if isinstance(use.value, str) and use.value not in self._parameters:
                raise ValueError(f"Coefficient {use.slot!r} references an undeclared parameter")
        if unused := self._parameters.keys() - self._parameter_contexts.keys():
            raise ValueError(f"Parameters are not referenced by component slots: {sorted(unused)}")
        for identity, context in self._parameter_contexts.items():
            quantity = context.quantity
            parameter = self.parameter(identity)
            if (
                parameter.distribution_transform == "dt_persistence_to_ct_decay"
                and quantity.value != "dynamics_decay"
            ):
                raise ValueError("Persistence coordinates describe a dynamics decay quantity")
        if self.distributions:
            from nof1_causal_lab.models.model_distributions import validate_distribution_memberships

            validate_distribution_memberships(self)
        return self

    def require_question(self) -> str:
        if self.question is None:
            raise IncompleteModelError("Authoring and extraction require a research question")
        return self.question

    def require_measurements(self) -> None:
        if self.measurement_clock is None or not self.indicators:
            raise IncompleteModelError("Measurements require a measurement clock and indicators")

    def require_priors(self) -> None:
        missing = [
            parameter.id
            for parameter in self.parameters
            if parameter.distribution is None and parameter.value is None
        ]
        if missing:
            raise IncompleteModelError(f"Compilation requires declared prior laws for {missing}")

    def check_execution(self) -> tuple[AnchorCertificate, ...]:
        from nof1_causal_lab.models.model_checks import check_execution

        return check_execution(self)
