"""Bind scalar expressions to exact JAX vector-field components."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax.numpy as jnp
import numpyro

from nof1_causal_lab.artifacts.expressions import (
    CoefficientExpression,
    Expression,
    StateExpression,
    expression_coefficients,
    expression_states,
)
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.ssm.compile.expressions import compile_expression
from nof1_causal_lab.models.ssm.structure.sites import (
    SiteDescriptor,
)
from nof1_causal_lab.utils.immutability import freeze_fields

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from jax import Array

    from nof1_causal_lab.artifacts.identity import ConstructId, ParameterId
    from nof1_causal_lab.artifacts.likelihood import LinkFunction
    from nof1_causal_lab.artifacts.parameter import ParameterCoordinate

    from .spec import PriorFn


class ExpressionComponent(eqx.Module):
    """One exact scalar contribution, with IDs bound once before numerical execution."""

    target: int = eqx.field(static=True)
    edge_owned: bool = eqx.field(static=True)
    expression: Expression = eqx.field(static=True)
    evaluate_fn: Callable[[Array, Mapping[str, Array]], Array] = eqx.field(static=True)

    def evaluate(self, values: Array, params: Mapping[str, Array]) -> Array:
        return self.evaluate_fn(values, params)

    def contribute(self, accumulator, eta, eta_per_edge, _t, params: Mapping[str, Array]):
        values = eta_per_edge[self.target] if self.edge_owned else eta
        return accumulator.at[self.target].add(self.evaluate(values, params))


@dataclass(frozen=True)
class ExpressionComponentSpec:
    """An expression with resolved state coordinates and direct scientific parameter keys."""

    expression: Expression
    target: int
    state_ids: tuple[ConstructId, ...]
    source: int | None
    kind: Literal["drift", "potential"] = "drift"

    state_index: Mapping[ConstructId, int] = field(init=False)
    coefficients: Mapping[CoefficientExpression, float | ParameterId] = field(init=False)

    def __post_init__(self) -> None:
        from nof1_causal_lab.compilation_errors import IncompleteModelError

        object.__setattr__(
            self,
            "state_index",
            MappingProxyType({identity: index for index, identity in enumerate(self.state_ids)}),
        )
        resolved: dict[CoefficientExpression, float | ParameterId] = {}
        for operand in expression_coefficients(self.expression):
            if operand.value is None:
                raise IncompleteModelError(f"Expression requires its {operand.role} coefficient")
            resolved[operand] = operand.value
        object.__setattr__(self, "coefficients", MappingProxyType(resolved))
        if self.kind == "potential" and (self.source is not None or self.sources - {self.target}):
            raise ValueError("A node potential may depend only on its owning state")
        freeze_fields(self)

    @property
    def edge_owned(self) -> bool:
        return self.source is not None

    @property
    def sources(self) -> frozenset[int]:
        return frozenset(self.state_index[key] for key in expression_states(self.expression))

    @property
    def parameters(self) -> tuple[tuple[ParameterId, CoefficientExpression], ...]:
        return tuple(
            (operand.value, operand)
            for operand in expression_coefficients(self.expression)
            if isinstance(operand.value, str)
        )

    def build(self) -> ExpressionComponent:
        operands, numerical = compile_expression(self.expression)

        def evaluate(values: Array, params: Mapping[str, Array]) -> Array:
            def operand_value(operand: StateExpression | CoefficientExpression) -> Array:
                match operand:
                    case StateExpression():
                        return values[self.state_index[operand.construct_id]]
                    case CoefficientExpression():
                        reference = self.coefficients[operand]
                        return (
                            params[reference]
                            if isinstance(reference, str)
                            else jnp.asarray(reference)
                        )

            return jnp.asarray(numerical(0.0, *(operand_value(operand) for operand in operands)))

        return ExpressionComponent(
            target=self.target,
            edge_owned=self.edge_owned,
            expression=self.expression,
            evaluate_fn=evaluate,
        )

    def parameter_sites(self, prefix: str) -> Iterator[tuple[ParameterId, SiteDescriptor]]:
        positions = (
            ((self.target, self.source, *sorted(self.sources - {self.source})),)
            if self.source is not None
            else (self.target,)
        )
        for index, (identity, operand) in enumerate(self.parameters):
            meaning = operand.meaning
            prior_field = meaning.quantity.value
            if meaning.quantity == SiteKind.DYNAMICS_WEIGHT:
                prior_field = (
                    "multiplicative_weight" if len(self.sources) > 1 else "linear_edge_weight"
                )
            yield (
                identity,
                SiteDescriptor(
                    name=f"{prefix}_p{index}",
                    shape=(),
                    support=meaning.support,
                    assembly_group="dynamics",
                    site_kind=meaning.quantity,
                    positions=positions,
                    prior_field=prior_field,
                ),
            )

    def iter_sites(self, prefix: str, *, n_latent: int) -> Iterator[SiteDescriptor]:
        if not 0 <= self.target < n_latent or any(i >= n_latent for i in self.sources):
            raise ValueError("Expression state axes must fit the vector field")
        for _, site in self.parameter_sites(prefix):
            yield site

    def sample_params(self, prefix: str, prior_fn: PriorFn) -> dict[str, Array]:
        return {
            identity: jnp.asarray(numpyro.sample(site.name, prior_fn(site.name)))
            for identity, site in self.parameter_sites(prefix)
        }

    def pack_params(self, prefix: str, samples: Mapping[str, Array]) -> dict[str, Array]:
        return {
            identity: jnp.asarray(samples[site.name])
            for identity, site in self.parameter_sites(prefix)
        }


# Observation operand arithmetic over (predictor, observation scale, gathered parameter values).
type OperandEvaluator = Callable[[Array, Array, tuple[Array, ...]], Array]


class BoundExpression(eqx.Module):
    """Resolved observation arithmetic and parsed link; parameter arrays are pytree leaves."""

    expression: Expression = eqx.field(static=True)
    evaluate_fn: OperandEvaluator = eqx.field(static=True)
    coordinates: tuple[ParameterCoordinate, ...] = eqx.field(static=True)
    values: tuple[Array, ...]
    event_size: int = eqx.field(static=True)
    sampling_size: int = eqx.field(static=True)
    response_fn: Callable[[Array], Array] = eqx.field(static=True)
    link: LinkFunction = eqx.field(static=True)

    def bind(self, samples: Mapping[str, Array]) -> BoundExpression:
        values = tuple(
            jnp.asarray(samples[coordinate.site_name])[coordinate.indices]
            for coordinate in self.coordinates
        )
        return BoundExpression(
            self.expression,
            self.evaluate_fn,
            self.coordinates,
            values,
            self.event_size,
            self.sampling_size,
            self.response_fn,
            self.link,
        )

    def evaluate(self, predictor: Array, scale: Array, observed: Array | None = None) -> Array:
        values = self.values
        if observed is not None:
            # Missing channels never differentiate undefined expression arithmetic.
            predictor = jnp.where(observed, predictor, 1.0)
            scale = jnp.where(observed, scale, 1.0)
            values = tuple(jnp.where(observed, value, 1.0) for value in values)
        return self.evaluate_fn(predictor, scale, values)
