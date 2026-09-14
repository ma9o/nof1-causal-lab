"""Bind scalar expressions to exact JAX vector-field components."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro

from nof1_causal_lab.artifacts.expressions import (
    CoefficientExpression,
    Expression,
    coefficient_key,
    expression_coefficients,
    expression_states,
    fold_expression,
)
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.ssm.structure.sites import SemanticBinding, make_site

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jax import Array

    from nof1_causal_lab.artifacts.identity import ConstructId, ParameterId
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor

    from .spec import PriorFn


SCALAR_OPERATIONS = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "power": operator.pow,
    "maximum": jnp.maximum,
}


def apply_expression_function(name, arguments):
    """Interpret scalar functions; discrete contrasts require observation context."""
    match name:
        case "exp":
            return jnp.exp(arguments[0])
        case "sigmoid":
            return jax.nn.sigmoid(arguments[0])
        case "normal_cdf":
            return jax.scipy.special.ndtr(arguments[0])
        case _:
            raise ValueError(f"{name} requires observation category metadata")


class ExpressionComponent(eqx.Module):
    """One exact scalar contribution, with IDs bound once before numerical execution."""

    target: int = eqx.field(static=True)
    edge_owned: bool = eqx.field(static=True)
    expression: Expression = eqx.field(static=True)
    state_ids: tuple[ConstructId, ...] = eqx.field(static=True)

    def evaluate(self, values, params):
        def coefficient(operand):
            reference = operand.value
            if reference is None:
                from nof1_causal_lab.compilation_errors import IncompleteModelError

                raise IncompleteModelError(f"Expression requires its {operand.role} coefficient")
            if isinstance(reference, (int, float)):
                return jnp.asarray(reference)
            return params[reference]

        return fold_expression(
            self.expression,
            literal=jnp.asarray,
            state_value=lambda identity: values[self.state_ids.index(identity)],
            coefficient_value=coefficient,
            binary=lambda name, left, right: SCALAR_OPERATIONS[name](left, right),
            call=apply_expression_function,
        )

    def contribute(self, accumulator, eta, eta_per_edge, _t, params):
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

    def __post_init__(self):
        if self.kind == "potential" and (self.source is not None or self.sources - {self.target}):
            raise ValueError("A node potential may depend only on its owning state")

    @property
    def edge_owned(self) -> bool:
        return self.source is not None

    @property
    def sources(self) -> frozenset[int]:
        return frozenset(self.state_ids.index(key) for key in expression_states(self.expression))

    @property
    def parameters(self) -> tuple[CoefficientExpression, ...]:
        return tuple(
            operand
            for operand in expression_coefficients(self.expression)
            if isinstance(operand.value, str)
        )

    def build(self) -> ExpressionComponent:
        return ExpressionComponent(
            target=self.target,
            edge_owned=self.edge_owned,
            expression=self.expression,
            state_ids=self.state_ids,
        )

    def parameter_sites(self, prefix: str) -> Iterator[tuple[ParameterId, SiteDescriptor]]:
        positions = (
            ((self.target, self.source, *sorted(self.sources - {self.source})),)
            if self.source is not None
            else (self.target,)
        )
        for index, operand in enumerate(self.parameters):
            meaning = operand.meaning
            prior_field = meaning.quantity.value
            if meaning.quantity == SiteKind.DYNAMICS_WEIGHT:
                prior_field = (
                    "multiplicative_weight" if len(self.sources) > 1 else "linear_edge_weight"
                )
            yield (
                coefficient_key(operand),
                make_site(
                    f"{prefix}_p{index}",
                    (),
                    meaning.support,
                    "dynamics",
                    meaning.quantity,
                    positions=positions,
                    priors_field=prior_field,
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

    def iter_semantic_bindings(
        self, prefix: str, *, latent_names: tuple[str, ...], component_index: int
    ):
        for identity, site in self.parameter_sites(prefix):
            yield SemanticBinding(
                parameter_name=identity,
                site_name=site.name,
                flat_index=0,
                site_kind=site.site_kind,
                prior_field=site.priors_field,
                construct_names=tuple(
                    latent_names[i] for i in sorted(self.sources | {self.target})
                ),
                component_index=component_index,
                effect_idx=self.target if self.edge_owned else None,
                cause_idx=self.source,
            )

    def pack_params(self, prefix: str, samples: dict[str, Array]) -> dict[str, Array]:
        return {
            identity: jnp.asarray(samples[site.name])
            for identity, site in self.parameter_sites(prefix)
        }
