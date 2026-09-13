"""The authoring/JSON boundary for native NumPyro prior distributions.

Only the approved family names and their authoring aliases are application
policy. NumPyro supplies argument constraints, broadcasting, densities,
sampling, moments, and transformed-distribution Jacobians.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms

from nof1_causal_lab.distributions import PriorDistributionFamily

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pydantic.json_schema import JsonSchemaValue

    from nof1_causal_lab.artifacts.distribution import CompiledDistribution


_ARGUMENT_ALIASES = {
    "loc": "mu",
    "scale": "sigma",
    "low": "lower",
    "high": "upper",
    "concentration1": "alpha",
    "concentration0": "beta",
    "v": "value",
}


def prior_argument_constraints(
    family: PriorDistributionFamily,
) -> dict[str, constraints.Constraint]:
    """Native argument constraints for the approved authoring surface."""
    if family == PriorDistributionFamily.TRUNCATED_NORMAL:
        return {
            **dist.Normal.arg_constraints,
            "low": constraints.real,
            "high": constraints.real,
        }
    if family == PriorDistributionFamily.DELTA:
        return {"v": dist.Delta.arg_constraints["v"]}
    return dict(getattr(dist, family.value).arg_constraints)


def authored_argument_names(family: PriorDistributionFamily) -> dict[str, str]:
    """Map NumPyro constructor names to the project's authoring vocabulary."""
    return {
        native: _ARGUMENT_ALIASES.get(native, native)
        for native in prior_argument_constraints(family)
    }


def distribution_schema_variants() -> list[JsonSchemaValue]:
    """Export native constructor signatures as inline, family-tagged JSON objects."""
    variants = []
    for family in PriorDistributionFamily:
        aliases = authored_argument_names(family)
        properties = {}
        for native, constraint in prior_argument_constraints(family).items():
            if constraint is constraints.positive:
                field = {"type": "number", "exclusiveMinimum": 0}
            elif constraint is constraints.real or constraints.is_dependent(constraint):
                field = {"type": "number"}
            else:
                raise ValueError(f"Unsupported JSON constraint for {family.value}.{native}")
            properties[aliases[native]] = field
        variants.append(
            {
                "type": "object",
                "properties": {
                    "distribution": {"const": family.value, "type": "string"},
                    "params": {
                        "type": "object",
                        "properties": properties,
                        "required": list(properties),
                        "additionalProperties": False,
                    },
                },
                "required": ["distribution", "params"],
            }
        )
    return variants


def distribution_from_params(
    family: PriorDistributionFamily | str,
    params: Mapping[str, Any],
) -> dist.Distribution:
    """Validate a complete authored parameter mapping and construct its law."""
    family = PriorDistributionFamily(family)
    arguments = authored_argument_names(family)
    expected = set(arguments.values())
    if set(params) != expected:
        raise ValueError(
            f"{family.value} requires exactly {sorted(expected)}; got {sorted(params)}"
        )
    native = {
        key: jnp.asarray(params[authored], dtype=jnp.float32) for key, authored in arguments.items()
    }
    if family in {
        PriorDistributionFamily.UNIFORM,
        PriorDistributionFamily.TRUNCATED_NORMAL,
    } and np.any(np.asarray(native["low"]) >= np.asarray(native["high"])):
        raise ValueError(f"{family.value} requires lower < upper")
    return getattr(dist, family.value)(**native, validate_args=True)


def distribution_support_bounds(distribution: dist.Distribution) -> tuple[Any, Any]:
    """Resolve scalar interval constraints, including transformed supports."""
    support = distribution.support
    if support is constraints.real:
        return -np.inf, np.inf
    if isinstance(support, type(constraints.interval(0.0, 1.0))):
        return support.lower_bound, support.upper_bound
    if isinstance(support, type(constraints.greater_than(0.0))):
        return support.lower_bound, np.inf
    if isinstance(support, type(constraints.less_than(0.0))):
        return -np.inf, support.upper_bound
    raise ValueError(f"Unsupported scalar prior support {support!r}")


def persistence_to_decay(distribution: dist.Distribution, interval_days: Any) -> dist.Distribution:
    """Push the authored persistence law through decay = -log(rho) / dt."""
    low, high = distribution_support_bounds(distribution)
    if np.any(np.asarray(low) < 0.0) or np.any(np.asarray(high) > 1.0):
        raise ValueError(
            "Persistence priors must have support within [0, 1]; author a Beta, "
            "Uniform, or TruncatedNormal with valid bounds."
        )
    interval = jnp.asarray(interval_days, dtype=jnp.float32)
    if np.any(np.asarray(interval) <= 0.0):
        raise ValueError("The prior reference interval must be positive")
    # Declare the forward map from positive decay to unit-interval persistence,
    # then invert it. Its inverse codomain remains positive under JAX tracing;
    # it does not depend on AffineTransform inferring the sign of a tracer.
    forward = transforms.ComposeTransform(
        [
            transforms.AffineTransform(0.0, -interval, domain=constraints.positive),
            transforms.ExpTransform(domain=constraints.interval(-jnp.inf, 0.0)),
        ]
    )
    return dist.TransformedDistribution(distribution, forward.inv)


def interval_effect_to_rate(
    distribution: dist.Distribution, interval_days: Any
) -> dist.Distribution:
    """Rescale the entire authored effect law, preserving its family and bounds."""
    interval = jnp.asarray(interval_days, dtype=jnp.float32)
    if np.any(np.asarray(interval) <= 0.0):
        raise ValueError("The prior reference interval must be positive")
    return dist.TransformedDistribution(
        distribution, transforms.AffineTransform(0.0, 1.0 / interval, domain=distribution.support)
    )


def prior_reference_value(distribution: dist.Distribution) -> jax.Array:
    """A diagnostic anchor: the base-law mean passed through declared transforms.

    For nonlinear transforms this is explicitly not the transformed law's mean.
    It never defines a density, a sampler, or a reported posterior moment.
    """
    if isinstance(distribution, (dist.ExpandedDistribution, dist.MaskedDistribution)):
        return jnp.broadcast_to(
            prior_reference_value(distribution.base_dist), distribution.batch_shape
        )
    if isinstance(distribution, dist.MixtureGeneral):
        return jnp.stack(
            [
                jnp.broadcast_to(
                    prior_reference_value(component), distribution.batch_shape
                ).reshape(-1)[index]
                for index, component in enumerate(distribution.component_distributions)
            ]
        ).reshape(distribution.batch_shape)
    if type(distribution) is dist.TransformedDistribution:
        value = prior_reference_value(distribution.base_dist)
        for transform in distribution.transforms:
            value = transform(value)
        return value
    return jnp.asarray(distribution.mean)


def _base_distribution_parts(distribution: dist.Distribution):
    if isinstance(distribution, dist.TwoSidedTruncatedDistribution):
        if type(distribution.base_dist) is not dist.Normal:
            raise ValueError("Only truncated Normal priors are supported")
        return PriorDistributionFamily.TRUNCATED_NORMAL, {
            "mu": distribution.base_dist.loc,
            "sigma": distribution.base_dist.scale,
            "lower": distribution.low,
            "upper": distribution.high,
        }
    family = PriorDistributionFamily(type(distribution).__name__)
    return family, {
        authored: getattr(distribution, native)
        for native, authored in authored_argument_names(family).items()
    }


def serialize_distribution(distribution: dist.Distribution) -> list[CompiledDistribution]:
    """Serialize scalar coordinates of an approved native distribution losslessly."""
    from nof1_causal_lab.artifacts.distribution import CompiledDistribution, DistributionTransform

    shape = distribution.batch_shape
    count = int(np.prod(shape))
    if isinstance(distribution, dist.MixtureGeneral):
        expected = np.eye(count).reshape((*shape, count))
        if not np.array_equal(np.asarray(distribution.mixing_distribution.probs), expected):
            raise ValueError("Only coordinate-selected prior products can be serialized")
        return [
            serialize_distribution(component)[index]
            for index, component in enumerate(distribution.component_distributions)
        ]

    base = distribution
    declared_transforms = []
    while (
        isinstance(base, (dist.ExpandedDistribution, dist.MaskedDistribution))
        or type(base) is dist.TransformedDistribution
    ):
        if type(base) is dist.TransformedDistribution:
            declared_transforms = [*base.transforms, *declared_transforms]
        base = base.base_dist
    family, parameters = _base_distribution_parts(base)

    def scalar(value, index):
        return float(np.broadcast_to(np.asarray(value), shape).reshape(-1)[index])

    recipes = []
    for index in range(count):
        serialized_transforms = []
        for transform in declared_transforms:
            if isinstance(transform, transforms.AffineTransform):
                serialized_transforms.append(
                    DistributionTransform(
                        kind="affine",
                        loc=scalar(transform.loc, index),
                        scale=scalar(transform.scale, index),
                    )
                )
            elif isinstance(transform, transforms.ExpTransform):
                serialized_transforms.append(DistributionTransform(kind="exp"))
            elif isinstance(transform.inv, transforms.ComposeTransform):
                parts = transform.inv.parts
                if (
                    len(parts) != 2
                    or not isinstance(parts[0], transforms.AffineTransform)
                    or not isinstance(parts[1], transforms.ExpTransform)
                ):
                    raise ValueError("Unsupported inverse-composite prior transform")
                if (
                    np.any(np.asarray(parts[0].loc) != 0.0)
                    or not isinstance(parts[0].domain, type(constraints.positive))
                    or np.any(np.asarray(parts[0].domain.lower_bound) != 0.0)
                ):
                    raise ValueError(
                        "A persistence transform must map positive decay to persistence"
                    )
                serialized_transforms.append(
                    DistributionTransform(
                        kind="persistence_to_decay", scale=-scalar(parts[0].scale, index)
                    )
                )
            else:
                raise ValueError(f"Unsupported compiled prior transform {transform!r}")
        recipes.append(
            CompiledDistribution(
                distribution=family,
                params={key: scalar(value, index) for key, value in parameters.items()},
                transforms=serialized_transforms,
            )
        )
    return recipes


def deserialize_distribution(recipe: CompiledDistribution) -> dist.Distribution:
    """Hydrate a scalar recipe using NumPyro's own distribution operations."""
    base = recipe.to_numpyro()
    for operation in recipe.transforms:
        base = _apply_distribution_transform(base, operation.kind, operation.loc, operation.scale)
    return base


def _apply_distribution_transform(distribution, kind, loc, scale):
    if kind == "persistence_to_decay":
        return persistence_to_decay(distribution, scale)
    transform = (
        transforms.ExpTransform()
        if kind == "exp"
        else transforms.AffineTransform(
            jnp.asarray(loc), jnp.asarray(scale), domain=distribution.support
        )
    )
    return dist.TransformedDistribution(distribution, transform)


def batch_prior_distributions(
    distributions: Sequence[dist.Distribution],
    shape: tuple[int, ...],
    *,
    support: constraints.Constraint,
) -> dist.Distribution:
    """Assemble independent coordinate laws with NumPyro/JAX composition.

    Equal distribution recipes batch their native constructor arguments. Different
    families use a deterministic categorical selector per coordinate; this is
    a product of the prescribed laws, with no uncertain mixture membership.
    """
    count = int(np.prod(shape))
    if len(distributions) != count:
        raise ValueError(
            f"Prior shape {shape} requires {count} coordinate laws, got {len(distributions)}"
        )
    recipes = [serialize_distribution(distribution)[0] for distribution in distributions]
    signatures = [
        (recipe.distribution, tuple(operation.kind for operation in recipe.transforms))
        for recipe in recipes
    ]
    if all(signature == signatures[0] for signature in signatures):
        parameters = [recipe.params for recipe in recipes]
        prior = distribution_from_params(
            recipes[0].distribution,
            {
                key: jnp.asarray([params[key] for params in parameters]).reshape(shape)
                for key in parameters[0]
            },
        )
        for index, operation in enumerate(recipes[0].transforms):
            loc = jnp.asarray([recipe.transforms[index].loc for recipe in recipes]).reshape(shape)
            scale = jnp.asarray([recipe.transforms[index].scale for recipe in recipes]).reshape(
                shape
            )
            prior = _apply_distribution_transform(prior, operation.kind, loc, scale)
        return prior
    selector = dist.Categorical(
        logits=jnp.where(jnp.eye(count).reshape((*shape, count)), 0.0, -jnp.inf)
    )
    # Inactive coordinate laws must not evaluate values outside their domains:
    # e.g. a real coordinate may be negative while another is LogNormal.
    # Native masking supplies support-interior values before differentiation.
    components = [
        distribution.mask(jnp.arange(count).reshape(shape) == index)
        for index, distribution in enumerate(distributions)
    ]
    return dist.MixtureGeneral(selector, components, support=support)
