"""Ownership, membership, and scientific event identities for probability laws."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.identity import scientific_id

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from nof1_causal_lab.artifacts.identity import ConstructId, DistributionId, ParameterId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.numpyro_json import NumPyroDistribution


def parameter_distribution_id(parameter_id: ParameterId) -> DistributionId:
    """Name a parameter's individual law independently of its current constructor."""
    return scientific_id("distribution", ["parameter", parameter_id])


def with_parameter_distributions(
    model: ModelSpec, laws: Mapping[ParameterId, NumPyroDistribution]
) -> ModelSpec:
    """Replace individual parameter laws and their memberships in one validated revision."""
    from nof1_causal_lab.numpyro_json import distribution_shape

    identities = {}
    for identity in laws:
        parameter = model.parameter(identity)
        identities[identity] = (
            parameter.distribution
            if parameter.distribution is not None
            and distribution_shape(model.distributions[parameter.distribution]) == ((), ())
            else parameter_distribution_id(identity)
        )
    parameters = tuple(
        parameter.model_copy(update={"distribution": identities[parameter.id]})
        if parameter.id in laws
        else parameter
        for parameter in model.parameters
    )
    referenced = {item.distribution for item in (*parameters, *model.constructs)}
    distributions = {
        identity: law for identity, law in model.distributions.items() if identity in referenced
    }
    distributions.update((identities[identity], law) for identity, law in laws.items())
    return model.revised(parameters=parameters, distributions=distributions)


def joint_distribution_id(
    model: ModelSpec,
    parameters: Collection[ParameterId],
    constructs: Collection[ConstructId],
    time_points: Sequence[float],
) -> DistributionId:
    """Identify a law's event coordinates without retaining execution-array mappings.

    Element identity includes scientific categorical/covariance bases. Changing
    a basis must not silently reinterpret an unchanged probability distribution.
    """
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    bindings, _ = parameter_bindings(model)
    by_id = {binding.parameter_id: binding for binding in bindings}
    members = [[identity, sorted(by_id[identity].coordinates)] for identity in sorted(parameters)]
    trajectories = [
        [identity, [float(value) for value in time_points]] for identity in sorted(constructs)
    ]
    return scientific_id("distribution", [members, trajectories])


def validate_distribution_memberships(model: ModelSpec) -> None:
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.numpyro_json import distribution_shape

    for identity in model.distributions:
        parameters = [
            parameter.id for parameter in model.parameters if parameter.distribution == identity
        ]
        constructs = [
            construct.id for construct in model.constructs if construct.distribution == identity
        ]
        shape = distribution_shape(model.distributions[identity])
        if shape == ((), ()):
            if len(parameters) != 1 or constructs:
                raise ValueError("A scalar distribution must belong to exactly one parameter")
            continue
        bindings = {binding.parameter_id: binding for binding in parameter_bindings(model)[0]}
        expected = joint_distribution_id(
            model,
            parameters,
            constructs,
            model.time_points,
        )
        if identity != expected:
            raise ValueError(
                f"Joint distribution identity does not match its scientific event coordinates: expected {expected}"
            )
        size = sum(len(bindings[key].coordinates) for key in parameters) + len(constructs) * len(
            model.time_points
        )
        if shape != ((), (size,)):
            raise ValueError(
                "A joint distribution must have one event coordinate per scientific quantity"
            )
