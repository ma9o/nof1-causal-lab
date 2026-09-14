"""Scientific event identities for shared probability laws."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.identity import scientific_id

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from nof1_causal_lab.artifacts.identity import ConstructId, DistributionId, ParameterId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


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

    bindings = {binding.parameter_id: binding for binding in parameter_bindings(model)[0]}
    for identity in model.distributions:
        parameters = [
            parameter.id for parameter in model.parameters if parameter.distribution == identity
        ]
        constructs = [
            construct.id for construct in model.constructs if construct.distribution == identity
        ]
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
        if distribution_shape(model.distributions[identity]) != ((), (size,)):
            raise ValueError(
                "A joint distribution must have one event coordinate per scientific quantity"
            )
