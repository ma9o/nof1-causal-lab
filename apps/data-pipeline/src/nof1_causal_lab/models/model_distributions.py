"""Ownership, membership, and scientific event identities for probability laws."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def validate_distribution_memberships(model: ModelSpec) -> None:
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
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
        bindings, _ = parameter_bindings(model)
        layout = JointLawLayout.from_bindings(
            bindings,
            parameters=parameters,
            constructs=constructs,
            time_points=model.time_points,
        )
        expected = layout.distribution_id
        if identity != expected:
            raise ValueError(
                f"Joint distribution identity does not match its scientific event coordinates: expected {expected}"
            )
        if shape != ((), (layout.width,)):
            raise ValueError(
                "A joint distribution must have one event coordinate per scientific quantity"
            )
