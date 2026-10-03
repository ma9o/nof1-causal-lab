"""Ownership of probability laws that holds before any compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.model_structure import StructuralSelection


def validate_distribution_memberships(model: ModelSpec) -> None:
    """A scalar law belongs to one parameter; joint laws are checked by the selection."""
    from nof1_causal_lab.numpyro_json import distribution_shape

    for identity, law in model.distributions.items():
        if distribution_shape(law) != ((), ()):
            continue
        parameters = [
            parameter.id for parameter in model.parameters if parameter.distribution == identity
        ]
        if len(parameters) != 1 or any(
            construct.distribution == identity for construct in model.constructs
        ):
            raise ValueError("A scalar distribution must belong to exactly one parameter")


def validate_joint_laws(selection: StructuralSelection) -> None:
    """Each joint law's identity and width follow the coordinates this scope compiles."""
    from nof1_causal_lab.models.model_structure import StructuralSelectionError
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel, compile_model
    from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
    from nof1_causal_lab.numpyro_json import distribution_shape

    model = selection.model
    joint = {
        identity: law
        for identity, law in model.distributions.items()
        if distribution_shape(law) != ((), ())
    }
    if not joint:
        return
    compiled = compile_model(selection)
    if not isinstance(compiled, CompiledModel):
        raise StructuralSelectionError(
            f"A fitted joint law requires an executable model: {compiled.message}"
        )
    bindings, _ = parameter_bindings(compiled)
    for identity, law in joint.items():
        layout = JointLawLayout.from_bindings(
            bindings,
            parameters=[
                parameter.id for parameter in model.parameters if parameter.distribution == identity
            ],
            constructs=[
                construct.id for construct in model.constructs if construct.distribution == identity
            ],
            time_points=model.time_points,
        )
        expected = layout.distribution_id
        if identity != expected:
            raise StructuralSelectionError(
                f"Joint distribution identity does not match its scientific event coordinates: expected {expected}"
            )
        if distribution_shape(law) != ((), (layout.width,)):
            raise StructuralSelectionError(
                "A joint distribution must have one event coordinate per scientific quantity"
            )
