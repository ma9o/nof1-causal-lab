"""Ownership of probability laws that holds before any compilation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def validate_distribution_memberships(model: ModelSpec) -> None:
    """Own scalar membership and retained joint identities before compilation."""
    from nof1_causal_lab.numpyro_json import distribution_shape

    joint = {
        identity
        for identity, law in model.distributions.items()
        if distribution_shape(law) != ((), ())
    }
    if set(model.law_layouts) != joint:
        raise ValueError("Each joint law requires exactly one retained scientific layout")
    grids = {layout.time_points for layout in model.law_layouts.values() if layout.constructs}
    if len(grids) > 1:
        raise ValueError("Joint trajectories must share their retained time grid")
    for identity, layout in model.law_layouts.items():
        parameters = tuple(
            sorted(
                parameter.id for parameter in model.parameters if parameter.distribution == identity
            )
        )
        constructs = tuple(
            sorted(
                construct.id for construct in model.constructs if construct.distribution == identity
            )
        )
        if (
            tuple(parameter for parameter, _ in layout.parameters) != parameters
            or layout.constructs != constructs
        ):
            raise ValueError("Joint layout membership must match its scientific quantities")
        if identity != layout.distribution_id or distribution_shape(
            model.distributions[identity]
        ) != ((), (layout.width,)):
            raise ValueError(
                "Joint law identity and event width must match its retained coordinates"
            )
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
