"""Author scientific components first; their slots declare the parameters to elicit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    hill as expr_hill,
)
from nof1_causal_lab.artifacts.expressions import (
    linear_effect,
    restoring_potential,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec

if TYPE_CHECKING:
    from collections.abc import Collection

    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def default_mechanism_id(owner_id: str, kind: str) -> str:
    """Allocate one default term; additional additive terms choose distinct persistent IDs."""
    return scientific_id("mechanism", [owner_id, kind])


def declare_dynamics(
    model: ModelSpec,
    *,
    self_limiting: Collection[str] = (),
    hill_edges: Collection[str] = (),
    centered_states: Collection[str] = (),
) -> ModelSpec:
    """Add missing dynamics using explicit choices, preserving existing components."""
    parameters = {parameter.id: parameter for parameter in model.parameters}

    def coefficient(owner_id, slot, name, transform=PriorAuthoringTransform.IDENTITY):
        identity = scientific_id("parameter", [owner_id, slot])
        parameters.setdefault(
            identity,
            ParameterSpec(
                id=identity,
                name=name,
                description=f"{slot} of {name}",
                distribution_transform=transform,
            ),
        )
        return identity

    states = set(model.state_order)
    edge_ids = {edge.id for edge in model.execution_edges}
    if (
        not set(self_limiting) <= states
        or not set(centered_states) <= states
        or not set(hill_edges) <= edge_ids
    ):
        raise ValueError("Dynamics choices must reference retained constructs and edges")
    constructs = []
    for construct in model.constructs:
        if (
            construct.id not in states
            or construct.temporal_status == "time_invariant"
            or construct.dynamics
        ):
            constructs.append(construct)
            continue
        term = default_mechanism_id(construct.id, "node_potential")
        mechanism = DynamicsMechanismSpec(
            id=term,
            kind="potential",
            expression=restoring_potential(
                construct.id,
                center=coefficient(term, "center", f"cint_{construct.name}")
                if construct.id in centered_states
                else 0,
                stiffness=coefficient(
                    term,
                    "stiffness",
                    f"rho_{construct.name}",
                    PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY,
                ),
                quartic=coefficient(term, "quartic", f"self_limit_{construct.name}")
                if construct.id in self_limiting
                else 0,
            ),
        )
        constructs.append(construct.model_copy(update={"dynamics": (mechanism,)}))
    edges = []
    for edge in model.edges:
        if edge.id not in edge_ids or edge.mechanisms:
            edges.append(edge)
            continue
        cause, effect = edge.cause, edge.effect
        if edge.id in hill_edges:
            term = default_mechanism_id(edge.id, "hill")
            mechanism = DynamicsMechanismSpec(
                id=term,
                expression=expr_hill(
                    expr_state(edge.cause.id),
                    **{
                        slot: coefficient(term, slot, f"hill_{slot}_{cause.name}_{effect.name}")
                        for slot in ("emax", "ec50", "n")
                    },
                ),
            )
        else:
            term = default_mechanism_id(edge.id, "linear")
            mechanism = DynamicsMechanismSpec(
                id=term,
                expression=linear_effect(
                    edge.cause.id,
                    coefficient(
                        term,
                        "weight",
                        f"beta_{cause.name}_{effect.name}",
                        PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE,
                    ),
                ),
            )
        edges.append(edge.model_copy(update={"mechanisms": (mechanism,)}))
    return model.revised(
        edges=replace_constructs(tuple(edges), tuple(constructs)),
        parameters=tuple(parameters.values()),
    )
