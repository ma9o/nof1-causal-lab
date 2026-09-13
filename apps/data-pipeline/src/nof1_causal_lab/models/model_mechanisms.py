"""Construct the supported scientific dynamics from explicit authoring choices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.mechanism import (
    EstimatedCoefficient,
    FixedCoefficient,
    HillEdgeMechanism,
    LinearEdgeMechanism,
    NodePotentialMechanism,
)
from nof1_causal_lab.artifacts.parameter import SiteKind

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from nof1_causal_lab.artifacts.mechanism import DynamicsMechanism
    from nof1_causal_lab.artifacts.statistical_model_spec import ParameterSpec
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan


def declare_dynamics_mechanisms(
    plan: StructuralPlan,
    parameters: Sequence[ParameterSpec],
    *,
    self_limiting: Collection[str] = (),
    hill_edges: Collection[str] = (),
    centered_states: Collection[str] = (),
) -> list[DynamicsMechanism]:
    """Materialize explicit linear/Hill and potential-well choices at authoring time.

    These are authoring defaults and choices, never compiler inference from a
    prior or parameter label. Fixed/free refinements can edit the resulting
    coefficient slots directly.
    """

    def estimated(quantity: SiteKind, owner_id: str) -> EstimatedCoefficient:
        matches = [
            parameter
            for parameter in parameters
            if parameter.quantity == quantity
            and any(owner.id == owner_id for owner in parameter.owners)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {quantity.value} parameter for {owner_id!r}; found {len(matches)}"
            )
        return EstimatedCoefficient(parameter_id=matches[0].id)

    states = set(plan.state_order)
    edge_ids = {edge.source_id for edge in plan.edges}
    if not set(self_limiting) <= states or not set(centered_states) <= states:
        raise ValueError("Self-dynamics choices must reference retained states")
    if not set(hill_edges) <= edge_ids:
        raise ValueError("Hill choices must reference retained edges")
    mechanisms: list[DynamicsMechanism] = []
    for key in plan.state_order:
        if plan.semantics.constructs[key].temporal_status == "time_invariant":
            continue
        mechanisms.append(
            NodePotentialMechanism(
                target_id=key,
                center=estimated(SiteKind.DYNAMICS_POTENTIAL_CENTER, key)
                if key in centered_states
                else FixedCoefficient(value=0),
                stiffness=estimated(SiteKind.DYNAMICS_DECAY, key),
                quartic=estimated(SiteKind.DYNAMICS_POTENTIAL_QUARTIC, key)
                if key in self_limiting
                else FixedCoefficient(value=0),
            )
        )
    order = {key: index for index, key in enumerate(plan.state_order)}
    inputs = {item.construct_id for item in plan.known_inputs}
    for edge in sorted(plan.edges, key=lambda e: (order[e.effect_id], order.get(e.cause_id, -1))):
        key = edge.source_id
        if key in hill_edges:
            if edge.cause_id in inputs:
                raise ValueError("Known inputs currently support linear mechanisms")
            mechanisms.append(
                HillEdgeMechanism(
                    edge_id=key,
                    emax=estimated(SiteKind.HILL_EMAX, key),
                    ec50=estimated(SiteKind.HILL_EC50, key),
                    n=estimated(SiteKind.HILL_N, key),
                )
            )
        else:
            mechanisms.append(
                LinearEdgeMechanism(
                    edge_id=key,
                    weight=estimated(
                        SiteKind.INPUT_EFFECT
                        if edge.cause_id in inputs
                        else SiteKind.DYNAMICS_WEIGHT,
                        key,
                    ),
                )
            )
    return mechanisms
