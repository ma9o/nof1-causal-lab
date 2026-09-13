"""Lower scientific mechanisms to native components using persistent graph IDs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.mechanism import (
    ConstantDriftMechanism,
    EstimatedCoefficient,
    FixedCoefficient,
    HillEdgeMechanism,
    LinearEdgeMechanism,
    NodePotentialMechanism,
    mechanism_coefficients,
)
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.ssm.dynamics.spec import (
    DynamicsSpec,
    HillEdgeSpec,
    LinearEdgeSpec,
    NodePotentialSpec,
    StateInterceptSpec,
)
from nof1_causal_lab.models.ssm.structure.parameters import Fixed, Free

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.mechanism import DynamicsMechanism, MechanismCoefficient
    from nof1_causal_lab.artifacts.statistical_model_spec import StatisticalModelSpec
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.models.ssm.dynamics.spec import ComponentSpec
    from nof1_causal_lab.models.ssm.structure.parameters import ParameterSlot


def native_coefficient(coefficient: MechanismCoefficient) -> ParameterSlot:
    return Fixed(coefficient.value) if isinstance(coefficient, FixedCoefficient) else Free()


def coefficient_quantities(
    mechanism: DynamicsMechanism, plan: StructuralPlan
) -> dict[str, SiteKind]:
    match mechanism:
        case NodePotentialMechanism():
            return {
                "center": SiteKind.DYNAMICS_POTENTIAL_CENTER,
                "stiffness": SiteKind.DYNAMICS_DECAY,
                "quartic": SiteKind.DYNAMICS_POTENTIAL_QUARTIC,
            }
        case ConstantDriftMechanism():
            return {"intercept": SiteKind.DYNAMICS_CINT}
        case LinearEdgeMechanism():
            edge = plan.semantics.edges[mechanism.edge_id]
            known_input = edge.cause_id in {item.construct_id for item in plan.known_inputs}
            return {"weight": SiteKind.INPUT_EFFECT if known_input else SiteKind.DYNAMICS_WEIGHT}
        case HillEdgeMechanism():
            return {"emax": SiteKind.HILL_EMAX, "ec50": SiteKind.HILL_EC50, "n": SiteKind.HILL_N}


def lower_mechanisms(model: StatisticalModelSpec, plan: StructuralPlan) -> DynamicsSpec:
    """Resolve graph IDs once; the native engine receives only indices and fixed/free slots.

    Bindings are keyed by scientific parameter ID. Component aliases remain
    runtime descriptions and never select a mechanism or its scientific owner.
    """
    state_index = {key: index for index, key in enumerate(plan.state_order)}
    retained_edges = {edge.source_id: edge for edge in plan.edges}
    definitions = {parameter.id: parameter for parameter in model.parameters}
    components: list[ComponentSpec] = []
    bound_parameters: set[str] = set()
    modeled_edges: set[str] = set()
    modeled_nodes: set[str] = set()

    for mechanism in model.mechanisms:
        if isinstance(mechanism, (NodePotentialMechanism, ConstantDriftMechanism)):
            owners = {mechanism.target_id}
        else:
            if mechanism.edge_id not in retained_edges:
                raise ValueError(
                    f"Mechanism references unknown retained edge {mechanism.edge_id!r}"
                )
            edge = retained_edges[mechanism.edge_id]
            owners = {edge.source_id, edge.cause_id, edge.effect_id}
        quantities = coefficient_quantities(mechanism, plan)
        for slot, coefficient in mechanism_coefficients(mechanism).items():
            if not isinstance(coefficient, EstimatedCoefficient):
                continue
            definition = definitions[coefficient.parameter_id]
            quantity = quantities[slot]
            if definition.quantity != quantity:
                raise ValueError(
                    f"Mechanism coefficient {slot!r} requires {quantity.value}, "
                    f"but {definition.name!r} declares {definition.quantity.value}"
                )
            if not owners <= {owner.id for owner in definition.owners}:
                raise ValueError(
                    f"Parameter {definition.name!r} owners disagree with its mechanism"
                )
            if definition.id in bound_parameters:
                raise ValueError("One parameter cannot own multiple independent runtime sites")
            bound_parameters.add(definition.id)

        if isinstance(mechanism, (NodePotentialMechanism, ConstantDriftMechanism)):
            if mechanism.target_id not in state_index:
                raise ValueError(
                    f"Mechanism references unknown retained state {mechanism.target_id!r}"
                )
            target = state_index[mechanism.target_id]
            modeled_nodes.add(mechanism.target_id)
            if plan.semantics.constructs[mechanism.target_id].temporal_status == "time_invariant":
                raise ValueError("Time-invariant states cannot have drift mechanisms")
            if isinstance(mechanism, NodePotentialMechanism):
                component = NodePotentialSpec(
                    target=target,
                    center=native_coefficient(mechanism.center),
                    stiffness=native_coefficient(mechanism.stiffness),
                    quartic=native_coefficient(mechanism.quartic),
                )
            else:
                component = StateInterceptSpec(target=target)
        else:
            edge = retained_edges[mechanism.edge_id]
            modeled_edges.add(mechanism.edge_id)
            if edge.cause_id not in state_index:
                if not isinstance(mechanism, LinearEdgeMechanism):
                    raise ValueError("Known inputs currently support linear mechanisms")
                # Known-input effects lower to the existing input matrix block.
                continue
            source, target = state_index[edge.cause_id], state_index[edge.effect_id]
            if isinstance(mechanism, LinearEdgeMechanism):
                component = LinearEdgeSpec(source=source, target=target)
            else:
                component = HillEdgeSpec(
                    source=source,
                    target=target,
                    emax=native_coefficient(mechanism.emax),
                    ec50=native_coefficient(mechanism.ec50),
                    n=native_coefficient(mechanism.n),
                )

        components.append(component)

    expected_nodes = {
        key
        for key in plan.state_order
        if plan.semantics.constructs[key].temporal_status != "time_invariant"
    }
    if modeled_nodes != expected_nodes or modeled_edges != set(retained_edges):
        raise ValueError(
            "Mechanisms must cover the retained dynamic states and edges: "
            f"missing states={sorted(expected_nodes - modeled_nodes)}, "
            f"missing edges={sorted(set(retained_edges) - modeled_edges)}"
        )
    return DynamicsSpec(n_latent=len(state_index), components=tuple(components))
