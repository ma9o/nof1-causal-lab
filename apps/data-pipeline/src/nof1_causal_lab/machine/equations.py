"""Equation projection of the same scientific mechanisms consumed by compilation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.mechanism import (
    ConstantDriftMechanism,
    FixedCoefficient,
    HillEdgeMechanism,
    LinearEdgeMechanism,
    NodePotentialMechanism,
)
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform
from nof1_causal_lab.machine.view_models import StateEquation

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.mechanism import MechanismCoefficient
    from nof1_causal_lab.artifacts.statistical_model_spec import StatisticalModelSpec
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan


def _text(label: str) -> str:
    escapes = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "_": " ",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return r"\text{" + "".join(escapes.get(character, character) for character in label) + "}"


def state_equations(model: StatisticalModelSpec, plan: StructuralPlan) -> list[StateEquation]:
    """Render actual continuous-time drift, including fixed and nonlinear coefficients.

    Symbols are parameter labels, so the authored-prior table remains the
    legend. A prior authored as persistence or interval effect is explicitly
    converted in the displayed drift, just as it is during compilation.
    """
    parameters = {parameter.id: parameter for parameter in model.parameters}
    terms: dict[str, list[str]] = defaultdict(list)

    def state(key: str, *, known_input: bool = False) -> str:
        symbol = "u" if known_input else r"\eta"
        return symbol + "_{" + _text(plan.semantics.constructs[key].name) + "}(t)"

    def coefficient(value: MechanismCoefficient) -> str:
        if isinstance(value, FixedCoefficient):
            return f"{value.value:g}"
        parameter = parameters[value.parameter_id]
        symbol = r"\theta_{" + _text(parameter.name) + "}"
        match parameter.prior_transform:
            case PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY:
                return r"\frac{-\log(" + symbol + r")}{\Delta_{" + _text(parameter.name) + "}}"
            case PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE:
                return r"\frac{" + symbol + r"}{\Delta_{" + _text(parameter.name) + "}}"
            case _:
                return symbol

    def zero(value: MechanismCoefficient) -> bool:
        return isinstance(value, FixedCoefficient) and value.value == 0

    inputs = {item.construct_id for item in plan.known_inputs}
    for mechanism in model.mechanisms:
        match mechanism:
            case NodePotentialMechanism():
                target = mechanism.target_id
                delta = (
                    state(target)
                    if zero(mechanism.center)
                    else (
                        r"\left("
                        + state(target)
                        + " - "
                        + coefficient(mechanism.center)
                        + r"\right)"
                    )
                )
                terms[target].append("-" + coefficient(mechanism.stiffness) + r"\," + delta)
                if not zero(mechanism.quartic):
                    terms[target].append(
                        "-" + coefficient(mechanism.quartic) + r"\," + delta + "^{3}"
                    )
            case ConstantDriftMechanism():
                terms[mechanism.target_id].append(coefficient(mechanism.intercept))
            case LinearEdgeMechanism() | HillEdgeMechanism():
                edge = plan.semantics.edges[mechanism.edge_id]
                source = state(edge.cause_id, known_input=edge.cause_id in inputs)
                if isinstance(mechanism, LinearEdgeMechanism):
                    term = coefficient(mechanism.weight) + r"\," + source
                else:
                    power = coefficient(mechanism.n)
                    dose = r"\max\!\left(" + source + r",0\right)^{" + power + "}"
                    term = (
                        coefficient(mechanism.emax)
                        + r"\,\frac{"
                        + dose
                        + "}{"
                        + (coefficient(mechanism.ec50) + "^{" + power + "} + " + dose + "}")
                    )
                terms[edge.effect_id].append(term)

    rows = []
    for key in plan.state_order:
        construct = plan.semantics.constructs[key]
        if construct.temporal_status == "time_invariant":
            equation = r"\mathrm{d}" + state(key) + " = 0"
        else:
            drift = " + ".join(terms[key]).replace(" + -", " - ")
            noise = r"\sum_j L_{" + _text(construct.name) + r",j}\,\mathrm{d}W_j(t)"
            equation = (
                r"\mathrm{d}" + state(key) + r" = \left[" + drift + r"\right]\mathrm{d}t + " + noise
            )
        rows.append(StateEquation(construct_id=key, label=construct.name, latex=equation))
    return rows
