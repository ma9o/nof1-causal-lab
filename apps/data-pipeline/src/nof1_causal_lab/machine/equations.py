"""Equation projection of the same scientific mechanisms consumed by compilation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.construct import CausalEdgeSpec
from nof1_causal_lab.artifacts.expressions import fold_expression
from nof1_causal_lab.artifacts.identity import ParameterId
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform
from nof1_causal_lab.machine.expression_latex import (
    LatexValue,
    binary_latex,
    call_latex,
    literal_latex,
)
from nof1_causal_lab.machine.view_models import StateEquation

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.expressions import Expression
    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId, ParameterId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


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


def _state_latex(model: ModelSpec, key: ConstructId) -> str:
    symbol = r"\eta"
    return symbol + "_{" + _text(model.get_construct(key).name) + "}(t)"


def _expression_latex(model: ModelSpec, expression: Expression) -> str:
    """Interpret the scientific tree, using authored parameter labels as its legend."""
    parameters = {parameter.id: parameter for parameter in model.parameters}

    def coefficient(value: float | ParameterId) -> str:
        if isinstance(value, (int, float)):
            return f"{value:g}"
        parameter = parameters[value]
        if parameter.value is not None:
            return f"{parameter.value:g}"
        symbol = r"\theta_{" + _text(parameter.name) + "}"
        match parameter.distribution_transform:
            case PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY:
                return r"\frac{-\log(" + symbol + r")}{\Delta_{" + _text(parameter.name) + "}}"
            case PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE:
                return r"\frac{" + symbol + r"}{\Delta_{" + _text(parameter.name) + "}}"
            case _:
                return symbol

    def rendered_coefficient(operand):
        reference = operand.value
        if reference is None:
            return LatexValue(r"\underbrace{?}_{\text{" + operand.role.replace("_", " ") + "}}")
        if isinstance(reference, (int, float)):
            return literal_latex(reference)
        value = parameters[reference].value
        return literal_latex(value) if value is not None else LatexValue(coefficient(reference))

    return fold_expression(
        expression,
        literal=literal_latex,
        state_value=lambda key: LatexValue(_state_latex(model, key)),
        coefficient_value=rendered_coefficient,
        binary=binary_latex,
        call=call_latex,
    ).text


def observation_equations(model: ModelSpec) -> dict[IndicatorId, str]:
    """Render each declared native conditional law, including incomplete operands."""
    return {
        indicator.id: "y_{"
        + _text(indicator.name)
        + r"}(t) \sim \operatorname{"
        + likelihood.law.distribution
        + r"}\left("
        + r",\; ".join(
            r"\mathrm{" + name + "}=" + _expression_latex(model, argument)
            for name, argument in likelihood.law.arguments.items()
        )
        + r"\right)"
        for indicator, likelihood in model.iter_likelihoods()
    }


def state_equations(model: ModelSpec) -> list[StateEquation]:
    """Render actual drift, explicitly converting interval-authored parameters to rates."""
    terms: dict[ConstructId, list[str]] = defaultdict(list)
    for owner, mechanism in model.iter_mechanisms():
        target = owner.effect.id if isinstance(owner, CausalEdgeSpec) else owner.id
        expression = _expression_latex(model, mechanism.expression)
        if mechanism.kind == "potential":
            expression = (
                r"-\frac{\partial}{\partial "
                + _state_latex(model, target)
                + "}"
                + (r"\left[" + expression + r"\right]")
            )
        terms[target].append(expression)

    rows = []
    for key in model.state_order:
        construct = model._constructs[key]
        if construct.temporal_status == "time_invariant":
            equation = r"\mathrm{d}" + _state_latex(model, key) + " = 0"
        else:
            drift = " + ".join(terms[key]).replace(" + -", " - ")
            noise = r"\sum_j L_{" + _text(construct.name) + r",j}\,\mathrm{d}W_j(t)"
            equation = (
                r"\mathrm{d}"
                + _state_latex(model, key)
                + r" = \left["
                + drift
                + r"\right]\mathrm{d}t + "
                + noise
            )
        rows.append(StateEquation(construct_id=key, label=construct.name, latex=equation))
    return rows


def confounder_equations(model: ModelSpec) -> list[StateEquation]:
    """Label the shared-noise dependencies derived from the scientific DAG."""
    groups: dict[ConstructId, set[ConstructId]] = defaultdict(set)
    for (first, second, kind), sources in model.induced_dependencies.items():
        if kind == "innovation_correlation":
            for owner in sources:
                groups[owner].update((first, second))
    return [
        StateEquation(
            construct_id=owner,
            label=model.get_construct(owner).name,
            latex="U_{"
            + _text(model.get_construct(owner).name)
            + r"}\to\{"
            + ",\\,".join(_text(model.get_construct(key).name) for key in sorted(states))
            + r"\}",
        )
        for owner, states in sorted(groups.items())
    ]
