"""Equation projection of the same scientific mechanisms consumed by compilation."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, cast

from sympy import Float, Function, Integer, Max, Symbol, evaluate, latex, log

from nof1_causal_lab.artifacts.construct import CausalEdgeSpec
from nof1_causal_lab.artifacts.expressions import (
    CoefficientExpression,
    LiteralExpression,
    StateExpression,
    symbolic_call,
    symbolic_expression,
)
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform
from nof1_causal_lab.models.model_structure import selected_state_ids

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.expressions import Expression
    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.model_structure import StructuralSelection


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
    """Render the shared symbolic formula with the authored scientific legend."""
    symbolic = symbolic_expression(expression)
    parameters = {parameter.id: parameter for parameter in model.parameters}
    replacements, labels = {}, {}
    for symbol, operand in symbolic.nodes.items():
        if isinstance(operand, StateExpression):
            labels[symbol] = _state_latex(model, operand.construct_id)
        elif isinstance(operand, (CoefficientExpression, LiteralExpression)):
            reference = operand.value
            if isinstance(operand, CoefficientExpression) and reference is None:
                labels[symbol] = r"\underbrace{?}_{\text{" + operand.role.replace("_", " ") + "}}"
            elif isinstance(reference, (int, float)):
                replacements[symbol] = (
                    Integer(reference) if float(reference).is_integer() else Float(reference)
                )
            elif isinstance(reference, str):
                parameter = parameters[reference]
                theta, interval = Symbol(reference), Symbol("interval:" + reference)
                labels[theta] = r"\theta_{" + _text(parameter.name) + "}"
                labels[interval] = r"\Delta_{" + _text(parameter.name) + "}"
                match parameter.transform.kind:
                    case PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY:
                        replacements[symbol] = -log(theta) / interval
                    case PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE:
                        replacements[symbol] = theta / interval
                    case _:
                        replacements[symbol] = theta
    with evaluate(False):
        root = symbolic.root.xreplace(replacements)
        root = root.replace(
            Function("maximum"), lambda left, right: Max(left, right, evaluate=False)
        )
        root = root.replace(
            Function("sigmoid"), lambda argument: symbolic_call("logistic", argument)
        )
        root = root.replace(Function("normal_cdf"), lambda argument: symbolic_call("Phi", argument))
    return cast("str", latex(root, symbol_names=labels))


def observation_equations(model: ModelSpec) -> dict[IndicatorId, str]:
    """Render each declared native conditional law, including incomplete operands."""
    return {
        indicator.observation.id: "y_{"
        + _text(indicator.observation.name)
        + r"}(t) \sim \operatorname{"
        + likelihood.law.distribution
        + r"}\left("
        + r",\; ".join(
            r"\mathrm{" + name + "}=" + _expression_latex(model, argument)
            for name, argument in likelihood.law.operands()
        )
        + r"\right)"
        for indicator, likelihood in model.iter_likelihoods()
    }


def state_equations(selection: StructuralSelection) -> dict[ConstructId, str]:
    """Render actual drift, explicitly converting interval-authored parameters to rates."""
    model = selection.model
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

    rows: dict[ConstructId, str] = {}
    for key in selected_state_ids(selection):
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
        rows[key] = equation
    return rows


def confounder_equations(selection: StructuralSelection) -> dict[ConstructId, str]:
    """Label the shared-noise dependencies derived from the scientific DAG."""
    model = selection.model
    groups: dict[ConstructId, set[ConstructId]] = defaultdict(set)
    for (first, second, kind), sources in selection.induced_dependencies.items():
        if kind == "innovation_correlation":
            for owner in sources:
                groups[owner].update((first, second))
    return {
        owner: "U_{"
        + _text(model.get_construct(owner).name)
        + r"}\to\{"
        + ",\\,".join(_text(model.get_construct(key).name) for key in sorted(states))
        + r"\}"
        for owner, states in sorted(groups.items())
    }
