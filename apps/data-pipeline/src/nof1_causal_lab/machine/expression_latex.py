"""Precedence-aware LaTeX interpretation of scalar expression operations."""

from dataclasses import dataclass

from nof1_causal_lab.artifacts.expressions import BinaryOperator, ExpressionFunction


@dataclass(frozen=True)
class LatexValue:
    text: str
    precedence: int = 4
    constant: float | None = None

    def grouped(self, precedence: int) -> str:
        return r"\left(" + self.text + r"\right)" if self.precedence < precedence else self.text


def literal_latex(value: float) -> LatexValue:
    mantissa, scientific, exponent = f"{value:g}".partition("e")
    if scientific:
        return LatexValue(mantissa + r"\times 10^{" + str(int(exponent)) + "}", 2, value)
    return LatexValue(mantissa, 4 if value >= 0 else 2, value)


def binary_latex(operator: BinaryOperator, left: LatexValue, right: LatexValue) -> LatexValue:
    """Render arithmetic, applying only exact scalar identities for readability."""
    match operator:
        case "add" | "subtract":
            if right.constant == 0:
                return left
            if operator == "add" and left.constant == 0:
                return right
            sign = " + " if operator == "add" else " - "
            return LatexValue(left.grouped(1) + sign + right.grouped(2), 1)
        case "multiply":
            if left.constant == 0 or right.constant == 0:
                return literal_latex(0)
            if left.constant == 1:
                return right
            if right.constant == 1:
                return left
            if left.constant == -1:
                return LatexValue("-" + right.grouped(2), 2)
            return LatexValue(left.grouped(2) + r"\," + right.grouped(2), 2)
        case "divide":
            return LatexValue(r"\frac{" + left.text + "}{" + right.text + "}", 3)
        case "power":
            return LatexValue(left.grouped(4) + "^{" + right.text + "}", 3)
        case "maximum":
            return LatexValue(r"\max\!\left(" + left.text + "," + right.text + r"\right)")


def call_latex(name: ExpressionFunction, arguments: tuple[LatexValue, ...]) -> LatexValue:
    label = {
        "exp": r"\exp",
        "sigmoid": r"\operatorname{logistic}",
        "normal_cdf": r"\Phi",
        "ordered_cutpoints": r"\operatorname{ordered\_cutpoints}",
        "category_logits": r"\operatorname{category\_logits}",
    }[name]
    return LatexValue(label + r"\left(" + ", ".join(value.text for value in arguments) + r"\right)")
