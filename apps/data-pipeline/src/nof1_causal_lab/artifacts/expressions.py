"""Scientific expressions shared by dynamics, conditional laws, bindings, and equations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from nof1_causal_lab.scalar_functions import hill_response, restoring_drift

from .identity import (
    ConstructId,  # noqa: TC001
    ParameterId,  # noqa: TC001
)
from .parameter import SiteKind, SupportClass

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

type CoefficientRole = Literal[
    "center",
    "decay",
    "quartic",
    "intercept",
    "weight",
    "emax",
    "ec50",
    "exponent",
    "loading",
    "observation_intercept",
    "observation_scale",
    "degrees_of_freedom",
    "shape",
    "dispersion",
    "concentration",
    "cutpoint_base",
    "cutpoint_gaps",
    "category_intercepts",
    "category_slopes",
    "diffusion_scale",
    "diffusion_loading",
    "process_degrees_of_freedom",
    "initial_mean",
    "initial_scale",
    "initial_correlation",
]
CONSTRUCT_COEFFICIENT_ROLES: frozenset[CoefficientRole] = frozenset(
    {
        "diffusion_scale",
        "diffusion_loading",
        "process_degrees_of_freedom",
        "initial_mean",
        "initial_scale",
        "initial_correlation",
    }
)
type BinaryOperator = Literal["add", "subtract", "multiply", "divide", "power", "maximum"]
type ExpressionFunction = Literal[
    "exp", "sigmoid", "normal_cdf", "ordered_cutpoints", "category_logits"
]


@dataclass(frozen=True)
class CoefficientMeaning:
    """Scientific meaning and support of one scalar operand."""

    quantity: SiteKind
    support: SupportClass
    allows_zero: bool = False


COEFFICIENT_MEANINGS: Mapping[CoefficientRole, CoefficientMeaning] = {
    "center": CoefficientMeaning(SiteKind.DYNAMICS_POTENTIAL_CENTER, SupportClass.REAL),
    "decay": CoefficientMeaning(SiteKind.DYNAMICS_DECAY, SupportClass.POSITIVE),
    "quartic": CoefficientMeaning(
        SiteKind.DYNAMICS_POTENTIAL_QUARTIC, SupportClass.POSITIVE, allows_zero=True
    ),
    "intercept": CoefficientMeaning(SiteKind.DYNAMICS_CINT, SupportClass.REAL),
    "weight": CoefficientMeaning(SiteKind.DYNAMICS_WEIGHT, SupportClass.REAL),
    "emax": CoefficientMeaning(SiteKind.HILL_EMAX, SupportClass.POSITIVE),
    "ec50": CoefficientMeaning(SiteKind.HILL_EC50, SupportClass.POSITIVE),
    "exponent": CoefficientMeaning(SiteKind.HILL_N, SupportClass.POSITIVE),
    "loading": CoefficientMeaning(SiteKind.LOADING, SupportClass.REAL),
    "observation_intercept": CoefficientMeaning(SiteKind.MANIFEST_MEANS, SupportClass.REAL),
    "observation_scale": CoefficientMeaning(
        SiteKind.MANIFEST_VAR_DIAG, SupportClass.POSITIVE, allows_zero=True
    ),
    "degrees_of_freedom": CoefficientMeaning(SiteKind.OBS_DF, SupportClass.POSITIVE),
    "shape": CoefficientMeaning(SiteKind.OBS_SHAPE, SupportClass.POSITIVE),
    "dispersion": CoefficientMeaning(SiteKind.OBS_R, SupportClass.POSITIVE),
    "concentration": CoefficientMeaning(SiteKind.OBS_CONCENTRATION, SupportClass.POSITIVE),
    "cutpoint_base": CoefficientMeaning(SiteKind.OBS_ORDERED_BASE, SupportClass.REAL),
    "cutpoint_gaps": CoefficientMeaning(SiteKind.OBS_ORDERED_GAPS, SupportClass.POSITIVE),
    "category_intercepts": CoefficientMeaning(SiteKind.OBS_CAT_INTERCEPTS, SupportClass.REAL),
    "category_slopes": CoefficientMeaning(SiteKind.OBS_CAT_SLOPES, SupportClass.REAL),
    "diffusion_scale": CoefficientMeaning(
        SiteKind.DIFFUSION_DIAG, SupportClass.POSITIVE, allows_zero=True
    ),
    "diffusion_loading": CoefficientMeaning(SiteKind.DIFFUSION_LOWER, SupportClass.REAL),
    "process_degrees_of_freedom": CoefficientMeaning(SiteKind.PROC_DF, SupportClass.POSITIVE),
    "initial_mean": CoefficientMeaning(SiteKind.T0_MEANS, SupportClass.REAL),
    "initial_scale": CoefficientMeaning(
        SiteKind.T0_VAR_DIAG, SupportClass.POSITIVE, allows_zero=True
    ),
    "initial_correlation": CoefficientMeaning(SiteKind.T0_VAR_LOWER, SupportClass.CORRELATION),
}


class ExpressionValue(BaseModel):
    """An immutable scalar expression, with arithmetic for scientific constructors."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    def __add__(self, other: ExpressionValue | float) -> BinaryExpression:
        return BinaryExpression(operator="add", left=self, right=expression(other))

    def __sub__(self, other: ExpressionValue | float) -> BinaryExpression:
        return BinaryExpression(operator="subtract", left=self, right=expression(other))

    def __mul__(self, other: ExpressionValue | float) -> BinaryExpression:
        return BinaryExpression(operator="multiply", left=self, right=expression(other))

    def __truediv__(self, other: ExpressionValue | float) -> BinaryExpression:
        return BinaryExpression(operator="divide", left=self, right=expression(other))

    def __pow__(self, other: ExpressionValue | float) -> BinaryExpression:
        return BinaryExpression(operator="power", left=self, right=expression(other))

    def __neg__(self) -> BinaryExpression:
        return LiteralExpression(value=-1) * self


class LiteralExpression(ExpressionValue):
    """A finite scalar constant in a model equation."""

    kind: Literal["literal"] = "literal"
    value: FiniteFloat


class StateExpression(ExpressionValue):
    """A construct's state or declared known input, referenced by identity."""

    kind: Literal["state"] = "state"
    construct_id: ConstructId


class CoefficientExpression(ExpressionValue):
    """A scientifically typed coefficient operand, literal or parameter reference."""

    kind: Literal["coefficient"] = "coefficient"
    role: CoefficientRole
    value: FiniteFloat | ParameterId | None = Field(
        default=None,
        description="Finite literal or persistent parameter ID; null leaves the operand unassigned.",
    )
    construct_ids: tuple[ConstructId, ...] = Field(
        default=(), description="Additional constructs participating in this coefficient use."
    )

    @property
    def meaning(self) -> CoefficientMeaning:
        return COEFFICIENT_MEANINGS[self.role]

    def validate_value(self, value: float) -> None:
        meaning = self.meaning
        if meaning.support == SupportClass.POSITIVE and (
            value < 0 if meaning.allows_zero else value <= 0
        ):
            requirement = "non-negative" if meaning.allows_zero else "positive"
            raise ValueError(f"{self.role} must be {requirement}; got {value}")

    @model_validator(mode="after")
    def validate_fixed_support(self) -> CoefficientExpression:
        if isinstance(self.value, (int, float)):
            self.validate_value(self.value)
        return self


class BinaryExpression(ExpressionValue):
    """A supported scalar operation composing two expressions."""

    kind: Literal["binary"] = "binary"
    operator: BinaryOperator
    left: Expression
    right: Expression


class CallExpression(ExpressionValue):
    """A supported mathematical function, including explicit discrete contrasts."""

    kind: Literal["call"] = "call"
    function: ExpressionFunction
    arguments: tuple[Expression, ...]

    @model_validator(mode="after")
    def validate_arity(self) -> CallExpression:
        expected = {"ordered_cutpoints": 2, "category_logits": 3}.get(self.function, 1)
        if len(self.arguments) != expected:
            raise ValueError(f"{self.function} requires {expected} arguments")
        return self


type Expression = Annotated[
    LiteralExpression | StateExpression | CoefficientExpression | BinaryExpression | CallExpression,
    Field(discriminator="kind"),
]


def expression(value: ExpressionValue | float) -> Expression:
    if isinstance(
        value,
        (
            LiteralExpression,
            StateExpression,
            CoefficientExpression,
            BinaryExpression,
            CallExpression,
        ),
    ):
        return value
    if isinstance(value, (float, int)):
        return LiteralExpression(value=value)
    raise TypeError("Arithmetic requires a concrete scalar expression or finite number")


def state(identity: ConstructId) -> StateExpression:
    return StateExpression(construct_id=identity)


def coefficient(
    value: float | ParameterId | None,
    role: CoefficientRole,
    *,
    construct_ids: tuple[ConstructId, ...] = (),
) -> CoefficientExpression:
    return CoefficientExpression(role=role, value=value, construct_ids=construct_ids)


def restoring_force(
    target: ConstructId,
    *,
    center: float | ParameterId | None,
    stiffness: float | ParameterId | None,
    quartic: float | ParameterId | None,
) -> Expression:
    """Restoring drift -stiffness * (x - center) - quartic * (x - center)^3."""
    return restoring_drift(
        state(target),
        coefficient(center, "center"),
        coefficient(stiffness, "decay"),
        coefficient(quartic, "quartic"),
    )


def linear_effect(source: ConstructId, weight: float | ParameterId) -> Expression:
    return coefficient(weight, "weight") * state(source)


def restoring_potential(
    target: ConstructId,
    *,
    center: float | ParameterId | None,
    stiffness: float | ParameterId | None,
    quartic: float | ParameterId | None,
) -> Expression:
    """Scalar node energy; Dynestyx differentiates it to obtain the restoring drift."""
    delta = state(target) - coefficient(center, "center")
    return (
        coefficient(stiffness, "decay") * delta**2 / 2
        + coefficient(quartic, "quartic") * delta**4 / 4
    )


def hill(
    source: Expression,
    *,
    emax: float | ParameterId | None,
    ec50: float | ParameterId | None,
    n: float | ParameterId | None,
) -> Expression:
    """The native non-negative Hill response, including its numerical denominator term."""
    return hill_response(
        source,
        coefficient(emax, "emax"),
        coefficient(ec50, "ec50"),
        coefficient(n, "exponent"),
        maximum=lambda value, lower: BinaryExpression(
            operator="maximum", left=value, right=LiteralExpression(value=lower)
        ),
    )


def walk_expression(value: Expression) -> Iterator[Expression]:
    yield value
    if isinstance(value, BinaryExpression):
        yield from walk_expression(value.left)
        yield from walk_expression(value.right)
    elif isinstance(value, CallExpression):
        for argument in value.arguments:
            yield from walk_expression(argument)


def expression_states(value: Expression) -> frozenset[ConstructId]:
    return frozenset(
        node.construct_id for node in walk_expression(value) if isinstance(node, StateExpression)
    )


def expression_coefficients(value: Expression) -> tuple[CoefficientExpression, ...]:
    """One use per distinct operand; formulas can mention a coefficient repeatedly."""
    return tuple(
        dict.fromkeys(
            node for node in walk_expression(value) if isinstance(node, CoefficientExpression)
        )
    )


def map_expression(value: Expression, transform: Callable[[Expression], Expression]) -> Expression:
    """Revise references or constants without interpreting the relationship's function."""
    if isinstance(value, BinaryExpression):
        value = BinaryExpression(
            operator=value.operator,
            left=map_expression(value.left, transform),
            right=map_expression(value.right, transform),
        )
    elif isinstance(value, CallExpression):
        value = CallExpression(
            function=value.function,
            arguments=tuple(map_expression(argument, transform) for argument in value.arguments),
        )
    return transform(value)


def fold_expression[T](
    value: Expression,
    *,
    literal: Callable[[float], T],
    state_value: Callable[[ConstructId], T],
    coefficient_value: Callable[[CoefficientExpression], T],
    binary: Callable[[BinaryOperator, T, T], T],
    call: Callable[[ExpressionFunction, tuple[T, ...]], T],
) -> T:
    """Interpret a tree as executable scalar arithmetic or rendered mathematics."""

    def visit(node: Expression) -> T:
        match node:
            case LiteralExpression():
                return literal(node.value)
            case StateExpression():
                return state_value(node.construct_id)
            case CoefficientExpression():
                return coefficient_value(node)
            case BinaryExpression():
                return binary(node.operator, visit(node.left), visit(node.right))
            case CallExpression():
                return call(node.function, tuple(visit(argument) for argument in node.arguments))

    return visit(value)


def linear_coefficient(value: Expression, source: ConstructId) -> float | ParameterId:
    """Recognize the scalar linear form supported by input and projection matrices."""
    if isinstance(value, BinaryExpression) and value.operator == "multiply":
        for lhs, rhs in ((value.left, value.right), (value.right, value.left)):
            if (
                isinstance(lhs, CoefficientExpression)
                and lhs.role == "weight"
                and rhs == state(source)
            ):
                if lhs.value is None:
                    from nof1_causal_lab.compilation_errors import IncompleteModelError

                    raise IncompleteModelError("Linear expression requires its coefficient")
                return lhs.value
    raise ValueError("This execution boundary requires one scalar linear coefficient")


def restoring_coefficients(
    value: Expression, target: ConstructId, *, kind: Literal["drift", "potential"] = "drift"
) -> tuple[CoefficientExpression, ...]:
    """Recognize restoring terms for the existing anchoring and scale checks.

    A role annotation alone is not a certificate. The complete term must equal
    the scientific constructor before it can establish an anchor.
    """
    operands = expression_coefficients(value)
    roles = {node.role: node for node in operands}
    constructor = restoring_potential if kind == "potential" else restoring_force
    if (
        len(operands) == 3
        and roles.keys() == {"center", "decay", "quartic"}
        and value
        == constructor(
            target,
            center=roles["center"].value,
            stiffness=roles["decay"].value,
            quartic=roles["quartic"].value,
        )
    ):
        return operands
    if isinstance(value, BinaryExpression) and value.operator == "add":
        return (
            *restoring_coefficients(value.left, target, kind=kind),
            *restoring_coefficients(value.right, target, kind=kind),
        )
    return ()


def coefficient_key(operand: CoefficientExpression) -> ParameterId:
    """The numerical operand key follows a parameter's persistent identity."""
    reference = operand.value
    if not isinstance(reference, str):
        raise TypeError("Fixed coefficients do not have numerical sample sites")
    return reference


def hill_applications(
    value: Expression,
) -> Iterator[tuple[Expression, CoefficientExpression, CoefficientExpression]]:
    """Recognize complete Hill applications for saturation diagnostics, including compositions."""
    for node in walk_expression(value):
        if not isinstance(node, BinaryExpression) or node.operator != "divide":
            continue
        numerator = node.left
        if not isinstance(numerator, BinaryExpression) or numerator.operator != "multiply":
            continue
        emax, power = numerator.left, numerator.right
        if not isinstance(emax, CoefficientExpression) or emax.role != "emax":
            continue
        if not isinstance(power, BinaryExpression) or power.operator != "power":
            continue
        dose, exponent = power.left, power.right
        if not isinstance(exponent, CoefficientExpression) or exponent.role != "exponent":
            continue
        if not isinstance(dose, BinaryExpression) or dose.operator != "maximum":
            continue
        for operand in expression_coefficients(node.right):
            if operand.role == "ec50" and node == hill(
                dose.left, emax=emax.value, ec50=operand.value, n=exponent.value
            ):
                yield dose.left, operand, exponent
