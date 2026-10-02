"""Interpret conditional distribution expressions for authoring and numerical lowering."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.expressions import (
    BinaryExpression,
    CallExpression,
    CoefficientExpression,
    LiteralExpression,
    StateExpression,
    coefficient,
)
from nof1_causal_lab.artifacts.likelihood import (
    VALID_LINKS_FOR_DISTRIBUTION,
    LinkFunction,
    ObservationLawSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.artifacts.expressions import (
        CoefficientRole,
        Expression,
        ExpressionFunction,
    )
    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.distributions import DistributionFamily


def function(name: ExpressionFunction, *arguments: Expression) -> CallExpression:
    return CallExpression(function=name, arguments=arguments)


def _operand(value: Expression, role: CoefficientRole) -> CoefficientExpression:
    if not isinstance(value, CoefficientExpression) or value.role != role:
        raise ValueError(f"Observation argument requires a {role} coefficient expression")
    return value


def _binary_argument(value: Expression, operator: str) -> BinaryExpression:
    if not isinstance(value, BinaryExpression) or value.operator != operator:
        raise ValueError(f"Observation argument requires a {operator} expression")
    return value


def _call(value: Expression, name: str) -> tuple[Expression, ...]:
    if not isinstance(value, CallExpression) or value.function != name:
        raise ValueError(f"Observation argument requires {name}(...)")
    return value.arguments


@dataclass(frozen=True)
class LikelihoodAnalysis:
    """Native affine response and coefficient uses derived from a conditional law."""

    family: DistributionFamily
    link: LinkFunction
    predictor: Expression
    intercept: CoefficientExpression
    loadings: Mapping[ConstructId, CoefficientExpression]
    auxiliary: tuple[CoefficientExpression, ...]

    @property
    def operands(self) -> tuple[CoefficientExpression, ...]:
        return (self.intercept, *self.loadings.values(), *self.auxiliary)


def _affine_terms(
    predictor: Expression,
) -> tuple[CoefficientExpression, dict[ConstructId, CoefficientExpression]]:
    if isinstance(predictor, StateExpression):
        return (
            coefficient(0, "observation_intercept"),
            {predictor.construct_id: coefficient(1, "loading")},
        )
    intercepts: list[CoefficientExpression] = []
    loadings: dict[ConstructId, CoefficientExpression] = {}

    def visit(value: Expression) -> None:
        if isinstance(value, BinaryExpression) and value.operator == "add":
            visit(value.left)
            visit(value.right)
        elif isinstance(value, CoefficientExpression):
            intercepts.append(_operand(value, "observation_intercept"))
        elif isinstance(value, BinaryExpression) and value.operator == "multiply":
            lhs, rhs = value.left, value.right
            if isinstance(lhs, StateExpression):
                lhs, rhs = rhs, lhs
            if not isinstance(rhs, StateExpression):
                raise ValueError("A measurement loading must multiply a scientific state")
            if rhs.construct_id in loadings:
                raise ValueError("An observation predictor has one loading per construct")
            loadings[rhs.construct_id] = _operand(lhs, "loading")
        else:
            raise ValueError("Observation predictors require an affine expression over states")

    visit(predictor)
    if len(intercepts) != 1 or not loadings:
        raise ValueError("An observation predictor requires one intercept and at least one state")
    return intercepts[0], loadings


def _response(
    value: Expression, links: tuple[LinkFunction, ...]
) -> tuple[Expression, LinkFunction]:
    for link in links:
        if link == LinkFunction.IDENTITY:
            return value, link
        name = {
            LinkFunction.LOG: "exp",
            LinkFunction.LOGIT: "sigmoid",
            LinkFunction.PROBIT: "normal_cdf",
        }.get(link)
        if name and isinstance(value, CallExpression) and value.function == name:
            return value.arguments[0], link
        if (
            link == LinkFunction.INVERSE
            and isinstance(value, BinaryExpression)
            and value.operator == "divide"
            and value.left == LiteralExpression(value=1)
        ):
            return value.right, link
    raise ValueError("Observation response is outside the supported expression grammar")


def likelihood_terms(law: ObservationLawSpec) -> LikelihoodAnalysis:
    auxiliary: list[CoefficientExpression] = []
    family = law.family
    match law.distribution:
        case "Delta":
            predictor, link = law.v, LinkFunction.IDENTITY
        case "Normal" | "StudentT":
            predictor, link = law.loc, LinkFunction.IDENTITY
            auxiliary.append(_operand(law.scale, "observation_scale"))
            if law.distribution == "StudentT":
                auxiliary.append(_operand(law.df, "degrees_of_freedom"))
        case "Poisson":
            predictor, link = _response(law.rate, (LinkFunction.LOG,))
        case "BernoulliLogits":
            predictor, link = law.logits, LinkFunction.LOGIT
        case "BernoulliProbs":
            predictor, link = _response(law.probs, (LinkFunction.PROBIT, LinkFunction.LOGIT))
        case "Gamma":
            shape = _operand(law.concentration, "shape")
            rate = _binary_argument(law.rate, "divide")
            if rate.left != shape:
                raise ValueError("Gamma rate must divide its concentration by the response mean")
            predictor, link = _response(rate.right, (LinkFunction.LOG, LinkFunction.INVERSE))
            auxiliary.append(shape)
        case "NegativeBinomial2":
            predictor, link = _response(law.mean, (LinkFunction.LOG,))
            auxiliary.append(_operand(law.concentration, "dispersion"))
        case "Beta":
            alpha = _binary_argument(law.concentration1, "multiply")
            concentration = _operand(alpha.right, "concentration")
            expected = (LiteralExpression(value=1) - alpha.left) * concentration
            if law.concentration0 != expected:
                raise ValueError(
                    "Beta concentrations must express complementary means and one concentration"
                )
            predictor, link = _response(alpha.left, (LinkFunction.LOGIT, LinkFunction.PROBIT))
            auxiliary.append(concentration)
        case "OrderedLogistic":
            predictor, link = law.predictor, LinkFunction.CUMULATIVE_LOGIT
            base, gaps = _call(law.cutpoints, "ordered_cutpoints")
            auxiliary.append(_operand(base, "cutpoint_base"))
            if gaps != LiteralExpression(value=0):
                auxiliary.append(_operand(gaps, "cutpoint_gaps"))
        case "Categorical":
            predictor, intercepts, slopes = _call(law.logits, "category_logits")
            link = LinkFunction.SOFTMAX
            auxiliary.extend(
                (_operand(intercepts, "category_intercepts"), _operand(slopes, "category_slopes"))
            )
    intercept, loadings = _affine_terms(predictor)
    if link not in VALID_LINKS_FOR_DISTRIBUTION[family]:
        raise ValueError(f"Unsupported response for {law.distribution}")
    return LikelihoodAnalysis(
        family, link, predictor, intercept, MappingProxyType(loadings), tuple(auxiliary)
    )
