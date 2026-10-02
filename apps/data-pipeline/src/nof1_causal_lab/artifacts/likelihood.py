"""Typed conditional laws; the sole observation constructor and family catalog."""

from __future__ import annotations

from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, assert_never, get_args

from pydantic import Field, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.distributions import DistributionFamily

from .evidence import LiteratureSource
from .expressions import CoefficientRole, Expression

if TYPE_CHECKING:
    from nof1_causal_lab.models.likelihoods import LikelihoodTerms


class LinkFunction(StrEnum):
    """Numerical response derived from a likelihood's argument expression."""

    IDENTITY = "identity"
    LOG = "log"
    INVERSE = "inverse"
    LOGIT = "logit"
    PROBIT = "probit"
    CUMULATIVE_LOGIT = "cumulative_logit"
    SOFTMAX = "softmax"


class _ObservationLaw(Value):
    """Shared catalog metadata; concrete laws own their required expression fields."""

    family: ClassVar[DistributionFamily]
    summary: ClassVar[str]
    links: ClassVar[tuple[str, ...]]
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ()


class DeltaLawSpec(_ObservationLaw):
    """The Delta conditional law."""

    distribution: Literal["Delta"] = "Delta"
    v: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.DELTA
    summary: ClassVar[str] = (
        "Exact observation of a state or its declared window summary, with no measurement noise. Missing observations impose no constraint. Particle inference supports direct point bindings; affine and interval constraints are not yet supported."
    )
    links: ClassVar[tuple[str, ...]] = ("identity",)


class NormalLawSpec(_ObservationLaw):
    """The Normal conditional law."""

    distribution: Literal["Normal"] = "Normal"
    loc: Expression
    scale: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.GAUSSIAN
    summary: ClassVar[str] = "Continuous unbounded data, approximately symmetric."
    links: ClassVar[tuple[str, ...]] = ("identity",)


class StudentTLawSpec(_ObservationLaw):
    """The StudentT conditional law."""

    distribution: Literal["StudentT"] = "StudentT"
    df: Expression
    loc: Expression
    scale: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.STUDENT_T
    summary: ClassVar[str] = "Continuous data with heavy tails or outliers."
    links: ClassVar[tuple[str, ...]] = ("identity",)
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ("degrees_of_freedom",)


class PoissonLawSpec(_ObservationLaw):
    """The Poisson conditional law."""

    distribution: Literal["Poisson"] = "Poisson"
    rate: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.POISSON
    summary: ClassVar[str] = "Count data with variance roughly tracking the mean."
    links: ClassVar[tuple[str, ...]] = ("log",)


class GammaLawSpec(_ObservationLaw):
    """The Gamma conditional law."""

    distribution: Literal["Gamma"] = "Gamma"
    concentration: Expression
    rate: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.GAMMA
    summary: ClassVar[str] = "Positive continuous data such as durations or reaction times."
    links: ClassVar[tuple[str, ...]] = ("log", "inverse")
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ("shape",)


class BernoulliLogitsLawSpec(_ObservationLaw):
    """The BernoulliLogits conditional law."""

    distribution: Literal["BernoulliLogits"] = "BernoulliLogits"
    logits: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.BERNOULLI
    summary: ClassVar[str] = "Binary outcomes with two possible states."
    links: ClassVar[tuple[str, ...]] = ("logit",)


class BernoulliProbsLawSpec(_ObservationLaw):
    """The BernoulliProbs conditional law."""

    distribution: Literal["BernoulliProbs"] = "BernoulliProbs"
    probs: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.BERNOULLI
    summary: ClassVar[str] = "Binary outcomes with two possible states."
    links: ClassVar[tuple[str, ...]] = ("probit",)


class NegativeBinomial2LawSpec(_ObservationLaw):
    """The NegativeBinomial2 conditional law."""

    distribution: Literal["NegativeBinomial2"] = "NegativeBinomial2"
    mean: Expression
    concentration: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.NEGATIVE_BINOMIAL
    summary: ClassVar[str] = "Overdispersed count data where variance exceeds the mean."
    links: ClassVar[tuple[str, ...]] = ("log",)
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ("dispersion",)


class BetaLawSpec(_ObservationLaw):
    """The Beta conditional law."""

    distribution: Literal["Beta"] = "Beta"
    concentration1: Expression
    concentration0: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.BETA
    summary: ClassVar[str] = "Proportions or rates strictly inside the unit interval."
    links: ClassVar[tuple[str, ...]] = ("logit", "probit")
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ("concentration",)


class OrderedLogisticLawSpec(_ObservationLaw):
    """The OrderedLogistic conditional law."""

    distribution: Literal["OrderedLogistic"] = "OrderedLogistic"
    predictor: Expression
    cutpoints: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.ORDERED_LOGISTIC
    summary: ClassVar[str] = (
        "Ordered categorical outcomes with ranked levels. Keeps a loading on the latent (fixed logistic scale), unlike `categorical`."
    )
    links: ClassVar[tuple[str, ...]] = ("cumulative_logit",)
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ("cutpoint_base", "cutpoint_gaps")


class CategoricalLawSpec(_ObservationLaw):
    """The Categorical conditional law."""

    distribution: Literal["Categorical"] = "Categorical"
    logits: Expression
    family: ClassVar[DistributionFamily] = DistributionFamily.CATEGORICAL
    summary: ClassVar[str] = (
        "Unordered multi-class outcomes. Choosing it removes the channel's loading (the class slopes are exactly redundant with it, so the compiler pins it); discrimination moves into `obs_cat_slopes`."
    )
    links: ClassVar[tuple[str, ...]] = ("softmax",)
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = (
        "category_intercepts",
        "category_slopes",
    )


type ObservationLawSpec = Annotated[
    DeltaLawSpec
    | NormalLawSpec
    | StudentTLawSpec
    | PoissonLawSpec
    | GammaLawSpec
    | BernoulliLogitsLawSpec
    | BernoulliProbsLawSpec
    | NegativeBinomial2LawSpec
    | BetaLawSpec
    | OrderedLogisticLawSpec
    | CategoricalLawSpec,
    Field(discriminator="distribution"),
]


# Derive catalogs from the same closed union that defines the authored schema.
OBSERVATION_LAW_TYPES: tuple[type[_ObservationLaw], ...] = get_args(
    get_args(ObservationLawSpec.__value__)[0]
)
OBSERVATION_FAMILY_SPECS = tuple({law.family: law for law in OBSERVATION_LAW_TYPES}.values())
OBSERVATION_LINK_VALUES_BY_DISTRIBUTION = {
    family: tuple(
        dict.fromkeys(
            link for law in OBSERVATION_LAW_TYPES if law.family == family for link in law.links
        )
    )
    for family in (law.family for law in OBSERVATION_FAMILY_SPECS)
}
VALID_LINKS_FOR_DISTRIBUTION = {
    family: frozenset(LinkFunction(link) for link in links)
    for family, links in OBSERVATION_LINK_VALUES_BY_DISTRIBUTION.items()
}


def observation_expressions(law: ObservationLawSpec) -> tuple[tuple[str, Expression], ...]:
    """Render typed operands without reconstructing a string-keyed argument bag."""
    match law:
        case DeltaLawSpec():
            return (("v", law.v),)
        case NormalLawSpec():
            return (
                ("loc", law.loc),
                ("scale", law.scale),
            )
        case StudentTLawSpec():
            return (
                ("df", law.df),
                ("loc", law.loc),
                ("scale", law.scale),
            )
        case PoissonLawSpec():
            return (("rate", law.rate),)
        case GammaLawSpec():
            return (
                ("concentration", law.concentration),
                ("rate", law.rate),
            )
        case BernoulliLogitsLawSpec():
            return (("logits", law.logits),)
        case BernoulliProbsLawSpec():
            return (("probs", law.probs),)
        case NegativeBinomial2LawSpec():
            return (
                ("mean", law.mean),
                ("concentration", law.concentration),
            )
        case BetaLawSpec():
            return (
                ("concentration1", law.concentration1),
                ("concentration0", law.concentration0),
            )
        case OrderedLogisticLawSpec():
            return (
                ("predictor", law.predictor),
                ("cutpoints", law.cutpoints),
            )
        case CategoricalLawSpec():
            return (("logits", law.logits),)
    assert_never(law)


class LikelihoodSpec(Value):
    """An indicator's conditional probability law and its scientific justification."""

    law: ObservationLawSpec
    standardized: bool = Field(
        default=False,
        description="Whether observations are mean-centered and scaled before fitting.",
    )
    reasoning: str = Field(description="Why this conditional law was chosen for the indicator")
    sources: tuple[LiteratureSource, ...] = ()

    @model_validator(mode="after")
    def parse_expression(self) -> Self:
        _ = self.parsed
        return self

    @cached_property
    def parsed(self) -> LikelihoodTerms:
        """Retain the supported expression grammar at its scientific owner boundary."""
        from nof1_causal_lab.models.likelihoods import likelihood_terms

        return likelihood_terms(self.law)
