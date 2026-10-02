"""Typed conditional laws; the sole observation constructor and family catalog."""

from __future__ import annotations

from enum import StrEnum
from functools import cached_property
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self, assert_never, get_args

from pydantic import ConfigDict, Field, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.distributions import DistributionFamily

from .evidence import LiteratureSource
from .expressions import CoefficientRole, Expression

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.models.likelihoods import LikelihoodAnalysis


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    family: ClassVar[DistributionFamily]
    summary: ClassVar[str]
    links: ClassVar[tuple[str, ...]]
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ()


class DeltaLawSpec[A](_ObservationLaw):
    """The Delta conditional law."""

    distribution: Literal["Delta"] = "Delta"
    v: A
    family: ClassVar[DistributionFamily] = DistributionFamily.DELTA
    summary: ClassVar[str] = (
        "Exact observation of a state or its declared window summary, with no measurement noise. Missing observations impose no constraint. Particle inference supports direct point bindings; affine and interval constraints are not yet supported."
    )
    links: ClassVar[tuple[str, ...]] = ("identity",)

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("v", self.v),)


class NormalLawSpec[A](_ObservationLaw):
    """The Normal conditional law."""

    distribution: Literal["Normal"] = "Normal"
    loc: A
    scale: A
    family: ClassVar[DistributionFamily] = DistributionFamily.GAUSSIAN
    summary: ClassVar[str] = "Continuous unbounded data, approximately symmetric."
    links: ClassVar[tuple[str, ...]] = ("identity",)

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("loc", self.loc), ("scale", self.scale))


class StudentTLawSpec[A](_ObservationLaw):
    """The StudentT conditional law."""

    distribution: Literal["StudentT"] = "StudentT"
    df: A
    loc: A
    scale: A
    family: ClassVar[DistributionFamily] = DistributionFamily.STUDENT_T
    summary: ClassVar[str] = "Continuous data with heavy tails or outliers."
    links: ClassVar[tuple[str, ...]] = ("identity",)
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ("degrees_of_freedom",)

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("df", self.df), ("loc", self.loc), ("scale", self.scale))


class PoissonLawSpec[A](_ObservationLaw):
    """The Poisson conditional law."""

    distribution: Literal["Poisson"] = "Poisson"
    rate: A
    family: ClassVar[DistributionFamily] = DistributionFamily.POISSON
    summary: ClassVar[str] = "Count data with variance roughly tracking the mean."
    links: ClassVar[tuple[str, ...]] = ("log",)

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("rate", self.rate),)


class GammaLawSpec[A](_ObservationLaw):
    """The Gamma conditional law."""

    distribution: Literal["Gamma"] = "Gamma"
    concentration: A
    rate: A
    family: ClassVar[DistributionFamily] = DistributionFamily.GAMMA
    summary: ClassVar[str] = "Positive continuous data such as durations or reaction times."
    links: ClassVar[tuple[str, ...]] = ("log", "inverse")
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ("shape",)

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("concentration", self.concentration), ("rate", self.rate))


class BernoulliLogitsLawSpec[A](_ObservationLaw):
    """The BernoulliLogits conditional law."""

    distribution: Literal["BernoulliLogits"] = "BernoulliLogits"
    logits: A
    family: ClassVar[DistributionFamily] = DistributionFamily.BERNOULLI
    summary: ClassVar[str] = "Binary outcomes with two possible states."
    links: ClassVar[tuple[str, ...]] = ("logit",)

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("logits", self.logits),)


class BernoulliProbsLawSpec[A](_ObservationLaw):
    """The BernoulliProbs conditional law."""

    distribution: Literal["BernoulliProbs"] = "BernoulliProbs"
    probs: A
    family: ClassVar[DistributionFamily] = DistributionFamily.BERNOULLI
    summary: ClassVar[str] = "Binary outcomes with two possible states."
    links: ClassVar[tuple[str, ...]] = ("logit", "probit")

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("probs", self.probs),)


class NegativeBinomial2LawSpec[A](_ObservationLaw):
    """The NegativeBinomial2 conditional law."""

    distribution: Literal["NegativeBinomial2"] = "NegativeBinomial2"
    mean: A
    concentration: A
    family: ClassVar[DistributionFamily] = DistributionFamily.NEGATIVE_BINOMIAL
    summary: ClassVar[str] = "Overdispersed count data where variance exceeds the mean."
    links: ClassVar[tuple[str, ...]] = ("log",)
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ("dispersion",)

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("mean", self.mean), ("concentration", self.concentration))


class BetaLawSpec[A](_ObservationLaw):
    """The Beta conditional law."""

    distribution: Literal["Beta"] = "Beta"
    concentration1: A
    concentration0: A
    family: ClassVar[DistributionFamily] = DistributionFamily.BETA
    summary: ClassVar[str] = "Proportions or rates strictly inside the unit interval."
    links: ClassVar[tuple[str, ...]] = ("logit", "probit")
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ("concentration",)

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("concentration1", self.concentration1), ("concentration0", self.concentration0))


class OrderedLogisticLawSpec[A](_ObservationLaw):
    """The OrderedLogistic conditional law."""

    distribution: Literal["OrderedLogistic"] = "OrderedLogistic"
    predictor: A
    cutpoints: A
    family: ClassVar[DistributionFamily] = DistributionFamily.ORDERED_LOGISTIC
    summary: ClassVar[str] = (
        "Ordered categorical outcomes with ranked levels. Keeps a loading on the latent (fixed logistic scale), unlike `categorical`."
    )
    links: ClassVar[tuple[str, ...]] = ("cumulative_logit",)
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = ("cutpoint_base", "cutpoint_gaps")

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("predictor", self.predictor), ("cutpoints", self.cutpoints))


class CategoricalLawSpec[A](_ObservationLaw):
    """The Categorical conditional law."""

    distribution: Literal["Categorical"] = "Categorical"
    logits: A
    family: ClassVar[DistributionFamily] = DistributionFamily.CATEGORICAL
    summary: ClassVar[str] = (
        "Unordered multi-class outcomes. Choosing it removes the channel's loading (the class slopes are exactly redundant with it, so the compiler pins it); discrimination moves into `obs_cat_slopes`."
    )
    links: ClassVar[tuple[str, ...]] = ("softmax",)
    parameter_roles: ClassVar[tuple[CoefficientRole, ...]] = (
        "category_intercepts",
        "category_slopes",
    )

    def operands(self) -> tuple[tuple[str, A], ...]:
        """Typed operands in their declared order."""
        return (("logits", self.logits),)


type Law[A] = Annotated[
    DeltaLawSpec[A]
    | NormalLawSpec[A]
    | StudentTLawSpec[A]
    | PoissonLawSpec[A]
    | GammaLawSpec[A]
    | BernoulliLogitsLawSpec[A]
    | BernoulliProbsLawSpec[A]
    | NegativeBinomial2LawSpec[A]
    | BetaLawSpec[A]
    | OrderedLogisticLawSpec[A]
    | CategoricalLawSpec[A],
    Field(discriminator="distribution"),
]


type ObservationLawSpec = Law[Expression]

# Derive catalogs from the same closed union that defines the authored schema.
OBSERVATION_LAW_TYPES: tuple[type[_ObservationLaw], ...] = tuple(
    law.__pydantic_generic_metadata__["origin"] for law in get_args(get_args(Law.__value__)[0])
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


def map_law[A, B](law: Law[A], function: Callable[[A], B]) -> Law[B]:
    """Transform every native operand, preserving its constructor and discriminator."""
    match law:
        case DeltaLawSpec():
            return DeltaLawSpec(v=function(law.v))
        case NormalLawSpec():
            return NormalLawSpec(loc=function(law.loc), scale=function(law.scale))
        case StudentTLawSpec():
            return StudentTLawSpec(
                df=function(law.df), loc=function(law.loc), scale=function(law.scale)
            )
        case PoissonLawSpec():
            return PoissonLawSpec(rate=function(law.rate))
        case GammaLawSpec():
            return GammaLawSpec(concentration=function(law.concentration), rate=function(law.rate))
        case BernoulliLogitsLawSpec():
            return BernoulliLogitsLawSpec(logits=function(law.logits))
        case BernoulliProbsLawSpec():
            return BernoulliProbsLawSpec(probs=function(law.probs))
        case NegativeBinomial2LawSpec():
            return NegativeBinomial2LawSpec(
                mean=function(law.mean), concentration=function(law.concentration)
            )
        case BetaLawSpec():
            return BetaLawSpec(
                concentration1=function(law.concentration1),
                concentration0=function(law.concentration0),
            )
        case OrderedLogisticLawSpec():
            return OrderedLogisticLawSpec(
                predictor=function(law.predictor), cutpoints=function(law.cutpoints)
            )
        case CategoricalLawSpec():
            return CategoricalLawSpec(logits=function(law.logits))
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
    def parsed(self) -> LikelihoodAnalysis:
        """Retain the supported expression grammar at its scientific owner boundary."""
        from nof1_causal_lab.models.likelihoods import likelihood_terms

        return likelihood_terms(self.law)
