"""A conditional probability law expressed over scientific states and coefficients."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nof1_causal_lab.distributions import (
    OBSERVATION_LINK_VALUES_BY_DISTRIBUTION,
    DistributionFamily,
)

from .evidence import LiteratureSource  # noqa: TC001
from .expressions import Expression  # noqa: TC001

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


VALID_LINKS_FOR_DISTRIBUTION = {
    family: {LinkFunction(link) for link in links}
    for family, links in OBSERVATION_LINK_VALUES_BY_DISTRIBUTION.items()
}

# Native constructor signatures define the bounded observation grammar. Argument
# values are expressions; their meaning is never repeated as independent link tags.
OBSERVATION_CONSTRUCTORS = {
    "Delta": (DistributionFamily.DELTA, ({"v"},)),
    "Normal": (DistributionFamily.GAUSSIAN, ({"loc", "scale"},)),
    "StudentT": (DistributionFamily.STUDENT_T, ({"df", "loc", "scale"},)),
    "Poisson": (DistributionFamily.POISSON, ({"rate"},)),
    "Gamma": (DistributionFamily.GAMMA, ({"concentration", "rate"},)),
    "Bernoulli": (DistributionFamily.BERNOULLI, ({"logits"}, {"probs"})),
    "NegativeBinomial2": (DistributionFamily.NEGATIVE_BINOMIAL, ({"mean", "concentration"},)),
    "Beta": (DistributionFamily.BETA, ({"concentration1", "concentration0"},)),
    "OrderedLogistic": (DistributionFamily.ORDERED_LOGISTIC, ({"predictor", "cutpoints"},)),
    "Categorical": (DistributionFamily.CATEGORICAL, ({"logits"},)),
}


class ObservationLaw(BaseModel):
    """A native probability constructor applied to model-dependent expressions."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    distribution: Literal[
        "Delta",
        "Normal",
        "StudentT",
        "Poisson",
        "Gamma",
        "Bernoulli",
        "NegativeBinomial2",
        "Beta",
        "OrderedLogistic",
        "Categorical",
    ]
    arguments: dict[str, Expression]

    @model_validator(mode="after")
    def validate_constructor(self) -> ObservationLaw:
        _, signatures = OBSERVATION_CONSTRUCTORS[self.distribution]
        if set(self.arguments) not in signatures:
            expected = " or ".join(str(sorted(signature)) for signature in signatures)
            raise ValueError(f"{self.distribution} requires exactly {expected}")
        return self

    @property
    def family(self) -> DistributionFamily:
        return OBSERVATION_CONSTRUCTORS[self.distribution][0]


class LikelihoodSpec(BaseModel):
    """An indicator's conditional probability law and its scientific justification."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    law: ObservationLaw
    standardized: bool = Field(
        default=False,
        description="Whether observations are mean-centered and scaled before fitting.",
    )
    reasoning: str = Field(description="Why this conditional law was chosen for the indicator")
    sources: tuple[LiteratureSource, ...] = ()

    @property
    def terms(self) -> LikelihoodTerms:
        """Ephemeral lowering of the expression into the supported numerical operations."""
        from nof1_causal_lab.models.likelihoods import likelihood_terms

        return likelihood_terms(self.law)

    @model_validator(mode="after")
    def validate_expression(self) -> LikelihoodSpec:
        # Valid partial laws contain explicit unassigned coefficient operands.
        # Unsupported formulas fail at this boundary rather than changing the fit.
        _ = self.terms
        return self
