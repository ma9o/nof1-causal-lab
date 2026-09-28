"""A model's generative observation definition and construct binding."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from nof1_causal_lab.distributions import VALID_LIKELIHOODS_FOR_DTYPE

from .likelihood import LikelihoodSpec  # noqa: TC001
from .observations import ObservationSpec


class IndicatorPolarity(StrEnum):
    """Indicator polarity states whether a measurement increases or decreases with its
    construct.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"  # noqa: V107 - native enum value construction


class IndicatorSpec(ObservationSpec):
    """Bind an observed-variable ID to a construct and an emission likelihood.

    Extraction instructions belong to DataPreparationSpec. The shared observation
    schema also permits generative models before any observations have been collected.
    """

    likelihood: LikelihoodSpec | None = None
    construct_polarity: IndicatorPolarity = Field(
        description="Whether higher values move with (positive) or against (negative) the construct."
    )

    @model_validator(mode="after")
    def validate_likelihood(self) -> IndicatorSpec:
        if (
            self.likelihood is not None
            and self.likelihood.law.family
            not in VALID_LIKELIHOODS_FOR_DTYPE[self.measurement_dtype]
        ):
            raise ValueError(
                f"Likelihood {self.likelihood.law.family.value!r} is incompatible with {self.measurement_dtype!r} indicator {self.id!r}"
            )
        return self
