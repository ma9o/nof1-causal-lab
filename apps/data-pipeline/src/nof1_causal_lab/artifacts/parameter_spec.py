"""A scientific quantity owns a value or participates in a probability law."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .identity import DistributionId, ParameterId  # noqa: TC001
from .parameter import PriorAuthoringTransform


class ParameterRole(StrEnum):
    """A parameter role identifies which part of the statistical model a parameter controls."""

    FIXED_EFFECT = "fixed_effect"
    AR_COEFFICIENT = "ar_coefficient"
    DYNAMICS_PARAMETER = "dynamics_parameter"  # noqa: V107 - public StrEnum value construction
    DYNAMICS_PARAMETER_POSITIVE = "dynamics_parameter_positive"
    RESIDUAL_SD = "residual_sd"
    STATE_INTERCEPT = "state_intercept"
    OBSERVATION_INTERCEPT = "observation_intercept"
    INITIAL_STATE_MEAN = "initial_state_mean"
    INITIAL_STATE_SD = "initial_state_sd"
    STATIC_STATE_SD = "static_state_sd"
    CORRELATION = "correlation"
    INITIAL_STATE_CORRELATION = "initial_state_correlation"
    LOADING = "loading"
    MEASUREMENT_ERROR_SD = "measurement_error_sd"
    OBSERVATION_HYPERPARAMETER = "observation_hyperparameter"  # noqa: V107 - public StrEnum value construction
    OBSERVATION_HYPERPARAMETER_POSITIVE = "observation_hyperparameter_positive"  # noqa: V107 - public StrEnum value construction


class ParameterConstraint(StrEnum):
    """A parameter constraint specifies the permitted range of a model parameter."""

    NONE = "none"  # noqa: V107 - consumed through StrEnum value construction
    POSITIVE = "positive"
    NEGATIVE = "negative"  # noqa: V107 - native enum value construction
    UNIT_INTERVAL = "unit_interval"  # noqa: V107 - native enum value construction
    CORRELATION = "correlation"


class ParameterSpec(BaseModel):
    """A named quantity's current uncertainty; component slots define its meaning."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    id: ParameterId
    name: str = Field(description="Authored parameter label; relationships use its persistent ID")
    description: str = Field(
        description="Human-readable description of what this parameter represents"
    )
    distribution_transform: Literal[
        PriorAuthoringTransform.IDENTITY,
        PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY,
        PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE,
        PriorAuthoringTransform.INITIAL_STATE_CORRELATION,
    ] = PriorAuthoringTransform.IDENTITY
    value: FiniteFloat | None = Field(
        default=None,
        description="Known constant on the model quantity scale, exclusive with a distribution.",
    )
    distribution: DistributionId | None = Field(
        default=None,
        description="Membership in a native law in ModelSpec.distributions; may be joint.",
    )
    reference_interval_days: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_definition(self) -> ParameterSpec:
        if self.value is not None and (
            self.distribution is not None
            or self.distribution_transform != PriorAuthoringTransform.IDENTITY
        ):
            raise ValueError(
                "A fixed quantity has a model-scale value and no distribution transformation or law"
            )
        return self
