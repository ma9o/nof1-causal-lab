"""A scientific quantity owns a value or participates in a probability law."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from .identity import DistributionId, ParameterId  # noqa: TC001
from .parameter import PriorAuthoringTransform


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
