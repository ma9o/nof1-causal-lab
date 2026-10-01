"""An uncertain scientific quantity participates in a probability law."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .identity import DistributionId, ParameterId
from .parameter import PriorAuthoringTransform


class ParameterSpec(BaseModel):
    """A named uncertain quantity; fixed coefficients are literals in component slots."""

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
    ] = Field(
        default=PriorAuthoringTransform.IDENTITY,
        description=(
            "Mapping from the authored law to the model quantity: identity leaves its scale "
            "unchanged; dt_persistence_to_ct_decay maps persistence p to -log(p) / interval; "
            "dt_effect_to_ct_rate divides an interval effect by its duration in days; "
            "initial_state_correlation applies the correlation support [-1, 1]. "
            "Joint laws require identity."
        ),
    )
    distribution: DistributionId | None = Field(
        default=None,
        description=(
            "Membership in a native law in ModelSpec.distributions; may be joint. "
            "None means the law has not been assigned yet."
        ),
    )
    reference_interval_days: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Positive duration in days over which an authored persistence or interval-effect "
            "law is defined, before conversion to continuous-time decay or rate. "
            "When omitted, persistence and interval effects use the model measurement clock."
        ),
    )
