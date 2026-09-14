"""Uncertainty owned by constructs, with explicit joint dependency coefficients."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from nof1_causal_lab.distributions import DistributionFamily

from .coefficient import Coefficient  # noqa: TC001
from .identity import ConstructId  # noqa: TC001


class StateCoupling(BaseModel):
    """A coefficient connecting an owned component to another construct."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    other_id: ConstructId
    coefficient: Coefficient


class InnovationSpec(BaseModel):
    """Continuous-time driving noise, including conditional loadings from common causes."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    distribution: Literal[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T] = (
        DistributionFamily.GAUSSIAN
    )
    scale: Coefficient
    loadings: tuple[StateCoupling, ...] = ()
    degrees_of_freedom: Coefficient | None = None

    @model_validator(mode="after")
    def validate_tail_parameter(self) -> InnovationSpec:
        if (
            self.distribution != DistributionFamily.STUDENT_T
            and self.degrees_of_freedom is not None
        ):
            raise ValueError("Only Student-t innovations have a degrees-of-freedom coefficient")
        return self


class InitialStateSpec(BaseModel):
    """Initial location, marginal scale and correlations, including shared baseline factors."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")
    mean: Coefficient
    scale: Coefficient
    correlations: tuple[StateCoupling, ...] = ()
