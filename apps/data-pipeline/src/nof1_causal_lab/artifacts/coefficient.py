"""Scientific coefficients shared by model expressions and probability components."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from .identity import ParameterId  # noqa: TC001


class FixedCoefficient(BaseModel):
    """A coefficient held at a specified value on the continuous-time model scale."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    kind: Literal["fixed"] = "fixed"
    value: FiniteFloat


class ParameterCoefficient(BaseModel):
    """A slot referencing a scientific parameter, whether fixed or estimated."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    kind: Literal["parameter"] = "parameter"
    parameter_id: ParameterId


type Coefficient = Annotated[FixedCoefficient | ParameterCoefficient, Field(discriminator="kind")]
