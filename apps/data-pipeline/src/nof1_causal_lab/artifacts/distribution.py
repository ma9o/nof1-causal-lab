"""One NumPyro-validated distribution contract shared by all prior forms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, override

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from nof1_causal_lab.distributions import PriorDistributionFamily  # noqa: TC001

if TYPE_CHECKING:
    from numpyro.distributions import Distribution
    from pydantic import GetJsonSchemaHandler
    from pydantic.json_schema import JsonSchemaValue
    from pydantic_core import CoreSchema


class DistributionSpec(BaseModel):
    """A distribution family and its complete, jointly validated constructor arguments."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    distribution: PriorDistributionFamily
    params: dict[str, FiniteFloat] = Field(description="Arguments to the declared NumPyro family")

    def to_numpyro(self) -> Distribution:
        """Construct the native law, including NumPyro's argument validation."""
        from nof1_causal_lab.prior_distributions import distribution_from_params

        return distribution_from_params(self.distribution, self.params)

    @model_validator(mode="after")
    def validate_distribution(self) -> DistributionSpec:
        self.to_numpyro()
        return self

    @classmethod
    @override
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        from nof1_causal_lab.prior_distributions import distribution_schema_variants

        schema = handler(core_schema)
        # Constrain the family/arguments pair together, including in subclasses.
        # Branches are native constructor signatures, not extra domain models.
        schema["oneOf"] = distribution_schema_variants()
        return schema


class DistributionTransform(BaseModel):
    """One supported NumPyro transform, applied after the base distribution."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    kind: Literal["affine", "exp", "persistence_to_decay"]
    loc: float = 0.0
    scale: float = 1.0

    @model_validator(mode="after")
    def validate_transform(self) -> DistributionTransform:
        if self.kind == "exp" and (self.loc != 0.0 or self.scale != 1.0):
            raise ValueError("An exponential transform has no location or scale parameters")
        if self.kind == "persistence_to_decay" and (self.loc != 0.0 or self.scale <= 0.0):
            raise ValueError("A persistence transform requires a positive reference interval")
        if self.kind == "affine" and self.scale == 0.0:
            raise ValueError("An affine distribution transform requires a nonzero scale")
        return self


class CompiledDistribution(DistributionSpec):
    """A scalar distribution recipe; NumPyro owns its density and transforms."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transforms: list[DistributionTransform] = []
