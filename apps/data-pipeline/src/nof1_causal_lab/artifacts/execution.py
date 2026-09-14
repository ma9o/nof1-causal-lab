"""Execution findings derived from the selected scientific model."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .identity import ConstructRef, EdgeRef, IndicatorRef  # noqa: TC001


class StructuralDisposition(StrEnum):
    """A structural disposition classifies how compilation uses or excludes an authored model
    entity.
    """

    RETAINED_STATE = "retained_state"
    MARGINALIZED = "marginalized"
    IDENTIFICATION_ONLY = "identification_only"
    RETAINED_EDGE = "retained_edge"
    PROJECTED_EDGE = "projected_edge"
    MANIFEST = "manifest"
    EXCLUDED_INDICATOR = "excluded_indicator"
    UNSUPPORTED = "unsupported"


class StructuralItemDisposition(BaseModel):
    """An item disposition explains the compilation decision for one identified authored
    entity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target: ConstructRef | EdgeRef | IndicatorRef = Field(discriminator="kind")
    disposition: StructuralDisposition
    reason: str

    @model_validator(mode="after")
    def validate_owner_kind(self) -> StructuralItemDisposition:
        allowed = {
            "construct": {
                StructuralDisposition.RETAINED_STATE,
                StructuralDisposition.UNSUPPORTED,
                StructuralDisposition.MARGINALIZED,
                StructuralDisposition.IDENTIFICATION_ONLY,
            },
            "edge": {
                StructuralDisposition.RETAINED_EDGE,
                StructuralDisposition.PROJECTED_EDGE,
                StructuralDisposition.UNSUPPORTED,
            },
            "indicator": {
                StructuralDisposition.MANIFEST,
                StructuralDisposition.EXCLUDED_INDICATOR,
            },
        }
        if self.disposition not in allowed[self.target.kind]:
            raise ValueError("Structural disposition does not apply to its owner kind")
        return self


class AnchorCertificate(BaseModel):
    """Compiler proof that one retained latent has location and scale anchors."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    construct_id: str
    construct_name: str
    location_anchor: Literal[
        "standardized_manifest",
        "exact_state_observation",
        "fixed_dynamics_center",
        "fixed_initial_mean",
    ]
    location_source_id: str | None = None
    scale_anchor: Literal["fixed_manifest_loading", "categorical_slope_pin"]
    scale_source_id: str
