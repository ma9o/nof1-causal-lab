"""Execution findings derived from the selected scientific model."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from nof1_causal_lab.artifacts.base import Value

from .identity import ConstructRef, EdgeRef, IndicatorRef


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


class StructuralItemDisposition(Value):
    """An item disposition explains the compilation decision for one identified authored
    entity.
    """

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
