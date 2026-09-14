"""Execution findings derived from the selected scientific model."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class StructuralDisposition(StrEnum):
    """A structural disposition classifies how compilation uses or excludes an authored model
    entity.
    """

    RETAINED_STATE = "retained_state"
    KNOWN_INPUT = "known_input"
    MARGINALIZED = "marginalized"
    IDENTIFICATION_ONLY = "identification_only"
    RETAINED_EDGE = "retained_edge"
    PROJECTED_EDGE = "projected_edge"
    MANIFEST = "manifest"
    KNOWN_INPUT_SOURCE = "known_input_source"
    EXCLUDED_INDICATOR = "excluded_indicator"


class StructuralItemDisposition(BaseModel):
    """An item disposition explains the compilation decision for one identified authored
    entity.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    source_kind: Literal["construct", "edge", "indicator"]
    disposition: StructuralDisposition
    reason: str

    @model_validator(mode="after")
    def validate_owner_kind(self) -> StructuralItemDisposition:
        allowed = {
            "construct": {
                StructuralDisposition.RETAINED_STATE,
                StructuralDisposition.KNOWN_INPUT,
                StructuralDisposition.MARGINALIZED,
                StructuralDisposition.IDENTIFICATION_ONLY,
            },
            "edge": {StructuralDisposition.RETAINED_EDGE, StructuralDisposition.PROJECTED_EDGE},
            "indicator": {
                StructuralDisposition.MANIFEST,
                StructuralDisposition.KNOWN_INPUT_SOURCE,
                StructuralDisposition.EXCLUDED_INDICATOR,
            },
        }
        if self.disposition not in allowed[self.source_kind]:
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


class ExecutionReadiness(BaseModel):
    """Current model requirements and latent anchors, computed without a stored receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    unmet_requirements: tuple[str, ...] = ()
    anchor_certificates: tuple[AnchorCertificate, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.unmet_requirements

    @model_validator(mode="after")
    def validate_findings(self) -> ExecutionReadiness:
        if self.unmet_requirements and self.anchor_certificates:
            raise ValueError("Incomplete execution checks cannot certify latent anchors")
        return self
