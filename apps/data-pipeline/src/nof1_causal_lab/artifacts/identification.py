"""Identification findings about an explicit scientific model and query."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base import ArtifactPayload
from .identity import ConstructId  # noqa: TC001

if TYPE_CHECKING:
    from .model_spec import ModelSpec


class IdentifiedTreatmentStatus(BaseModel):
    """Details on how a treatment effect is identified."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["identified"] = "identified"
    method: Literal["do_calculus"] = Field(
        description="Nonparametric identification; linear-IV arguments do not certify ModelSpec."
    )
    estimand: str = Field(description="Nonparametric estimand returned by do-calculus")
    marginalized_confounders: list[ConstructId] = Field(
        default_factory=list,
        description="Unobserved confounders the estimand integrates out",
    )
    instruments: list[ConstructId] = Field(
        default_factory=list,
        description="Instrument constructs appearing in the nonparametric identification argument",
    )


class NonIdentifiableTreatmentStatus(BaseModel):
    """Context on why a treatment effect is not identifiable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["not_identified"] = "not_identified"
    confounders: list[ConstructId] = Field(
        default_factory=list,
        description="Unobserved constructs blocking identification",
    )
    notes: str | None = Field(
        default=None,
        description="Optional explanation if confounders cannot be enumerated",
    )


class IdentifiabilityStatus(BaseModel):
    """Status of causal effect identifiability."""

    identifiable_treatments: dict[ConstructId, IdentifiedTreatmentStatus] = Field(
        default_factory=dict,
        description="Treatment IDs mapped to their identification strategy and supporting construct IDs",
    )
    non_identifiable_treatments: dict[ConstructId, NonIdentifiableTreatmentStatus] = Field(
        default_factory=dict,
        description="Treatment IDs mapped to the construct IDs blocking identification",
    )

    @model_validator(mode="after")
    def validate_exclusive_results(self) -> IdentifiabilityStatus:
        if self.identifiable_treatments.keys() & self.non_identifiable_treatments.keys():
            raise ValueError("A treatment cannot be both identified and non-identifiable")
        return self


class IdentificationReport(ArtifactPayload):
    """Positive and negative causal identification findings for the model's default query."""

    outcome: ConstructId | None
    status: IdentifiabilityStatus

    def validate_model(self, model: ModelSpec) -> None:
        owners = set(self.status.identifiable_treatments) | set(
            self.status.non_identifiable_treatments
        )
        if self.outcome is not None:
            owners.add(self.outcome)
        for finding in self.status.identifiable_treatments.values():
            owners.update(finding.marginalized_confounders)
            owners.update(finding.instruments)
        for finding in self.status.non_identifiable_treatments.values():
            owners.update(finding.confounders)
        unknown = owners - {construct.id for construct in model.constructs}
        if unknown:
            raise ValueError(f"Identification references unknown construct IDs: {sorted(unknown)}")

    @property
    def estimable_treatments(self) -> list[ConstructId]:
        return list(self.status.identifiable_treatments)

    @property
    def non_identifiable(self) -> dict[ConstructId, NonIdentifiableTreatmentStatus]:
        return self.status.non_identifiable_treatments
