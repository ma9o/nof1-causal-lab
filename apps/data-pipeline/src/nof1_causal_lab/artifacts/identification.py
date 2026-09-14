"""Identification findings about an explicit scientific model and query."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

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


class IdentificationReport(ArtifactPayload):
    """Positive and negative causal identification findings for the model's default query."""

    outcome: ConstructId | None
    treatments: dict[
        ConstructId,
        Annotated[
            IdentifiedTreatmentStatus | NonIdentifiableTreatmentStatus,
            Field(discriminator="status"),
        ],
    ] = Field(
        default_factory=dict,
        description="One tagged identification result per treatment, including its supporting evidence",
    )

    def validate_model(self, model: ModelSpec) -> None:
        owners = set(self.treatments)
        if self.outcome is not None:
            owners.add(self.outcome)
        for finding in self.treatments.values():
            if finding.status == "identified":
                owners.update(finding.marginalized_confounders)
                owners.update(finding.instruments)
            else:
                owners.update(finding.confounders)
        unknown = owners - {construct.id for construct in model.constructs}
        if unknown:
            raise ValueError(f"Identification references unknown construct IDs: {sorted(unknown)}")

    @property
    def estimable_treatments(self) -> list[ConstructId]:
        return [
            identity
            for identity, finding in self.treatments.items()
            if finding.status == "identified"
        ]

    @property
    def non_identifiable(self) -> dict[ConstructId, NonIdentifiableTreatmentStatus]:
        return {
            identity: finding
            for identity, finding in self.treatments.items()
            if finding.status == "not_identified"
        }
