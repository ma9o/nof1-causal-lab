"""Composite causal-design artifact models and validation."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .identity import ConstructId  # noqa: TC001
from .latent_structure import LatentStructure  # noqa: TC001
from .measurement_structure import (  # noqa: TC001
    KnownInput,
    MeasurementStructure,
    ScientificOnlyConstruct,
)


class IdentifiedTreatmentStatus(BaseModel):
    """Details on how a treatment effect is identified."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["identified"] = "identified"
    method: str = Field(
        description="Identification strategy (e.g., do_calculus, instrumental_variable)"
    )
    estimand: str = Field(description="Closed-form estimand or IV placeholder")
    marginalized_confounders: list[ConstructId] = Field(
        default_factory=list,
        description="Unobserved confounders the estimand integrates out",
    )
    instruments: list[ConstructId] = Field(
        default_factory=list,
        description="Instrumental variables used (if method=instrumental_variable)",
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


type TreatmentIdentification = Annotated[
    IdentifiedTreatmentStatus | NonIdentifiableTreatmentStatus, Field(discriminator="status")
]


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


class CausalDesign(BaseModel):
    """Scientific causal design before executable structural compilation."""

    latent: LatentStructure = Field(description="Theoretical causal structure (topological)")
    measurement: MeasurementStructure = Field(description="Operationalization into indicators")
    identifiability: IdentifiabilityStatus | None = Field(
        default=None, description="Identifiability status of target causal effects"
    )
    known_inputs: list[KnownInput] = Field(
        default_factory=list,
        description="Authored observed-input declarations compiled by StructuralPlan",
    )
    scientific_only_constructs: list[ScientificOnlyConstruct] = Field(
        default_factory=list,
        description="Measured constructs explicitly excluded from the executable SSM",
    )

    @model_validator(mode="after")
    def validate_causal_design(self) -> CausalDesign:
        """Validate authored measurement and executable-disposition declarations."""
        constructs = {construct.id: construct for construct in self.latent.constructs}
        construct_ids = set(constructs)
        if self.identifiability is not None:
            status = self.identifiability
            references = set(status.identifiable_treatments) | set(
                status.non_identifiable_treatments
            )
            for identified in status.identifiable_treatments.values():
                references.update(identified.marginalized_confounders)
                references.update(identified.instruments)
            for blocked in status.non_identifiable_treatments.values():
                references.update(blocked.confounders)
            if unknown := references - construct_ids:
                raise ValueError(
                    f"Identification references unknown construct IDs: {sorted(unknown)}"
                )
        for indicator in self.measurement.indicators:
            if indicator.construct_id not in constructs:
                raise ValueError(
                    f"Indicator '{indicator.name}' references unknown construct '{indicator.construct_id}'"
                )

        indicator_lookup = {indicator.id: indicator for indicator in self.measurement.indicators}
        known_input_ids = {known_input.construct_id for known_input in self.known_inputs}
        if len(known_input_ids) != len(self.known_inputs):
            raise ValueError("CausalDesign known_inputs contains duplicate constructs")
        scientific_only_ids = {item.construct_id for item in self.scientific_only_constructs}
        if len(scientific_only_ids) != len(self.scientific_only_constructs):
            raise ValueError(
                "CausalDesign scientific_only_constructs contains duplicate constructs"
            )
        overlap = known_input_ids & scientific_only_ids
        if overlap:
            raise ValueError(
                "CausalDesign constructs cannot be both known inputs and scientific-only: "
                f"{sorted(overlap)}"
            )
        unknown_scientific_only = scientific_only_ids - construct_ids
        if unknown_scientific_only:
            raise ValueError(
                "CausalDesign scientific_only_constructs reference unknown constructs: "
                f"{sorted(unknown_scientific_only)}"
            )
        observed_construct_ids = {
            indicator.construct_id for indicator in self.measurement.indicators
        }
        unmeasured_scientific_only = scientific_only_ids - observed_construct_ids
        if unmeasured_scientific_only:
            raise ValueError(
                "CausalDesign scientific_only_constructs must have measurement evidence: "
                f"{sorted(unmeasured_scientific_only)}"
            )
        for known_input in self.known_inputs:
            if known_input.construct_id not in construct_ids:
                raise ValueError(
                    "CausalDesign known_input references unknown construct: "
                    f"{known_input.construct_id!r}"
                )
            source_indicator = indicator_lookup.get(known_input.source_indicator_id)
            if source_indicator is None:
                raise ValueError(
                    "CausalDesign known_input references unknown source_indicator: "
                    f"{known_input.source_indicator_id!r}"
                )
            if source_indicator.construct_id != known_input.construct_id:
                raise ValueError(
                    "CausalDesign known_input source_indicator must measure the same "
                    f"construct: {known_input.source_indicator_id!r} measures "
                    f"{source_indicator.construct_id!r}, expected "
                    f"{known_input.construct_id!r}"
                )

        return self


__all__ = [
    "CausalDesign",
    "IdentifiabilityStatus",
    "IdentifiedTreatmentStatus",
    "NonIdentifiableTreatmentStatus",
    "TreatmentIdentification",
]


class IdentificationReport(BaseModel):
    """The positive identification finding.

    Only produced when at least one treatment effect is explicitly
    identifiable. Negative findings remain in ``causal_design.identifiability``.
    """

    model_config = ConfigDict(extra="forbid")

    outcome_id: ConstructId
    estimable_treatments: list[ConstructId] = Field(min_length=1)
    non_identifiable_treatments: dict[ConstructId, NonIdentifiableTreatmentStatus] = Field(
        default_factory=dict
    )


class CausalDesignArtifact(BaseModel):
    """The causal-design file stores the derived scientific design."""

    model_config = ConfigDict(extra="forbid")
    causal_design: CausalDesign
