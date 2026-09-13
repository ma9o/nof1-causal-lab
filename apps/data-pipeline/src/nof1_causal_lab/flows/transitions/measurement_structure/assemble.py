"""measurement-structure causal-design assembly."""

from nof1_causal_lab.artifacts.causal_design import (
    CausalDesign,
    IdentifiabilityStatus,
    IdentifiedTreatmentStatus,
    NonIdentifiableTreatmentStatus,
)
from nof1_causal_lab.artifacts.latent_structure import LatentStructure
from nof1_causal_lab.artifacts.measurement_structure import (
    KnownInput,
    MeasurementStructure,
    ScientificOnlyConstruct,
)
from nof1_causal_lab.json_types import UncheckedJsonObject


def build_causal_design(
    latent_structure: UncheckedJsonObject,
    measurement_structure: UncheckedJsonObject,
    identifiability_status: UncheckedJsonObject | None = None,
    *,
    known_inputs: list[UncheckedJsonObject],
    scientific_only_constructs: list[UncheckedJsonObject],
) -> CausalDesign:
    """Combine scientific and measurement semantics into a CausalDesign."""
    latent = LatentStructure.model_validate(latent_structure)
    by_name = {construct.name: construct.id for construct in latent.constructs}
    identification = None
    if identifiability_status is not None:
        # Symbolic identification uses graph names; persisted results use authored IDs.
        identification = IdentifiabilityStatus(
            identifiable_treatments={
                by_name[name]: IdentifiedTreatmentStatus(
                    method=result["method"],
                    estimand=result["estimand"],
                    marginalized_confounders=[
                        by_name[item] for item in result.get("marginalized_confounders", [])
                    ],
                    instruments=[by_name[item] for item in result.get("instruments", [])],
                )
                for name, result in identifiability_status["identifiable_treatments"].items()
            },
            non_identifiable_treatments={
                by_name[name]: NonIdentifiableTreatmentStatus(
                    confounders=[by_name[item] for item in result["confounders"]],
                    notes=result.get("notes"),
                )
                for name, result in identifiability_status["non_identifiable_treatments"].items()
            },
        )
    return CausalDesign(
        latent=latent,
        measurement=MeasurementStructure.model_validate(measurement_structure),
        identifiability=identification,
        known_inputs=[KnownInput.model_validate(item) for item in known_inputs],
        scientific_only_constructs=[
            ScientificOnlyConstruct.model_validate(item) for item in scientific_only_constructs
        ],
    )
