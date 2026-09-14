"""Identify causal queries from one scientific model, retaining every finding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.identification import (
    IdentificationReport,
    IdentifiedTreatmentStatus,
    NonIdentifiableTreatmentStatus,
)
from nof1_causal_lab.models.model_inputs import identification_input
from nof1_causal_lab.utils.identifiability import check_identifiability

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def identify_model(model: ModelSpec) -> IdentificationReport:
    if model.default_outcome is None:
        return IdentificationReport(outcome=None)
    inputs = identification_input(model)
    # ModelSpec permits nonlinear dynamics; a linear-IV argument cannot establish
    # identification for this model, including during partial authoring.
    result = check_identifiability(inputs["graph"], inputs["observations"], iv_allowed=False)
    by_name = {construct.name: construct.id for construct in model.constructs}
    return IdentificationReport(
        outcome=model.default_outcome,
        treatments={
            **{
                by_name[name]: IdentifiedTreatmentStatus(
                    method=finding["method"],
                    estimand=finding["estimand"],
                    marginalized_confounders=[
                        by_name[item] for item in finding.get("marginalized_confounders", [])
                    ],
                    instruments=[by_name[item] for item in finding.get("instruments", [])],
                )
                for name, finding in result["identifiable_treatments"].items()
            },
            **{
                by_name[name]: NonIdentifiableTreatmentStatus(
                    confounders=[by_name[item] for item in finding["confounders"]],
                    notes=finding.get("notes"),
                )
                for name, finding in result["non_identifiable_treatments"].items()
            },
        },
    )
