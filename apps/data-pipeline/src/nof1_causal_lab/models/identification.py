"""Identify causal queries from one scientific model, retaining every finding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.identification import (
    IdentificationReport,
    IdentifiedTreatmentStatus,
    NonIdentifiableTreatmentStatus,
)
from nof1_causal_lab.utils.identifiability import check_identifiability, get_observed_constructs

if TYPE_CHECKING:
    from nof1_causal_lab.models.model_structure import StructuralSelection


def identify_model(selection: StructuralSelection) -> IdentificationReport:
    """Identify each treatment's effect on the outcome that scopes the selection."""
    model, outcome = selection.model, selection.outcome
    if outcome is None:
        return IdentificationReport(outcome=None)
    # ModelSpec permits nonlinear dynamics; a linear-IV argument cannot establish
    # identification for this model, including during partial authoring.
    result = check_identifiability(
        model.constructs,
        model.edges,
        outcome_id=outcome,
        observed_constructs=get_observed_constructs(model.constructs),
        iv_allowed=False,
    )
    by_name = {construct.name: construct.id for construct in model.constructs}
    return IdentificationReport(
        outcome=outcome,
        treatments={
            **{
                by_name[name]: IdentifiedTreatmentStatus(
                    estimand=finding.estimand,
                    marginalized_confounders=tuple(
                        by_name[item] for item in finding.marginalized_confounders
                    ),
                    instruments=tuple(by_name[item] for item in finding.instruments),
                )
                for name, finding in result.identifiable_treatments.items()
            },
            **{
                by_name[name]: NonIdentifiableTreatmentStatus(
                    confounders=tuple(by_name[item] for item in finding.confounders),
                    notes=finding.notes,
                )
                for name, finding in result.non_identifiable_treatments.items()
            },
        },
    )
