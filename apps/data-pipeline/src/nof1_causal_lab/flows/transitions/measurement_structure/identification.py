"""Positive identification-report derivation from the canonical causal results."""

from __future__ import annotations

import logging

from nof1_causal_lab.artifacts.causal_design import CausalDesign, IdentificationReport

logger = logging.getLogger(__name__)


def derive_identification_report(causal_design: CausalDesign) -> IdentificationReport | None:
    """Emit a positive gate only when the causal design identifies a treatment effect."""
    status = causal_design.identifiability
    if status is None or not status.identifiable_treatments:
        logger.warning("No estimable intervention targets remain; identification_report withheld")
        return None
    constructs = {construct.id: construct for construct in causal_design.latent.constructs}
    target = causal_design.latent.default_outcome
    if target is None:
        raise ValueError("Identification requires a selected default outcome")
    outcome = constructs[target.id]
    for treatment_id, result in status.non_identifiable_treatments.items():
        blockers = ", ".join(constructs[cid].name for cid in result.confounders)
        logger.warning(
            "Excluded non-identifiable effect: %s -> %s (%s)",
            constructs[treatment_id].name,
            outcome.name,
            blockers or result.notes or "not identified",
        )
    return IdentificationReport(
        outcome_id=outcome.id,
        estimable_treatments=list(status.identifiable_treatments),
        non_identifiable_treatments=status.non_identifiable_treatments,
    )
