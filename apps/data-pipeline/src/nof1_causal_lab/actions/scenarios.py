"""Certify and summarize paired histories without running another simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.scenarios import CausalEffectResult
from nof1_causal_lab.models.causal_proofs import (
    CausalCertificationError,
    CertifiedCausalAnalysis,
    certify_identified_estimand,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.study.records import StudyRevision
    from nof1_causal_lab.study.store import ArtifactStore


def summarize_causal_simulation(
    selection: StructuralSelection,
    report: SimulationReport,
    *,
    store: ArtifactStore,
    inference: StudyRevision | None,
) -> SimulationReport:
    """Report the question's numeric effect only when identification and exact-fit evidence support it."""
    model, outcome = selection.model, selection.outcome
    if not report.design.interventions:
        return report
    if inference is None or outcome is None or outcome not in report.state_ids:
        return report.without_causal_result(
            "Causal effects require the question's outcome as a model state and a committed production fit for this model revision."
        )
    try:
        identification = selection.identification
        CertifiedCausalAnalysis(
            model=model,
            model_revision=report.model,
            identification=identification,
            estimands=tuple(
                certify_identified_estimand(
                    model,
                    identification,
                    model_revision=report.model,
                    treatment=model.get_construct(target).name,
                    outcome=model.get_construct(outcome).name,
                )
                for target in sorted({event.target for event in report.design.interventions})
            ),
            inference=inference,
        )
    except CausalCertificationError as exc:
        return report.without_causal_result(str(exc))
    assert report.reference_latent_paths is not None
    assert report.reference_observations is not None
    reference = store.read_array(report.reference_latent_paths)
    action = store.read_array(report.latent_paths)
    if not np.isfinite(reference).all() or not np.isfinite(action).all():
        return report.without_causal_result(
            "Non-finite histories do not support numeric causal effects."
        )
    result = CausalEffectResult(
        outcome=outcome,
        labels={construct.id: construct.name for construct in model.constructs},
        warnings=()
        if np.isfinite(store.read_array(report.observations)[:, -1]).all()
        and np.isfinite(store.read_array(report.reference_observations)[:, -1]).all()
        else (
            "Some indicator contrasts are unavailable because their measurement windows extend before simulation start.",
        ),
    )
    return report.with_causal_result(result)
