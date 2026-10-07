"""Certify and summarize paired histories without running another simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.display_frames import central_frame
from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.scenarios import CausalEffectResult
from nof1_causal_lab.artifacts.simulation import PairedArmSimulation
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
    import jax.numpy as jnp

    from nof1_causal_lab.models.ssm.counterfactual.estimands import summarize_draws
    from nof1_causal_lab.study.lineage import fitted_law_report
    from nof1_causal_lab.study.store import read_model

    model, outcome = selection.model, selection.outcome
    arms = report.evidence.arms
    if not isinstance(arms, PairedArmSimulation):
        return report
    if inference is None or outcome is None or outcome not in report.evidence.state_ids:
        return report.without_causal_result(
            "Causal effects require the question's outcome as a model state and a committed production fit for this model revision."
        )
    try:
        identification = selection.identification
        CertifiedCausalAnalysis(
            model=model,
            model_revision=report.evidence.model,
            identification=identification,
            estimands=tuple(
                certify_identified_estimand(
                    model,
                    identification,
                    model_revision=report.evidence.model,
                    treatment=model.get_construct(target).name,
                    outcome=model.get_construct(outcome).name,
                )
                for target in sorted(
                    {event.target for event in report.evidence.design.interventions}
                )
            ),
            inference=inference,
            fitted_model=read_model(store, report.evidence.model.revision),
            report=fitted_law_report(store, [inference], report.evidence.model.revision),
        )
    except CausalCertificationError as exc:
        return report.without_causal_result(str(exc))
    reference = arms.reference.latent_paths.values
    action = arms.action.latent_paths.values
    if not np.isfinite(reference).all() or not np.isfinite(action).all():
        return report.without_causal_result(
            "Non-finite histories do not support numeric causal effects."
        )
    outcome_index = report.evidence.state_ids.index(outcome)
    differences = action[:, :, outcome_index] - reference[:, :, outcome_index]
    frame = central_frame(differences)
    assert frame is not None, "Finite paired histories have a display range"
    observed = arms.action.observations.values
    reference_observed = arms.reference.observations.values
    manifest_differences = observed[:, -1] - reference_observed[:, -1]
    result = CausalEffectResult(
        outcome=outcome,
        labels={construct.id: construct.name for construct in model.constructs},
        differences=NumericalArray.from_numpy(differences),
        frame=frame,
        summary=summarize_draws(jnp.asarray(differences[:, -1])),
        reference_mean=float(reference[:, -1, outcome_index].mean()),
        manifest_effects={
            identity: float(manifest_differences[:, index].mean())
            for index, identity in enumerate(report.evidence.observation_layout.indicator_ids)
            if np.isfinite(manifest_differences[:, index]).all()
        },
        warnings=()
        if np.isfinite(manifest_differences).all()
        else (
            "Some indicator contrasts are unavailable because their measurement windows extend before simulation start.",
        ),
    )
    return report.with_causal_result(result)
