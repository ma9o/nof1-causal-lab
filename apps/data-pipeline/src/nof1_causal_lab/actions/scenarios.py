"""Certify and summarize paired histories without running another simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.scenarios import CausalEffectResult
from nof1_causal_lab.models.causal_proofs import (
    CertifiedCausalAnalysis,
    certify_identified_estimand,
)
from nof1_causal_lab.models.identification import identify_model
from nof1_causal_lab.models.ssm.counterfactual.estimands import summarize_draws

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord


def summarize_causal_simulation(
    model: ModelSpec,
    report: SimulationReport,
    *,
    store: ArtifactStore,
    inference: TransitionRecord | None,
) -> SimulationReport:
    """Report numeric causal effects only when identification and exact-fit evidence support them."""
    if not report.design.interventions:
        return report
    if inference is None or model.default_outcome is None:
        return report.model_copy(
            update={
                "causal_unavailable_reason": "Causal effects require an identified model outcome and a committed production fit for this model revision.",
            }
        )
    outcome = model.default_outcome
    try:
        identification = identify_model(model)
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
    except ValueError as exc:
        return report.model_copy(update={"causal_unavailable_reason": str(exc)})
    assert report.reference_latent_paths is not None
    assert report.reference_observations is not None
    reference = store.read_array(report.reference_latent_paths)
    action = store.read_array(report.latent_paths)
    if not np.isfinite(reference).all() or not np.isfinite(action).all():
        return report.model_copy(
            update={
                "causal_unavailable_reason": "Non-finite histories do not support numeric causal effects."
            }
        )
    effects = (
        action[:, :, report.state_ids.index(outcome)]
        - reference[:, :, report.state_ids.index(outcome)]
    )
    days = np.asarray(report.times) - report.times[0]
    trajectory = [
        {"day": float(t), "effect": float(value)}
        for t, value in zip(days[1:], effects.mean(axis=0)[1:], strict=True)
    ]
    difference = store.read_array(report.observations) - store.read_array(
        report.reference_observations
    )
    manifest: dict[str, float] = {
        identity: float(difference[:, -1, i].mean())
        for i, identity in enumerate(report.observation_layout.indicator_ids)
        if np.isfinite(difference[:, -1, i]).all()
    }
    result = CausalEffectResult(
        outcome=outcome,
        time_grid_days=days.tolist(),
        labels={construct.id: construct.name for construct in model.constructs},
        summary=summarize_draws(effects[:, -1]),
        effect_trajectory=trajectory,
        trajectory_peak=max(trajectory, key=lambda point: abs(point["effect"])),
        trajectories={
            identity: {
                "reference_mean": reference[:, :, i].mean(axis=0).tolist(),
                "action_mean": action[:, :, i].mean(axis=0).tolist(),
            }
            for i, identity in enumerate(report.state_ids)
        },
        manifest_effects=manifest,
        reference_mean=float(reference[:, -1, report.state_ids.index(outcome)].mean()),
        warnings=[]
        if len(manifest) == len(report.observation_layout.indicator_ids)
        else [
            "Some indicator contrasts are unavailable because their measurement windows extend before simulation start."
        ],
    )
    return report.model_copy(update={"causal_result": result})
