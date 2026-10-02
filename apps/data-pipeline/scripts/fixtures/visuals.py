"""Recorded plot fixtures for the single comprehensive workbench story."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.actions.simulation_summaries import (
    paired_effect_trajectory,
    summarize_simulation,
)
from nof1_causal_lab.artifacts.effects import EffectSummary
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.study.visual_models import MechanismViewRequest
from nof1_causal_lab.study.visuals import (
    recorded_simulation_paths,
)

if TYPE_CHECKING:
    from nof1_causal_lab.study.snapshots import ModelReader


def workbench_visuals(reader: ModelReader, template):
    """Generate explicit illustrative draws, then use production projection code.

    The demonstration's archived fit has no original joint draws. Its observation
    history stays source grounded; new simulation paths are explicitly illustrative,
    and their summaries are derived from these paths, never inverted into fake draws.
    """
    model = reader.model
    assert model is not None
    assert reader.data_metadata is not None
    report = SimulationReport.model_validate(template["report"])
    rng = np.random.default_rng(41)
    time = np.asarray(report.times)
    phase = rng.uniform(-0.5, 0.5, (report.draws, 1, 1))
    modes = np.where(np.arange(report.draws) % 2, 1, -1)[:, None, None]
    dimensions = np.arange(len(report.state_ids))[None, None, :]
    reference = modes * (1.2 + np.sin(time[None, :, None] * 0.9 + phase + dimensions))
    differences = modes * 0.4 + 0.3 * (1 - np.exp(-time[None, :, None] / 2))
    action = reference + differences
    variables = report.observation_layout.variables
    observations = np.stack(
        [action[:, :, index % len(report.state_ids)] for index in range(len(variables))], axis=-1
    )
    reference_observations = np.stack(
        [reference[:, :, index % len(report.state_ids)] for index in range(len(variables))], axis=-1
    )
    mask = np.ones_like(observations, dtype=bool)
    mask[:, 2, :] = False
    arrays = {
        "action": action,
        "reference": reference,
        "observations": observations,
        "reference_observations": reference_observations,
        "mask": mask,
    }
    updates = {
        "latent_paths": "action",
        "reference_latent_paths": "reference",
        "observations": "observations",
        "reference_observations": "reference_observations",
        "observation_layout": report.observation_layout.revised(mask="mask"),
        "predictive": summarize_simulation(
            model,
            state_ids=report.state_ids,
            variables=variables,
            latent_paths=action,
            observations=observations,
            mask=mask,
            reference_latent_paths=reference,
            reference_observations=reference_observations,
            fit_reliability="converged",
        ),
    }
    if report.causal_result is not None:
        index = report.state_ids.index(report.causal_result.outcome)
        delta = action[:, :, index] - reference[:, :, index]
        final = delta[:, -1]
        trajectory = paired_effect_trajectory(tuple(float(value) for value in time), delta)
        updates["causal_result"] = report.causal_result.revised(
            summary=EffectSummary(
                mean=float(final.mean()),
                median=float(np.median(final)),
                lower_95=float(np.quantile(final, 0.025)),
                upper_95=float(np.quantile(final, 0.975)),
                prob_positive=float(np.mean(final > 0)),
            ),
            effect_trajectory=trajectory,
            trajectory_peak=None,
            reference_mean=float(reference[:, -1, index].mean()),
            manifest_effects=None,
        )
    report = report.revised(**updates)
    effect = None
    if report.causal_result is not None:
        effect = (
            report.causal_result.outcome,
            action[:, :, report.state_ids.index(report.causal_result.outcome)]
            - reference[:, :, report.state_ids.index(report.causal_result.outcome)],
        )
    paths = recorded_simulation_paths(
        report,
        action,
        arrays["observations"],
        mask,
        reference,
        reference_observations,
        effect,
        start=0,
    )
    observations = {}
    for variable in reader.data_metadata.value.variables:
        history = reader.observation_history(variable.id)
        assert history is not None
        observations[variable.id] = history.model_dump(mode="json")
    owners = [edge.id for edge in model.edges if edge.mechanisms] + [
        construct.id for construct in model.constructs if construct.dynamics
    ]
    return {
        "note": "Simulation paths are explicitly illustrative, not a production fit or scientific evidence. Summaries are derived from these exact paths.",
        "report": report.model_dump(mode="json"),
        "simulation": paths.model_dump(mode="json"),
        "observations": observations,
        "parameters": reader.parameter_draws().model_dump(mode="json"),
        "mechanisms": {
            owner: reader.mechanism_curves(MechanismViewRequest(owner_id=owner)).model_dump(
                mode="json"
            )
            for owner in owners
        },
    }
