"""Certified causal queries composed from the common trajectory simulation action."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from math import ceil
from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.scenarios import SimulationResult
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.models.causal_proofs import (
    CertifiedCausalAnalysis,
    certify_identified_estimand,
)
from nof1_causal_lab.models.identification import identify_model

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ModelRevision
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.simulation import CausalSimulationSpec, SimulationReport
    from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord


def simulate_causal(
    model: ModelSpec,
    design: CausalSimulationSpec,
    *,
    revision: ModelRevision,
    store: ArtifactStore,
    inference: TransitionRecord,
) -> SimulationReport:
    """Certify the selected immutable fit, then use paired nonlinear paths and emissions."""
    from nof1_causal_lab.actions.simulate import simulate
    from nof1_causal_lab.machine.artifact_files import parquet_filename
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.models.ssm.counterfactual.estimands import summarize_draws

    query = design.query
    identification = identify_model(model)
    outcome = model.get_construct(query.outcome).name
    treatments = sorted({model.get_construct(clamp.target).name for clamp in query.clamps})
    CertifiedCausalAnalysis(
        model=model,
        model_revision=revision,
        identification=identification,
        estimands=tuple(
            certify_identified_estimand(
                model, identification, model_revision=revision, treatment=treatment, outcome=outcome
            )
            for treatment in treatments
        ),
        inference=inference,
    )
    start_index, start_timestamp = None, None
    start = 0.0
    if query.start.kind == "abducted":
        start_index = query.start.time_index
        panel_version = inference.diagnostics["input_pins"]["panel"]
        panel = store.read_parquet_file("panel", panel_version, parquet_filename("panel", "panel"))
        anchors = [
            datetime.fromisoformat(str(t).replace("Z", "+00:00"))
            for t in panel["anchor_time"].to_list()
        ]
        origin = min(t.replace(tzinfo=UTC) if t.tzinfo is None else t for t in anchors)
        retained_times = [origin + timedelta(days=t) for t in model.time_points]
        if query.start.time is not None:
            requested = datetime.fromisoformat(query.start.time.replace("Z", "+00:00"))
            requested = requested.replace(tzinfo=UTC) if requested.tzinfo is None else requested
            if requested not in retained_times:
                raise ValueError("Start time must match a retained fitted-state timestamp")
            start_index = retained_times.index(requested)
        if start_index is None:
            start_index = len(model.time_points) - 1
        if start_index >= len(model.time_points):
            raise ValueError("Start index exceeds the retained fitted-state grid")
        start = model.time_points[start_index]
        start_timestamp = retained_times[start_index].isoformat()
    horizon = query.readout.horizon_days
    relative_grid = sorted(
        {
            *np.linspace(0, horizon, max(1, ceil(horizon / model.model_clock_days)) + 1).tolist(),
            *[
                float(day)
                for clamp in query.clamps
                for day in (clamp.from_day, clamp.to_day)
                if day is not None
            ],
        }
    )
    trajectory = SimulationSpec(
        times=tuple(start + t for t in relative_grid),
        draws=design.draws,
        seed=design.seed,
        initial_state="equilibrium" if query.start.kind == "baseline" else "retained",
        state_time=start if query.start.kind == "abducted" else None,
        process_noise=design.process_noise,
        observation_noise=design.observation_noise,
        interventions=tuple(query.clamps),
        context="prediction",
        checks=(),
    )
    report = simulate(model, trajectory, revision=revision, write_array=store.write_array)
    assert report.reference_latent_paths is not None
    assert report.reference_observations is not None
    reference = store.read_array(report.reference_latent_paths)
    action = store.read_array(report.latent_paths)
    if not np.isfinite(reference).all() or not np.isfinite(action).all():
        raise ValueError(
            "Causal simulation produced non-finite paths; no numeric effect is reportable"
        )
    outcome_index = report.state_ids.index(query.outcome)
    effects = action[:, :, outcome_index] - reference[:, :, outcome_index]
    effect_trajectory = (
        [
            {"day": t, "effect": float(v)}
            for t, v in zip(relative_grid[1:], effects.mean(axis=0)[1:], strict=True)
        ]
        if query.readout.estimand == "trajectory"
        else None
    )
    manifest_effects = None
    warnings = (
        []
        if design.process_noise
        else ["Paths follow deterministic nonlinear drift; future process noise is disabled."]
    )
    if query.readout.projection in {"manifest", "both"}:
        observed_difference = store.read_array(report.observations) - store.read_array(
            report.reference_observations
        )
        manifest_effects = {
            name: float(observed_difference[:, -1, i].mean())
            for i, name in enumerate(numeric.observation_names(model))
            if np.isfinite(observed_difference[:, -1, i]).all()
        }
        if len(manifest_effects) != len(report.indicator_ids):
            warnings.append(
                "Some indicator contrasts are unavailable because their measurement windows extend before the simulation start."
            )
    result = SimulationResult(
        request=query,
        model=revision,
        time_grid_days=relative_grid,
        start_time_index=start_index,
        start_time=start_timestamp,
        labels={construct.id: construct.name for construct in model.constructs},
        summary=summarize_draws(effects[:, -1]),
        effect_trajectory=effect_trajectory,
        trajectory_peak=max(effect_trajectory, key=lambda point: abs(point["effect"]))
        if effect_trajectory
        else None,
        trajectories={
            identity: {
                "reference_mean": reference[:, :, i].mean(axis=0).tolist(),
                "action_mean": action[:, :, i].mean(axis=0).tolist(),
            }
            for i, identity in enumerate(report.state_ids)
        },
        manifest_effects=manifest_effects,
        reference_mean=float(reference[:, -1, outcome_index].mean()),
        warnings=warnings,
    )
    return report.model_copy(update={"design": design, "causal_result": result})
