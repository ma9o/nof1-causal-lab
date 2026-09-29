"""Read saved simulation arrays without generating or persisting scientific results."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.machine.view_models import (
    SimulationTrajectories,
    SimulationTrajectoryBands,
    TrajectorySummary,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.machine.store import ArtifactStore


def simulation_time_origin(store: ArtifactStore, report: SimulationReport) -> datetime | None:
    """Resolve model day zero from conditioning provenance, never the current panel."""
    import polars as pl

    from nof1_causal_lab.actions.data_checks import read_data_metadata
    from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
    from nof1_causal_lab.machine.store import read_model

    revision = report.law.fitted_panel_revision if report.law is not None else None
    if revision is None:
        return None
    metadata = read_data_metadata(store, revision)
    if isinstance(metadata.source, SimulationReplicateRef):
        # Simulation imports use a synthetic epoch, not a known calendar origin.
        return None
    fitted = store.read_meta("model", report.model.revision)
    while fitted.produced_by != "run:posterior" or fitted.derived_from.get("panel") != revision:
        parent = fitted.derived_from.get("model")
        if parent is None:
            raise ValueError("Saved simulation conditioning provenance has no fitted model")
        fitted = store.read_meta("model", parent)
    # Fitting keeps anchors only for indicators in its own model. Extra panel
    # variables and later measurement edits must not shift this calendar origin.
    indicators = [indicator.id for indicator in read_model(store, fitted.revision).indicators]
    panel = store.read_parquet_file("panel", revision, "panel.parquet")
    anchors = panel.filter(pl.col("indicator_id").is_in(indicators))["anchor_time"]
    if anchors.dtype == pl.String:
        anchors = anchors.str.to_datetime(strict=False, time_zone="UTC")
    origin = anchors.min()
    if not isinstance(origin, datetime):
        raise ValueError("A fitted calendar panel must have datetime observation anchors")
    return origin.replace(tzinfo=UTC) if origin.tzinfo is None else origin.astimezone(UTC)


def _summarize(values: np.ndarray, mask: np.ndarray) -> TrajectorySummary:
    if not np.isfinite(values[mask]).all():
        raise ValueError("Saved simulation has non-finite values at observed times")
    means, lower, upper = [], [], []
    for samples, present in zip(values.T, mask.T, strict=True):
        selected = samples[present]
        if selected.size:
            bounds = np.quantile(selected, [0.025, 0.975])
            means.append(float(np.mean(selected)))
            lower.append(float(bounds[0]))
            upper.append(float(bounds[1]))
        else:
            means.append(None)
            lower.append(None)
            upper.append(None)
    return TrajectorySummary(
        mean=tuple(means),
        lower=tuple(lower),
        upper=tuple(upper),
        n_draws=tuple(int(n) for n in mask.sum(axis=0)),
    )


def simulation_trajectories(
    model: ModelSpec,
    report: SimulationReport,
    *,
    read_array: Callable[[str], np.ndarray],
    time_origin: datetime | None = None,
) -> SimulationTrajectories:
    """Summarize exactly the histories and variable ordering retained by this simulation."""
    outcome = model.default_outcome
    if outcome is None:
        return SimulationTrajectories(
            times=report.times,
            time_origin=time_origin,
            outcome=None,
            outcome_state=None,
            indicators={},
        )
    construct = model.get_construct(outcome)
    layout = report.observation_layout
    state_shape = (report.draws, len(report.times), len(report.state_ids))
    observation_shape = (report.draws, len(report.times), len(layout.variables))

    def read(identity: str, shape: tuple[int, ...]) -> np.ndarray:
        values = read_array(identity)
        if values.shape != shape:
            raise ValueError("Saved simulation arrays do not match their recorded layout")
        return values

    action = read(report.latent_paths, state_shape)
    reference = (
        read(report.reference_latent_paths, state_shape)
        if report.reference_latent_paths is not None
        else None
    )
    observations = read(report.observations, observation_shape)
    reference_observations = (
        read(report.reference_observations, observation_shape)
        if report.reference_observations is not None
        else None
    )
    mask = read(layout.mask, observation_shape)
    if mask.dtype != np.bool_:
        raise ValueError("Saved simulation observation mask must be boolean")
    index = report.state_ids.index(outcome)
    state_mask = np.ones(state_shape[:2], dtype=bool)
    indicators = {indicator.id for indicator in construct.indicators}
    return SimulationTrajectories(
        times=report.times,
        time_origin=time_origin,
        outcome=outcome,
        outcome_state=SimulationTrajectoryBands(
            label=construct.name,
            action=_summarize(action[:, :, index], state_mask),
            reference=_summarize(reference[:, :, index], state_mask)
            if reference is not None
            else None,
        ),
        indicators={
            variable.id: SimulationTrajectoryBands(
                label=variable.name,
                action=_summarize(observations[:, :, i], mask[:, :, i]),
                reference=_summarize(reference_observations[:, :, i], mask[:, :, i])
                if reference_observations is not None
                else None,
            )
            for i, variable in enumerate(layout.variables)
            if variable.id in indicators
        },
    )
