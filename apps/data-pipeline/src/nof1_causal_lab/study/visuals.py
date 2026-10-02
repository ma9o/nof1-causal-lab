"""Lossless reads of recorded data and draws for the model workbench."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nof1_causal_lab.study.visual_models import (
    EmpiricalPoint,
    ObservationHistory,
    PathSeries,
    RecordedPath,
    SimulationPaths,
)
from nof1_causal_lab.utils.time_coordinates import ObservationInstant

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
    from nof1_causal_lab.artifacts.observations import ObservationSpec
    from nof1_causal_lab.artifacts.simulation import SimulationReport


def finite_values(values: np.ndarray) -> tuple[float | None, ...]:
    return tuple(float(value) if np.isfinite(value) else None for value in values)


def empirical_points(values: np.ndarray) -> tuple[EmpiricalPoint, ...]:
    unique, counts = np.unique(values[np.isfinite(values)], return_counts=True)
    cumulative = np.cumsum(counts) / counts.sum() if counts.size else []
    return tuple(
        EmpiricalPoint(value=value, probability=float(probability), count=int(count))
        for value, probability, count in zip(unique, cumulative, counts, strict=True)
    )


def observation_history(
    metadata: PreparedDataMetadata, variable: ObservationSpec, panel: pl.DataFrame
) -> ObservationHistory:
    origin = ObservationInstant.origin(metadata.time_origin)

    def days(column: str) -> tuple[float | None, ...]:
        return tuple(
            ObservationInstant(value).relative_to(origin).days if value is not None else None
            for value in panel[column]
        )

    values = panel["value"].cast(pl.Float64).to_numpy()
    return ObservationHistory(
        indicator_id=variable.id,
        label=variable.name,
        times=tuple(
            ObservationInstant(value).relative_to(origin).days for value in panel["anchor_time"]
        ),
        values=finite_values(values),
        support_start=days("support_start"),
        support_end=days("support_end"),
        time_origin=metadata.time_origin,
        levels=variable.ordinal_levels or variable.categorical_levels,
        empirical=empirical_points(values),
    )


def recorded_simulation_paths(
    report: SimulationReport,
    latent: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    reference: np.ndarray | None,
    reference_observed: np.ndarray | None,
    effect: tuple[str, np.ndarray] | None,
    *,
    start: int,
) -> SimulationPaths:
    """Project materialized paths without reducing their histories."""

    def paths(values: np.ndarray) -> tuple[RecordedPath, ...]:
        return tuple(
            RecordedPath(draw=start + index, values=finite_values(row))
            for index, row in enumerate(values)
        )

    return SimulationPaths(
        times=report.times,
        time_origin=report.time_origin,
        total_draws=report.draws,
        start=start,
        count=len(latent),
        states={
            identity: PathSeries(
                label=report.predictive.states[identity].label,
                action=paths(latent[:, :, index]),
                reference=paths(reference[:, :, index]) if reference is not None else (),
            )
            for index, identity in enumerate(report.state_ids)
        },
        indicators={
            variable.id: PathSeries(
                label=variable.name,
                action=paths(np.where(mask[:, :, index], observed[:, :, index], np.nan)),
                reference=paths(
                    np.where(mask[:, :, index], reference_observed[:, :, index], np.nan)
                )
                if reference_observed is not None
                else (),
                levels=variable.ordinal_levels or variable.categorical_levels,
            )
            for index, variable in enumerate(report.observation_layout.variables)
        },
        effect=PathSeries(label=effect[0], action=paths(effect[1])) if effect is not None else None,
    )
