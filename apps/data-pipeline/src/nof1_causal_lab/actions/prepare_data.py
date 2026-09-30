"""Materialize recorded simulation observations without extraction."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nof1_causal_lab.utils.data import observation_row_schema
from nof1_causal_lab.utils.observation_rows import validate_observation_rows
from nof1_causal_lab.utils.time_coordinates import SYNTHETIC_EPOCH

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.measurements import ObservationRecord
    from nof1_causal_lab.artifacts.simulation import SimulationReport


def read_simulation_observations(
    report: SimulationReport,
    replicate: int,
    *,
    read_array: Callable[[str], np.ndarray],
) -> pl.DataFrame:
    """Read emitted observations on their recorded dates or absolute model days."""
    if not 0 <= replicate < report.draws:
        raise ValueError(f"Simulation replicate must be between 0 and {report.draws - 1}")
    times = np.asarray(report.times)
    layout = report.observation_layout
    starts = read_array(layout.support_start_times)
    ends = read_array(layout.support_end_times)
    mask = read_array(layout.mask)
    if (
        starts.shape != (len(times), len(report.observation_layout.indicator_ids))
        or ends.shape != starts.shape
    ):
        raise ValueError("Simulation support does not match the recorded observation layout")
    observations = read_array(report.observations)
    if observations.shape != (
        report.draws,
        len(times),
        len(report.observation_layout.indicator_ids),
    ):
        raise ValueError("Simulation observations do not match the recorded draws and design")
    values = observations[replicate]
    if mask.shape != observations.shape or mask.dtype != np.bool_:
        raise ValueError("Simulation observation mask must align with the recorded draws")
    observed = mask[replicate]
    if not np.isfinite(values[observed]).all():
        raise ValueError("Simulation replicate has non-finite emissions at observed times")
    if not observed.any() or np.isinf(values).any():
        raise ValueError("Simulation replicate contains infinite values or no observed values")
    if not (np.isfinite(starts[observed]).all() and np.isfinite(ends[observed]).all()):
        raise ValueError("Observed simulation values must have finite support boundaries")

    origin = report.time_origin if report.time_origin is not None else SYNTHETIC_EPOCH

    def _timestamp(day: float) -> str | None:
        if np.isnan(day):
            return None
        return (origin + timedelta(days=float(day))).isoformat(timespec="microseconds")

    rows: list[ObservationRecord] = [
        {
            "indicator_id": identity,
            "value": None if not observed[t, i] else float(values[t, i]),
            "anchor_time": _timestamp(time),
            "support_kind": layout.variables[i].support_kind.value,
            "summary_operator": layout.variables[i].summary_operator.value,
            "anchor_policy": layout.variables[i].anchor_policy.value,
            "observation_window": layout.variables[i].observation_window,
            "support_start": _timestamp(starts[t, i]),
            "support_end": _timestamp(ends[t, i]),
        }
        for t, time in enumerate(times)
        for i, identity in enumerate(report.observation_layout.indicator_ids)
    ]
    # Emissions already use the model's numeric codes, including unobserved
    # category levels. Extraction's label encoding must not run a second time.
    panel = (
        pl.DataFrame(rows, schema=observation_row_schema() | {"value": pl.Float64})
        .with_columns(
            pl.col("anchor_time", "support_start", "support_end")
            .str.to_datetime(format="%+")
            .dt.replace_time_zone(None)
        )
        .sort("indicator_id", "anchor_time")
    )
    return validate_observation_rows(panel, layout.variables)


def prepare_simulation_panel(
    report: SimulationReport,
    replicate: int,
    *,
    read_array: Callable[[str], np.ndarray],
) -> pl.DataFrame:
    """Materialize a replicate with day zero at its first time, preserving known dates."""
    panel = read_simulation_observations(report, replicate, read_array=read_array)
    if report.time_origin is None:
        panel = panel.with_columns(
            pl.col("anchor_time", "support_start", "support_end") - timedelta(days=report.times[0])
        )
    return panel
