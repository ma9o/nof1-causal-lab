"""Prepare generated or already extracted canonical observations without extraction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nof1_causal_lab.utils.aggregations import fill_null_expression
from nof1_causal_lab.utils.data import observation_row_schema, support_window_tick_frame

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.data_preparation import ObservationTableSpec
    from nof1_causal_lab.artifacts.measurements import ObservationRecord
    from nof1_causal_lab.artifacts.simulation import SimulationReport


def _observation_datetimes(data: pl.DataFrame, column: str) -> pl.DataFrame:
    """Normalize canonical string/date timestamps to UTC-naive panel datetimes."""
    dtype = data.schema[column]
    value = pl.col(column)
    if dtype == pl.String:
        value = value.str.to_datetime(time_zone="UTC")
    elif dtype == pl.Date:
        value = value.cast(pl.Datetime).dt.replace_time_zone("UTC")
    elif isinstance(dtype, pl.Datetime):
        value = (
            value.dt.convert_time_zone("UTC")
            if dtype.time_zone is not None
            else value.dt.replace_time_zone("UTC")
        )
    else:
        raise ValueError(f"Observation column {column!r} must contain ISO timestamps or datetimes")
    return data.with_columns(value.dt.replace_time_zone(None).alias(column))


def prepare_observation_panel(data: pl.DataFrame, definition: ObservationTableSpec) -> pl.DataFrame:
    """Select declared rows and reject incompatible values/support before saving any panel."""
    from nof1_causal_lab.artifacts.duration import parse_duration_to_hours

    columns = observation_row_schema()
    if missing := columns.keys() - set(data.columns):
        raise ValueError(f"Observation table is missing canonical columns: {sorted(missing)}")
    data = data.select(*columns).filter(
        pl.col("indicator_id").is_in([variable.id for variable in definition.variables])
    )
    if data.is_empty():
        raise ValueError("Declared observation variables have no rows")
    data = _observation_datetimes(data, "anchor_time")
    if data["anchor_time"].null_count():
        raise ValueError("Observation rows require non-null anchor_time")
    source = definition.source
    if source.start is not None:
        data = data.filter(
            pl.col("anchor_time") >= datetime.combine(source.start, datetime.min.time())
        )
    if source.end is not None:
        data = data.filter(
            pl.col("anchor_time") < datetime.combine(source.end, datetime.min.time())
        )
    if missing := {variable.id for variable in definition.variables} - set(data["indicator_id"]):
        raise ValueError(
            f"Declared observation variables have no rows in the selected interval: {sorted(missing)}"
        )
    data = data.with_columns(pl.col("value").cast(pl.Float64, strict=True))
    for column in ("support_start", "support_end"):
        data = _observation_datetimes(data, column)

    for variable in definition.variables:
        rows = data.filter(pl.col("indicator_id") == variable.id)
        for field, expected in (
            ("support_kind", variable.support_kind.value),
            ("summary_operator", variable.summary_operator.value),
            ("anchor_policy", variable.anchor_policy.value),
        ):
            if rows[field].null_count() or not (rows[field] == expected).all():
                raise ValueError(f"Observation {variable.id} has incompatible {field}")
        assert variable.observation_window is not None
        if any(
            window is None
            or parse_duration_to_hours(window)
            != parse_duration_to_hours(variable.observation_window)
            for window in rows["observation_window"].unique()
        ):
            raise ValueError(f"Observation {variable.id} has an incompatible observation window")
        start, end, anchor = pl.col("support_start"), pl.col("support_end"), pl.col("anchor_time")
        bound = start if variable.anchor_policy == "support_start" else end
        invalid_support = (
            (end < start)
            | (bound != anchor)
            | (pl.col("value").is_not_null() & (start.is_null() | end.is_null()))
            | (start.is_null() != end.is_null())
        )
        if variable.support_kind == "interval":
            invalid_support |= end == start
        if rows.select(invalid_support.any()).item():
            raise ValueError(f"Observation {variable.id} has invalid support boundaries or anchor")

    data = _fill_observation_nulls(data, definition)
    if not data["value"].drop_nulls().is_finite().all():
        raise ValueError("Observed values must be finite; use null for missing observations")
    for variable in definition.variables:
        values = data.filter(pl.col("indicator_id") == variable.id)["value"].drop_nulls()
        dtype = variable.measurement_dtype
        if dtype == "binary" and not values.is_in([0, 1]).all():
            raise ValueError(f"Binary observation {variable.id} requires values 0 or 1")
        if dtype == "count" and ((values < 0) | (values != values.floor())).any():
            raise ValueError(f"Count observation {variable.id} requires non-negative integers")
        if dtype in {"ordinal", "categorical"}:
            levels = variable.ordinal_levels if dtype == "ordinal" else variable.categorical_levels
            assert levels is not None
            if ((values < 0) | (values >= len(levels)) | (values != values.floor())).any():
                raise ValueError(f"Observation {variable.id} codes must index the declared levels")
    return data.sort("indicator_id", "anchor_time")


def _fill_observation_nulls(data: pl.DataFrame, definition: ObservationTableSpec) -> pl.DataFrame:
    """Apply Polars fill_null on sorted grids after variable and date selection."""
    source = definition.source
    start = (
        datetime.combine(source.start, datetime.min.time())
        if source.start is not None
        else data["anchor_time"].min()
    )
    end = (
        datetime.combine(source.end, datetime.min.time()) - timedelta(microseconds=1)
        if source.end is not None
        else data["anchor_time"].max()
    )
    span = pl.DataFrame({"anchor_time": [start, end]})
    frames = []
    for variable in definition.variables:
        rows = data.filter(pl.col("indicator_id") == variable.id)
        if variable.fill_null is None:
            frames.append(rows)
            continue
        if rows["anchor_time"].n_unique() != rows.height:
            raise ValueError("Null filling requires one row per indicator and anchor_time")
        window = variable.observation_window
        assert window is not None
        ticks = (
            support_window_tick_frame(span, window, "anchor_time")
            .rename({"__tick__": "anchor_time"})
            .filter(pl.col("anchor_time").is_between(start, end))
        )
        times = pl.concat([ticks, rows.select("anchor_time")], how="vertical_relaxed").unique()
        completed = times.join(rows, on="anchor_time", how="left").sort("anchor_time")
        at_start = variable.anchor_policy == "support_start"
        completed = completed.with_columns(
            fill_null_expression(
                pl.col("value"), variable.fill_null, limit=variable.fill_null_limit
            ),
            pl.col("indicator_id").fill_null(variable.id),
            pl.col("support_kind").fill_null(variable.support_kind.value),
            pl.col("summary_operator").fill_null(variable.summary_operator.value),
            pl.col("anchor_policy").fill_null(variable.anchor_policy.value),
            pl.col("observation_window").fill_null(window),
            pl.col("support_start").fill_null(
                pl.col("anchor_time")
                if at_start
                else pl.col("anchor_time").dt.offset_by(f"-{window}")
            ),
            pl.col("support_end").fill_null(
                pl.col("anchor_time").dt.offset_by(window) if at_start else pl.col("anchor_time")
            ),
        ).select(data.columns)
        frames.append(completed)
    return pl.concat(frames, how="vertical_relaxed")


def prepare_simulation_panel(
    report: SimulationReport,
    replicate: int,
    *,
    read_array: Callable[[str], np.ndarray],
) -> pl.DataFrame:
    """Preserve emitted values, missingness and support without exposing generating truths."""
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

    # Model days receive a fixed synthetic calendar origin. Keep missing rows so
    # the fitting grid retains times at which no indicator was observed.
    origin = datetime(1970, 1, 1, tzinfo=UTC)

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
    return (
        pl.DataFrame(rows, schema=observation_row_schema() | {"value": pl.Float64})
        .with_columns(
            pl.col("anchor_time", "support_start", "support_end")
            .str.to_datetime(format="%+")
            .dt.replace_time_zone(None)
        )
        .sort("indicator_id", "anchor_time")
    )
