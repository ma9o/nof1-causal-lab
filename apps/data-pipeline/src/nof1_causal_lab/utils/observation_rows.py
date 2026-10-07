"""Shared structural validation of prepared and compared observation histories."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

import polars as pl

from nof1_causal_lab.artifacts.duration import Duration
from nof1_causal_lab.utils.observation_semantics import AnchorPolicy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nof1_causal_lab.artifacts.observations import ResolvedObservationSpec


def validate_observation_rows(
    data: pl.DataFrame, variables: Sequence[ResolvedObservationSpec]
) -> pl.DataFrame:
    """Normalize UTC timestamps and validate values/support without filling or encoding."""
    columns = observation_row_schema()
    if missing := columns.keys() - set(data.columns):
        raise ValueError(f"Observation table is missing canonical columns: {sorted(missing)}")
    identities = {variable.id for variable in variables}
    if len(identities) != len(variables):
        raise ValueError("Observation variables must have unique identities")
    if unknown := set(data["indicator_id"]) - identities:
        raise ValueError(f"Observations have no variable definition: {sorted(unknown)}")
    data = data.select(*columns).with_columns(pl.col("value").cast(pl.Float64, strict=True))
    for column in ("anchor_time", "support_start", "support_end"):
        dtype = data.schema[column]
        value = pl.col(column)
        if dtype == pl.String:
            value = value.str.to_datetime(time_zone="UTC")
        elif dtype == pl.Date or dtype == pl.Null:
            value = value.cast(pl.Datetime).dt.replace_time_zone("UTC")
        elif isinstance(dtype, pl.Datetime):
            value = (
                value.dt.convert_time_zone("UTC")
                if dtype.time_zone is not None
                else value.dt.replace_time_zone("UTC")
            )
        else:
            raise ValueError(f"Observation {column} must contain timestamps")
        data = data.with_columns(value.dt.replace_time_zone(None).alias(column))
    if data["anchor_time"].null_count():
        raise ValueError("Observations require non-null anchor_time")
    if not data["value"].drop_nulls().is_finite().all():
        raise ValueError("Observed values must be finite; use null for missing observations")
    for variable in variables:
        rows = data.filter(pl.col("indicator_id") == variable.id)
        if rows["anchor_time"].n_unique() != rows.height:
            raise ValueError(f"Observation {variable.id} has duplicate anchors within a history")
        for field, expected in (
            ("support_kind", variable.support_kind.value),
            ("summary_operator", variable.summary_operator.value),
            ("anchor_policy", variable.anchor_policy.value),
        ):
            if rows[field].null_count() or not (rows[field] == expected).all():
                raise ValueError(f"Observation {variable.id} has inconsistent {field}")
        if any(
            window is None or Duration(window).seconds != variable.observation_window.seconds
            for window in rows["observation_window"].unique()
        ):
            raise ValueError(f"Observation {variable.id} has an incompatible observation window")
        start, end, anchor = pl.col("support_start"), pl.col("support_end"), pl.col("anchor_time")
        bound = start if variable.anchor_policy == "support_start" else end
        invalid = (
            (end < start)
            | (bound != anchor)
            | (start.is_null() != end.is_null())
            | (pl.col("value").is_not_null() & (start.is_null() | end.is_null()))
        )
        if variable.support_kind == "interval":
            invalid |= end == start
        if rows.select(invalid.any()).item():
            raise ValueError(f"Observation {variable.id} has invalid measurement support")
        values = rows["value"].drop_nulls()
        dtype = variable.measurement_dtype
        if dtype == "binary" and not values.is_in([0, 1]).all():
            raise ValueError(f"Binary observation {variable.id} requires values 0 or 1")
        if dtype == "count" and ((values < 0) | (values != values.floor())).any():
            raise ValueError(f"Count observation {variable.id} requires non-negative integers")
        levels = variable.ordinal_levels or variable.categorical_levels
        if (
            levels is not None
            and ((values < 0) | (values >= len(levels)) | (values != values.floor())).any()
        ):
            raise ValueError(f"Observation {variable.id} has values outside its codebook")
    return data.sort("indicator_id", "anchor_time")


def prepared_time_origin(data: pl.DataFrame, start: date | None) -> datetime:
    """Resolve a files panel's origin once, before any model selects its indicators."""
    if start is not None:
        return datetime.combine(start, datetime.min.time(), tzinfo=UTC)
    origin = (
        data.select(pl.min_horizontal("anchor_time", "support_start", "support_end"))
        .to_series()
        .min()
    )
    if not isinstance(origin, datetime):
        raise ValueError("Prepared observations must have a temporal origin")
    return origin.replace(tzinfo=UTC)


logger = logging.getLogger(__name__)


OBSERVATION_ROW_SCHEMA = {
    "indicator_id": pl.Utf8,
    "value": pl.Utf8,
    "anchor_time": pl.Utf8,
    "support_kind": pl.Utf8,
    "summary_operator": pl.Utf8,
    "anchor_policy": pl.Utf8,
    "observation_window": pl.Utf8,
    "support_start": pl.Utf8,
    "support_end": pl.Utf8,
}


def ensure_datetime_column(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    """Normalize the raw temporal axis to UTC-naive datetimes."""
    dtype = df.schema[time_col]
    value = pl.col(time_col)
    if dtype == pl.String:
        value = value.str.to_datetime(time_zone="UTC").dt.replace_time_zone(None)
    elif dtype == pl.Date:
        value = value.cast(pl.Datetime)
    elif isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        value = value.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return df.with_columns(value.alias(time_col))


def support_window_tick_frame(
    df: pl.DataFrame,
    model_clock: str,
    time_col: str,
    *,
    start: date | None = None,
    end: date | None = None,
) -> pl.DataFrame:
    """Select whole calendar support windows, including empty windows inside the span."""
    if df.is_empty():
        return pl.DataFrame(schema={"__tick__": pl.Datetime})

    df = ensure_datetime_column(df, time_col)
    observed_ticks = df.select(pl.col(time_col).dt.truncate(model_clock).alias("__tick__"))
    first, last = observed_ticks.select(
        pl.col("__tick__").min().alias("start"),
        pl.col("__tick__").max().alias("end"),
    ).row(0)
    if first is None or last is None:
        return pl.DataFrame(schema={"__tick__": observed_ticks.schema["__tick__"]})
    lower = datetime.combine(start, datetime.min.time()) if start is not None else first
    upper = datetime.combine(end, datetime.min.time()) if end is not None else last
    bounds = pl.Series([lower, upper]).dt.truncate(model_clock)
    if bounds[0] > bounds[1]:
        return pl.DataFrame(schema={"__tick__": observed_ticks.schema["__tick__"]})
    ticks = pl.DataFrame(
        {
            "__tick__": pl.datetime_range(
                bounds[0], bounds[1], interval=model_clock, eager=True
            ).cast(observed_ticks.schema["__tick__"])
        }
    )
    if start is not None:
        ticks = ticks.filter(pl.col("__tick__") >= lower)
    if end is not None:
        ticks = ticks.filter(pl.col("__tick__").dt.offset_by(model_clock) <= upper)
    return ticks


def bucket_by_clock(
    df: pl.DataFrame,
    model_clock: str,
    time_col: str,
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[tuple[str, pl.DataFrame]]:
    """Group rows into complete model-clock support windows.

    Args:
        df: Source rows containing a date or datetime column.
        model_clock: Polars duration spelling such as ``1d`` or ``4h``.
        time_col: Timestamp column used to assign rows to clock ticks.
        start: Optional lower calendar bound; only windows starting at or after
            this bound are retained.
        end: Optional upper calendar bound; only windows ending at or before
            this bound are retained.

    Returns:
        Chronologically ordered pairs of ISO window-start strings and row frames.
        Empty windows inside the selected span are retained. Without explicit
        bounds, the span runs from the first observed tick through the last.
    """
    df = ensure_datetime_column(df, time_col)

    # Truncate to tick boundaries
    bucketed = df.with_columns(pl.col(time_col).dt.truncate(model_clock).alias("__tick__")).sort(
        time_col
    )
    tick_frame = support_window_tick_frame(df, model_clock, time_col, start=start, end=end)

    groups = {
        tick_val[0]: group_df.drop("__tick__")
        for tick_val, group_df in bucketed.group_by("__tick__", maintain_order=True)
    }
    empty_events = df.head(0)
    result = []
    tick_datetimes: list[datetime] = tick_frame["__tick__"].to_list()
    for tick_dt in tick_datetimes:
        events = groups.get(tick_dt, empty_events)
        tick_id = tick_dt.isoformat()
        result.append((tick_id, events))

    return result


def observation_row_schema() -> dict[str, pl.DataType | type[pl.DataType]]:
    """Schema for canonical long-format observation rows."""
    return dict(OBSERVATION_ROW_SCHEMA)


def annotate_observation_rows(
    df: pl.DataFrame,
    variables: Sequence[ResolvedObservationSpec],
    *,
    time_col: str = "timestamp",
) -> pl.DataFrame:
    """Attach observation metadata to long-format extraction rows.

    The input ``time_col`` is the support-window start emitted by the computed
    and semantic extraction paths. The canonical observation-row contract keeps:
    - ``anchor_time``: latent-grid attachment time for the observation
    - ``support_start`` / ``support_end``: realized support bounds

    Canonical support semantics are always derived from the measurement structure,
    not preserved from any caller-supplied row metadata.
    """
    if df.is_empty():
        return pl.DataFrame(schema=observation_row_schema())
    if "indicator_id" not in df.columns:
        raise ValueError("Extraction rows must carry an indicator_id")
    indicator_ids = {variable.id for variable in variables}
    unknown = set(df["indicator_id"].unique()) - indicator_ids
    if unknown:
        raise ValueError(f"Extraction rows reference unknown indicators: {unknown}")
    for col_name, dtype in OBSERVATION_ROW_SCHEMA.items():
        if col_name not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col_name))

    indicator_rows = [
        {
            "indicator_id": variable.id,
            "support_kind_meta": variable.support_kind.value,
            "summary_operator_meta": variable.summary_operator.value,
            "anchor_policy_meta": variable.anchor_policy.value,
            "observation_window_meta": variable.observation_window.source,
        }
        for variable in variables
    ]
    kind_df = (
        pl.DataFrame(
            indicator_rows,
            schema={
                "indicator_id": pl.Utf8,
                "support_kind_meta": pl.Utf8,
                "summary_operator_meta": pl.Utf8,
                "anchor_policy_meta": pl.Utf8,
                "observation_window_meta": pl.Utf8,
            },
        )
        if indicator_rows
        else pl.DataFrame(
            schema={
                "indicator_id": pl.Utf8,
                "support_kind_meta": pl.Utf8,
                "summary_operator_meta": pl.Utf8,
                "anchor_policy_meta": pl.Utf8,
                "observation_window_meta": pl.Utf8,
            }
        )
    )

    if kind_df.height > 0:
        df = df.join(kind_df, on="indicator_id", how="left")
    else:
        df = df.with_columns(
            pl.lit(None, dtype=pl.Utf8).alias("support_kind_meta"),
            pl.lit(None, dtype=pl.Utf8).alias("summary_operator_meta"),
            pl.lit(None, dtype=pl.Utf8).alias("anchor_policy_meta"),
            pl.lit(None, dtype=pl.Utf8).alias("observation_window_meta"),
        )

    ts_expr = (
        # extraction merges support-window starts coming from both paths:
        # computed rows use naive bucket strings while semantic rows can carry
        # the same UTC boundary with an explicit `+00:00` suffix from the
        # worker header. Normalize the redundant UTC suffix so mixed batches
        # parse consistently into the same support bounds.
        pl.col(time_col)
        .str.replace(r"[Zz]$", "")
        .str.replace(r"[+-]\d{2}:\d{2}$", "")
        .str.to_datetime(strict=False)
        if df.schema.get(time_col) == pl.Utf8
        else pl.col(time_col)
    )
    support_start_expr = ts_expr.dt.to_string("%Y-%m-%dT%H:%M:%S")
    observation_window_expr = pl.col("observation_window_meta")
    support_kind_expr = pl.col("support_kind_meta")
    summary_operator_expr = pl.col("summary_operator_meta")
    anchor_policy_expr = pl.col("anchor_policy_meta")
    support_end_expr = (
        pl.when(observation_window_expr.is_not_null())
        .then(ts_expr.dt.offset_by(observation_window_expr).dt.to_string("%Y-%m-%dT%H:%M:%S"))
        .otherwise(support_start_expr)
    )
    anchor_time_expr = (
        pl.when(anchor_policy_expr == AnchorPolicy.SUPPORT_START.value)
        .then(support_start_expr)
        .otherwise(support_end_expr)
    )

    df = df.with_columns(
        anchor_time_expr.alias("anchor_time"),
        support_kind_expr.alias("support_kind"),
        summary_operator_expr.alias("summary_operator"),
        anchor_policy_expr.alias("anchor_policy"),
        observation_window_expr.alias("observation_window"),
        support_start_expr.alias("support_start"),
        support_end_expr.alias("support_end"),
    ).drop(
        "support_kind_meta",
        "summary_operator_meta",
        "anchor_policy_meta",
        "observation_window_meta",
    )

    if time_col != "anchor_time" and time_col in df.columns:
        df = df.drop(time_col)

    return df


def pivot_to_wide(df: pl.DataFrame, *, time_origin: datetime) -> pl.DataFrame:
    """Pivot recorded observations onto a shared model-day axis.

    Args:
        df: Long-format rows with ``indicator_id``, ``value``, and ``anchor_time``.
        time_origin: Calendar instant of model day zero, or ``None`` to use the
            calendar-free origin owned by ``ObservationInstant``.

    Returns:
        A frame with a fractional-day ``time`` column and one numeric column per
        indicator. Repeated indicator/anchor pairs are averaged; missing cells
        remain null. Empty input returns an empty frame.

    Raises:
        ValueError: Nonempty input has no ``anchor_time`` column.
    """
    if df.is_empty():
        return pl.DataFrame()

    time_col = "anchor_time"
    if time_col not in df.columns:
        raise ValueError("Observation data must include an 'anchor_time' column.")

    # Parse string timestamps to datetime before pivoting so the
    # datetime→fractional-days conversion below always triggers.
    df = ensure_datetime_column(df, time_col)

    wide_data = (
        df.with_columns(pl.col("value").cast(pl.Float64, strict=False))
        .pivot(on="indicator_id", index=time_col, values="value", aggregate_function="mean")
        .sort(time_col)
    )

    if wide_data.schema[time_col].base_type() in (pl.Datetime, pl.Date):
        from nof1_causal_lab.utils.time_coordinates import ModelTime, ObservationInstant

        origin = ObservationInstant(time_origin)
        wide_data = wide_data.with_columns(
            ModelTime.bind_column(pl.col(time_col), origin).alias(time_col)
        )

    if time_col in wide_data.columns:
        wide_data = wide_data.rename({time_col: "time"})

    # --- Sparsity validation ---
    indicator_cols = [c for c in wide_data.columns if c != "time"]
    if indicator_cols:
        n_rows = wide_data.height
        per_indicator: list[str] = []
        total_null = 0
        total_cells = 0
        for col in indicator_cols:
            n_null = wide_data[col].null_count()
            n_obs = n_rows - n_null
            total_null += n_null
            total_cells += n_rows
            if n_null > 0:
                pct = n_null / n_rows * 100
                per_indicator.append(f"{col}: {n_obs}/{n_rows} observed ({pct:.0f}% missing)")

        if total_cells > 0:
            overall_pct = total_null / total_cells * 100
            if overall_pct > 50:
                logger.warning(
                    "Sparse observation matrix: %.0f%% missing (%d/%d cells). "
                    "Multi-granularity indicators may cause excessive sparsity. "
                    "Per-indicator: %s",
                    overall_pct,
                    total_null,
                    total_cells,
                    "; ".join(per_indicator) if per_indicator else "all complete",
                )
            elif per_indicator:
                logger.info(
                    "Observation matrix sparsity: %.0f%% missing. %s",
                    overall_pct,
                    "; ".join(per_indicator),
                )

    return wide_data
