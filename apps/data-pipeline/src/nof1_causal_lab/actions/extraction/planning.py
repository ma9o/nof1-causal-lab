"""extraction deterministic planning helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import polars as pl

    from nof1_causal_lab.workers.context import MeasurementContext, MeasurementIndicator
    from nof1_causal_lab.workers.schemas import WorkerOutput

logger = logging.getLogger(__name__)


def project_to_source_columns(
    df: pl.DataFrame,
    indicators: list[MeasurementIndicator],
) -> pl.DataFrame:
    """Project DataFrame to only the columns referenced by indicators."""
    source_cols: set[str] = set()
    for indicator in indicators:
        source_cols.update(indicator.get("source_columns", []))

    if not source_cols:
        return df

    missing = source_cols - set(df.columns)
    if missing:
        logger.warning(
            "extraction: source_columns not found in DataFrame, skipping them: %s",
            sorted(missing),
        )
    keep = [column for column in df.columns if column in source_cols]
    if not keep:
        return df

    dropped = len(df.columns) - len(keep)
    if dropped:
        logger.info(
            "extraction: projected %d→%d columns (dropped %d)",
            len(df.columns),
            len(keep),
            dropped,
        )
    return df.select(keep)


def prepare_semantic_chunks(
    *,
    raw_df: pl.DataFrame,
    semantic_inds: list[MeasurementIndicator],
    measurement_structure: MeasurementContext,
    model_clock: str,
    time_col: str,
    max_events_per_window: int,
    start: date | None = None,
    end: date | None = None,
) -> tuple[list[str], list[list[str]], list[MeasurementContext], WorkerOutput]:
    """Plan requests with source values; empty windows yield deterministic null rows."""
    from nof1_causal_lab.utils.causal_design import make_measurement_extraction_context
    from nof1_causal_lab.utils.data import bucket_by_clock
    from nof1_causal_lab.workers.schemas import WindowExtraction, WorkerOutput
    from nof1_causal_lab.workers.windows import format_window_chunk

    chunk_texts: list[str] = []
    chunk_window_starts: list[list[str]] = []
    chunk_contexts: list[MeasurementContext] = []
    empty_output = WorkerOutput()

    for indicator in semantic_inds:
        observation_window = indicator.get("observation_window") or model_clock
        semantic_group = [indicator]
        semantic_spec: MeasurementContext = {
            **measurement_structure,
            "model_clock": observation_window,
            "indicators": semantic_group,
        }
        extraction_ctx = make_measurement_extraction_context(semantic_spec)

        projected = project_to_source_columns(raw_df, semantic_group)
        if time_col not in projected.columns:
            projected = projected.with_columns(raw_df[time_col])

        windows = bucket_by_clock(projected, observation_window, time_col, start=start, end=end)
        logger.info(
            "extraction: bucketed %d rows into %d support windows (window=%s, indicators=%d)",
            len(projected),
            len(windows),
            observation_window,
            len(semantic_group),
        )

        if not windows:
            continue

        display_cols = [column for column in projected.columns if column != time_col]
        value_cols = indicator.get("source_columns") or display_cols
        for window in windows:
            window_start, events = window
            if not any(events[column].count() for column in value_cols if column in events.columns):
                empty_output.extractions.append(
                    WindowExtraction(
                        indicator_id=indicator["id"], window_start=window_start, value=None
                    )
                )
                continue
            chunk = [window]
            chunk_texts.append(
                format_window_chunk(chunk, time_col, display_cols, max_events_per_window)
            )
            chunk_window_starts.append([window_start for window_start, _ in chunk])
            chunk_contexts.append(extraction_ctx)

    return chunk_texts, chunk_window_starts, chunk_contexts, empty_output
