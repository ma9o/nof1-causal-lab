"""extraction materialization helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.measurements import ObservationRecord


def materialize_panel(
    observation_rows: list[ObservationRecord],
    measurement_structure: UncheckedJsonObject,
) -> pl.DataFrame:
    """Encode observation rows into the canonical panel, including a typed empty table."""
    from nof1_causal_lab.utils.aggregations import _encode_non_continuous
    from nof1_causal_lab.utils.data import observation_row_schema

    if observation_rows:
        data_for_model = pl.DataFrame(observation_rows)
    else:
        data_for_model = pl.DataFrame(schema=observation_row_schema())

    if len(data_for_model) > 0:
        dtype_lookup = {
            indicator["id"]: indicator.get("measurement_dtype", "continuous")
            for indicator in measurement_structure.get("indicators", [])
            if indicator.get("name")
        }
        ordinal_levels_lookup: dict[str, list[str]] = {
            ind["id"]: ind["ordinal_levels"]
            for ind in measurement_structure.get("indicators", [])
            if ind.get("ordinal_levels")
        }
        data_for_model = _encode_non_continuous(data_for_model, dtype_lookup, ordinal_levels_lookup)
        data_for_model = data_for_model.with_columns(
            pl.col("value").cast(pl.Float64, strict=False).alias("value"),
            pl.col("anchor_time")
            .str.replace(r"[Zz]$", "")
            .str.replace(r"[+-]\d{2}:\d{2}$", "")
            .str.to_datetime(strict=False)
            .alias("anchor_time"),
            pl.col("support_start")
            .str.replace(r"[Zz]$", "")
            .str.replace(r"[+-]\d{2}:\d{2}$", "")
            .str.to_datetime(strict=False)
            .alias("support_start"),
            pl.col("support_end")
            .str.replace(r"[Zz]$", "")
            .str.replace(r"[+-]\d{2}:\d{2}$", "")
            .str.to_datetime(strict=False)
            .alias("support_end"),
        ).drop_nulls(subset=["anchor_time"])
        data_for_model = data_for_model.sort("indicator_id", "anchor_time")

    return data_for_model
