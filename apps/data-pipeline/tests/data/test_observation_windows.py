"""Preparation and simulation resolve the same calendar support, without invented counts."""

from datetime import datetime

import polars as pl
import pytest

from nof1_causal_lab.artifacts.data_preparation import FilePreparationSpec
from nof1_causal_lab.utils.aggregations import compute_indicators
from nof1_causal_lab.utils.observation_rows import (
    annotate_observation_rows,
    observation_window_boundaries,
    observation_window_bounds,
    validate_observation_rows,
)
from nof1_causal_lab.utils.observation_semantics import AnchorPolicy

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    ("window", "start", "missing", "end", "days"),
    [
        ("1mo", "2023-02-01", "2023-03-01", "2023-04-01", 28),
        ("1mo", "2024-02-01", "2024-03-01", "2024-04-01", 29),
        ("1y", "2024-01-01", "2025-01-01", "2026-01-01", 366),
        ("7d", "2024-02-29", "2024-03-07", "2024-03-14", 7),
    ],
)
def test_period_counts_keep_exact_bounds_and_missing_windows(window, start, missing, end, days):
    preparation = FilePreparationSpec.model_validate(
        {
            "source": {"files": ["counts.csv"], "start": start, "end": end},
            "definition": {
                "default_window": "1d",
                "variables": [
                    {
                        "observation": {
                            "id": "indicator:count",
                            "name": "Count",
                            "measurement_dtype": "count",
                            "aggregation": "sum",
                            "observation_window": window,
                        },
                        "extraction": {
                            "kind": "computed",
                            "how_to_measure": "Retain each reported period count.",
                            "source_columns": ["count"],
                        },
                    }
                ],
            },
        }
    )
    context = preparation.extraction_context()
    variable = context.indicators[0].observation.resolved(context.window(context.indicators[0]))
    raw = pl.DataFrame({"timestamp": [start, missing], "count": [10, None]})
    rows = validate_observation_rows(
        annotate_observation_rows(compute_indicators(raw, context, "timestamp"), (variable,)),
        (variable,),
    )
    assert rows["value"].to_list() == [10, None]
    assert (rows["support_end"][0] - rows["support_start"][0]).days == days
    assert rows["support_start"][0] == datetime.fromisoformat(start)

    # Simulation resolves these same windows from their end anchors.
    lower, upper = observation_window_bounds(
        pl.col("anchor_time"), pl.lit(window), anchor_policy=AnchorPolicy.SUPPORT_END
    )
    simulated = rows.select(lower.alias("support_start"), upper.alias("support_end"))
    assert simulated.equals(rows.select("support_start", "support_end"))
    if window in {"1mo", "1y"}:
        boundaries = observation_window_boundaries(
            variable.observation_window, datetime.fromisoformat(start), datetime.fromisoformat(end)
        )
        assert boundaries == tuple(datetime.fromisoformat(value) for value in (start, missing, end))
        # A calendar count cannot be attached to a partial reporting period.
        unaligned = pl.DataFrame({"anchor_time": [datetime.fromisoformat(start).replace(day=2)]})
        assert unaligned.select(lower, upper.alias("end")).row(0) == (None, None)
