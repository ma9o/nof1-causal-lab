"""Tests for aggregation utility functions.

Covers: _build_agg_expr, _build_map_groups_fn, _encode_non_continuous, compute_indicators.
"""

from datetime import datetime

import polars as pl
import pytest

from nof1_causal_lab.utils.aggregations import (
    _build_agg_expr,
    _build_map_groups_fn,
    _encode_non_continuous,
    compute_indicators,
)


def _make_df(values: list[float]) -> pl.DataFrame:
    """Create a simple DataFrame with a 'value' column."""
    return pl.DataFrame({"value": values})


# =============================================================================
# _build_agg_expr
# =============================================================================


class TestBuildAggExpr:
    @pytest.mark.parametrize(
        ("agg_name", "values", "expected"),
        [
            ("mean", [1.0, 2.0, 3.0], 2.0),
            ("sum", [1.0, 2.0, 3.0], 6.0),
            ("min", [3.0, 1.0, 2.0], 1.0),
            ("max", [3.0, 1.0, 2.0], 3.0),
            ("count", [1.0, 2.0, 3.0], 3.0),
            ("median", [1.0, 5.0, 3.0], 3.0),
            ("first", [7.0, 2.0, 3.0], 7.0),
            ("last", [7.0, 2.0, 9.0], 9.0),
            ("range", [1.0, 5.0, 3.0], 4.0),
            ("p25", [1.0, 2.0, 3.0, 4.0], 2.0),
            ("p75", [1.0, 2.0, 3.0, 4.0], 3.0),
            ("iqr", [1.0, 2.0, 3.0, 4.0], 1.0),
            ("cv", [10.0, 12.0, 8.0], 0.2),
            ("instability", [1.0, 3.0, 2.0, 4.0], 3.0),
        ],
    )
    def test_supported_aggregations(self, agg_name, values, expected):
        df = _make_df(values)
        result = df.select(_build_agg_expr(agg_name))
        assert result["value"][0] == pytest.approx(expected)

    def test_std(self):
        df = _make_df([1.0, 2.0, 3.0])
        result = df.select(_build_agg_expr("std"))
        assert result["value"][0] == pytest.approx(1.0)  # sample std (ddof=1)

    @pytest.mark.parametrize(
        ("agg_name", "expected"),
        [
            ("mean", 42.0),
            ("sum", 42.0),
            ("min", 42.0),
            ("max", 42.0),
            ("count", 1.0),
            ("median", 42.0),
            ("first", 42.0),
            ("last", 42.0),
        ],
    )
    def test_single_value(self, agg_name, expected):
        """Aggregating a single value works for the basic scalar reducers."""
        result = _make_df([42.0]).select(_build_agg_expr(agg_name))
        assert result["value"][0] == pytest.approx(expected), f"{agg_name} failed on single value"

    def test_cv_zero_mean(self):
        """CV with zero mean returns null (guarded by abs(mean) > 1e-15)."""
        df = _make_df([-1.0, 1.0])  # mean = 0
        result = df.select(_build_agg_expr("cv"))
        assert result["value"][0] is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            _build_agg_expr("nonexistent_agg")


# =============================================================================
# _build_map_groups_fn
# =============================================================================


class TestBuildMapGroupsFn:
    def test_trend_positive_slope(self):
        """Increasing values should give positive slope."""
        fn = _build_map_groups_fn("trend")
        df = pl.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "group": ["a"] * 4})
        result = fn(df)
        assert result["value"][0] > 0

    def test_trend_zero_slope(self):
        """Constant values should give zero slope."""
        fn = _build_map_groups_fn("trend")
        df = pl.DataFrame({"value": [5.0, 5.0, 5.0], "group": ["a"] * 3})
        result = fn(df)
        assert abs(result["value"][0]) < 1e-10

    def test_trend_single_point(self):
        """Single data point should give zero slope."""
        fn = _build_map_groups_fn("trend")
        df = pl.DataFrame({"value": [5.0], "group": ["a"]})
        result = fn(df)
        assert abs(result["value"][0]) < 1e-10

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown map_groups"):
            _build_map_groups_fn("nonexistent")


# =============================================================================
# _encode_non_continuous
# =============================================================================


class TestEncodeNonContinuous:
    def test_binary_true_false(self):
        df = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:869e1d0209fb25a7fc06",
                    "indicator:869e1d0209fb25a7fc06",
                    "indicator:869e1d0209fb25a7fc06",
                ],
                "value": ["true", "false", "yes"],
            }
        )
        result = _encode_non_continuous(df, {"indicator:869e1d0209fb25a7fc06": "binary"})
        values = result.sort("value")["value"].to_list()
        # "false" -> "0.0", "true" -> "1.0", "yes" -> "1.0"
        assert "0.0" in values
        assert "1.0" in values

    def test_ordinal_numeric_codes_passthrough(self):
        df = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:036fd134b9ad32d7a2ca",
                    "indicator:036fd134b9ad32d7a2ca",
                    "indicator:036fd134b9ad32d7a2ca",
                ],
                "value": ["0", "1", "2"],
            }
        )
        result = _encode_non_continuous(
            df,
            {"indicator:036fd134b9ad32d7a2ca": "ordinal"},
            ordinal_levels_lookup={"indicator:036fd134b9ad32d7a2ca": ["low", "medium", "high"]},
        )
        vals = sorted(float(v) for v in result["value"].to_list())
        assert vals == [0.0, 1.0, 2.0]

    def test_ordinal_out_of_range_code_becomes_null(self):
        df = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:036fd134b9ad32d7a2ca",
                    "indicator:036fd134b9ad32d7a2ca",
                ],
                "value": ["2", "3"],
            }
        )
        result = _encode_non_continuous(
            df,
            {"indicator:036fd134b9ad32d7a2ca": "ordinal"},
            ordinal_levels_lookup={"indicator:036fd134b9ad32d7a2ca": ["low", "medium", "high"]},
        ).sort("value", nulls_last=True)
        assert result["value"].to_list() == ["2.0", None]

    def test_continuous_passthrough(self):
        df = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:c5b118ae552981435d7b",
                    "indicator:c5b118ae552981435d7b",
                ],
                "value": [70.5, 80.2],
            }
        )
        result = _encode_non_continuous(df, {"indicator:c5b118ae552981435d7b": "continuous"})
        assert result["value"].to_list() == [70.5, 80.2]

    def test_empty_dtype_lookup(self):
        df = pl.DataFrame({"indicator_id": ["indicator:1f4c67cecb9238ee1a80"], "value": [1.0]})
        result = _encode_non_continuous(df, {})
        assert result["value"].to_list() == [1.0]

    def test_mixed_indicators(self):
        """Only non-continuous indicators should be encoded; continuous left unchanged."""
        df = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:869e1d0209fb25a7fc06",
                    "indicator:c5b118ae552981435d7b",
                ],
                "value": ["true", "70.5"],
            }
        )
        result = _encode_non_continuous(
            df,
            {
                "indicator:869e1d0209fb25a7fc06": "binary",
                "indicator:c5b118ae552981435d7b": "continuous",
            },
        )
        mood_row = result.filter(pl.col("indicator_id") == "indicator:869e1d0209fb25a7fc06")
        weight_row = result.filter(pl.col("indicator_id") == "indicator:c5b118ae552981435d7b")
        assert float(mood_row["value"][0]) == 1.0
        assert weight_row["value"][0] == "70.5"


# =============================================================================
# compute_indicators
# =============================================================================


def _make_raw_df() -> pl.DataFrame:
    """Create a raw DataFrame spanning 3 days with heart_rate and steps."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 1, 8, 0),
                datetime(2024, 1, 1, 12, 0),
                datetime(2024, 1, 1, 18, 0),
                datetime(2024, 1, 2, 9, 0),
                datetime(2024, 1, 2, 15, 0),
                datetime(2024, 1, 3, 10, 0),
            ],
            "heart_rate": [72.0, 85.0, 68.0, 74.0, 90.0, 70.0],
            "steps": [1000, 3000, 500, 2000, 4000, 1500],
        }
    )


class TestComputeIndicators:
    def test_single_mean(self):
        """Mean of heart_rate across 3 daily ticks."""
        df = _make_raw_df()
        indicators = [
            {
                "id": "indicator:c21b43949b3712e734c8",
                "name": "avg_hr",
                "source_columns": ["heart_rate"],
                "aggregation": "mean",
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert result.columns == ["indicator_id", "value", "timestamp"]
        assert result["indicator_id"].to_list() == ["indicator:c21b43949b3712e734c8"] * 3
        # Day 1: mean(72, 85, 68) = 75.0
        values = [float(v) for v in result["value"].to_list()]
        assert abs(values[0] - 75.0) < 0.01

    def test_sum_aggregation(self):
        """Sum of steps across daily ticks."""
        df = _make_raw_df()
        indicators = [
            {
                "id": "indicator:a0ce08437c19d06aafd1",
                "name": "total_steps",
                "source_columns": ["steps"],
                "aggregation": "sum",
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        values = [float(v) for v in result["value"].to_list()]
        # Day 1: 1000+3000+500=4500, Day 2: 2000+4000=6000, Day 3: 1500
        assert values == [4500.0, 6000.0, 1500.0]

    def test_multiple_indicators(self):
        """Two computed indicators in one call."""
        df = _make_raw_df()
        indicators = [
            {
                "id": "indicator:c21b43949b3712e734c8",
                "name": "avg_hr",
                "source_columns": ["heart_rate"],
                "aggregation": "mean",
            },
            {
                "id": "indicator:a0ce08437c19d06aafd1",
                "name": "total_steps",
                "source_columns": ["steps"],
                "aggregation": "sum",
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        # 3 ticks * 2 indicators = 6 rows
        assert len(result) == 6
        assert set(result["indicator_id"].to_list()) == {
            "indicator:c21b43949b3712e734c8",
            "indicator:a0ce08437c19d06aafd1",
        }

    def test_output_schema(self):
        """Output columns are exactly {indicator, value, timestamp} as Utf8."""
        df = _make_raw_df()
        indicators = [
            {
                "id": "indicator:c21b43949b3712e734c8",
                "name": "avg_hr",
                "source_columns": ["heart_rate"],
                "aggregation": "mean",
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert result.columns == ["indicator_id", "value", "timestamp"]
        assert result.schema["indicator_id"] == pl.Utf8
        assert result.schema["value"] == pl.Utf8
        assert result.schema["timestamp"] == pl.Utf8

    def test_empty_indicators(self):
        """Empty indicator list returns empty DataFrame with correct schema."""
        df = _make_raw_df()
        result = compute_indicators(df, [], "1d", "timestamp")
        assert result.columns == ["indicator_id", "value", "timestamp"]
        assert len(result) == 0

    def test_trend_aggregation(self):
        """Trend (map_groups path) computes OLS slope."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 18, 0),
                ],
                "hr": [70.0, 75.0, 80.0],  # increasing → positive slope
            }
        )
        indicators = [
            {
                "id": "indicator:cacaf060b8afc7a952d9",
                "name": "hr_trend",
                "source_columns": ["hr"],
                "aggregation": "trend",
            }
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert len(result) == 1
        assert float(result["value"][0]) > 0  # positive slope

    def test_missing_source_column(self):
        """Missing source column is skipped with warning, not crash."""
        df = _make_raw_df()
        indicators = [
            {
                "id": "indicator:938f71ec0999bbbe342c",
                "name": "missing",
                "source_columns": ["nonexistent_col"],
                "aggregation": "mean",
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert len(result) == 0

    def test_null_source_values(self):
        """Source column with nulls aggregates correctly."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 18, 0),
                ],
                "hr": [72.0, None, 68.0],
            }
        )
        indicators = [
            {
                "id": "indicator:c21b43949b3712e734c8",
                "name": "avg_hr",
                "source_columns": ["hr"],
                "aggregation": "mean",
            }
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        assert len(result) == 1
        # mean(72, 68) = 70.0 (null ignored)
        assert abs(float(result["value"][0]) - 70.0) < 0.01

    def test_first_ignores_leading_nulls(self):
        """Point aggregations should use the first observed value, not the first row."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 9, 0),
                    datetime(2024, 1, 1, 10, 0),
                ],
                "care_setting": [None, "home", "clinic"],
            }
        )
        indicators = [
            {
                "id": "indicator:c0486b3cc6b5559e95d0",
                "name": "first_setting",
                "source_columns": ["care_setting"],
                "measurement_dtype": "categorical",
                "aggregation": "first",
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["home"]

    def test_categorical_last_preserves_string_value(self):
        """Direct categorical computed indicators should preserve raw labels."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                ],
                "care_setting": ["home", "clinic"],
            }
        )
        indicators = [
            {
                "id": "indicator:c664114dcb2ded460d6e",
                "name": "last_setting",
                "source_columns": ["care_setting"],
                "measurement_dtype": "categorical",
                "aggregation": "last",
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["clinic"]

    def test_ordinal_last_encodes_label_to_numeric_code(self):
        """Direct ordinal computed indicators should emit canonical integer codes."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                ],
                "mood_label": ["bad", "good"],
            }
        )
        indicators = [
            {
                "id": "indicator:5dc4b94693df7e0aef53",
                "name": "closing_mood",
                "source_columns": ["mood_label"],
                "measurement_dtype": "ordinal",
                "aggregation": "last",
                "ordinal_levels": ["bad", "ok", "good"],
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["2"]

    def test_count_aggregation_counts_non_null_string_values(self):
        """Count aggregations should not null out string source columns before counting."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 9, 0),
                    datetime(2024, 1, 1, 10, 0),
                ],
                "message_text": ["alpha", None, "beta"],
            }
        )
        indicators = [
            {
                "id": "indicator:6a123c22171bbe49ee54",
                "name": "text_events",
                "source_columns": ["message_text"],
                "measurement_dtype": "count",
                "aggregation": "count",
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["2"]

    def test_computed_rule_multi_column_formula(self):
        """Computed rules can deterministically derive window values from multiple columns."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 9, 0),
                ],
                "systolic_bp": [120.0, 150.0, 110.0],
                "diastolic_bp": [80.0, 90.0, 70.0],
            }
        )
        indicators = [
            {
                "id": "indicator:46df40a36557ed795b7b",
                "name": "map",
                "source_columns": ["systolic_bp", "diastolic_bp"],
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "computed_rule": "mean(diastolic_bp + (systolic_bp - diastolic_bp) / 3)",
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        values = [float(value) for value in result["value"].to_list()]
        assert values[0] == pytest.approx((80 + (120 - 80) / 3 + 90 + (150 - 90) / 3) / 2)
        assert values[1] == pytest.approx(70 + (110 - 70) / 3)

    def test_computed_rule_filtered_count_preserves_zero_vs_null(self):
        """Filtered deterministic counts should distinguish observed zero from no observation."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 1, 14, 0),
                    datetime(2024, 1, 2, 9, 0),
                    datetime(2024, 1, 3, 10, 0),
                ],
                "event_type": ["med_admin", "med_admin", "note", "med_admin", "note"],
                "admin_status": ["missed", "taken", None, "taken", None],
            }
        )
        indicators = [
            {
                "id": "indicator:097d80d6767b143a71b4",
                "name": "missed_doses",
                "source_columns": ["event_type", "admin_status"],
                "measurement_dtype": "count",
                "aggregation": "sum",
                "computed_rule": 'None if count_true(event_type == "med_admin") == 0 else sum(1 if (event_type == "med_admin" and admin_status == "missed") else 0)',
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
        ]
        assert result["value"].to_list() == ["1", "0", None]

    def test_direct_aggregations_materialize_empty_support_windows(self):
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 2, 10, 0),
                    datetime(2024, 1, 4, 10, 0),
                ],
                "event_id": ["a", None, "b"],
                "score": [5.0, None, 7.0],
            }
        )
        indicators = [
            {
                "id": "indicator:e8bcfd4304d709a5cce3",
                "name": "event_count",
                "source_columns": ["event_id"],
                "measurement_dtype": "count",
                "aggregation": "count",
            },
            {
                "id": "indicator:422841bed8b563beeec0",
                "name": "score_mean",
                "source_columns": ["score"],
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        counts = result.filter(pl.col("indicator_id") == "indicator:e8bcfd4304d709a5cce3")
        means = result.filter(pl.col("indicator_id") == "indicator:422841bed8b563beeec0")
        assert counts["timestamp"].to_list() == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
            "2024-01-04T00:00:00",
        ]
        assert counts["value"].to_list() == ["1", "0", None, "1"]
        assert means["value"].to_list() == ["5.0", None, None, "7.0"]

    def test_computed_rule_count_materializes_empty_support_windows(self):
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 2, 10, 0),
                    datetime(2024, 1, 4, 10, 0),
                ],
                "title": ["stress spike", "ordinary update", "another ordinary update"],
            }
        )
        indicators = [
            {
                "id": "indicator:42b9030df461ee342451",
                "name": "stress_mentions",
                "source_columns": ["title"],
                "measurement_dtype": "count",
                "aggregation": "count",
                "computed_rule": 'count_true(contains(lower(coalesce(title, "")), "stress"))',
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
            "2024-01-04T00:00:00",
        ]
        assert result["value"].to_list() == ["1", "0", None, "0"]

    def test_computed_rule_binary_flag_preserves_zero_vs_null(self):
        """Binary deterministic window flags should keep observed negative distinct from missing."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 8, 0),
                    datetime(2024, 1, 2, 12, 0),
                    datetime(2024, 1, 3, 8, 0),
                    datetime(2024, 1, 3, 12, 0),
                ],
                "spo2_pct": [95.0, 94.0, 91.0, 95.0, None, None],
            }
        )
        indicators = [
            {
                "id": "indicator:86e4453f8f098e1007ef",
                "name": "low_spo2",
                "source_columns": ["spo2_pct"],
                "measurement_dtype": "binary",
                "aggregation": "last",
                "computed_rule": "1 if any(spo2_pct < 92) else (0 if count_non_null(spo2_pct) > 0 else None)",
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["value"].to_list() == ["0", "1", None]

    def test_computed_rule_contains_any_literal_list(self):
        """contains_any() should accept literal string lists in computed rules."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 9, 0),
                ],
                "title_url": [
                    "https://facebook.com/some-post",
                    "https://example.com",
                    "https://reddit.com/r/polars",
                ],
            }
        )
        indicators = [
            {
                "id": "indicator:cce51902cd8da81b8457",
                "name": "social_media_hits",
                "source_columns": ["title_url"],
                "measurement_dtype": "count",
                "aggregation": "count",
                "computed_rule": 'count_true(contains_any(title_url, ["facebook.com", "reddit.com"]))',
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == ["2024-01-01T00:00:00", "2024-01-02T00:00:00"]
        assert result["value"].to_list() == ["1", "1"]

    def test_computed_rule_nested_contains_any_with_if_else(self):
        """Nested computed rules should handle contains_any() list literals inside bool expressions."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 8, 0),
                    datetime(2024, 1, 1, 12, 0),
                    datetime(2024, 1, 2, 9, 0),
                    datetime(2024, 1, 2, 14, 0),
                    datetime(2024, 1, 3, 10, 0),
                ],
                "title": [
                    "Stress management",
                    None,
                    "ordinary browsing",
                    None,
                    "nothing relevant",
                ],
                "title_url": [
                    None,
                    "https://example.com",
                    "https://burnout.example/article",
                    "https://example.com/other",
                    None,
                ],
            }
        )
        indicators = [
            {
                "id": "indicator:1d23877d34bd268f1d4d",
                "name": "stress_content_count",
                "source_columns": ["timestamp", "title", "title_url"],
                "measurement_dtype": "count",
                "aggregation": "count",
                "computed_rule": 'None if count_non_null(timestamp) == 0 else count_true(contains_any(lower(coalesce(title, "")), ["stress", "burnout"]) or contains_any(lower(coalesce(title_url, "")), ["stress", "burnout"]))',
            }
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == [
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
        ]
        assert result["value"].to_list() == ["1", "1", "0"]

    def test_timestamp_format_matches_bucket_by_clock(self):
        """Computed timestamps match the ISO format from bucket_by_clock."""
        df = _make_raw_df()
        indicators = [
            {
                "id": "indicator:c21b43949b3712e734c8",
                "name": "avg_hr",
                "source_columns": ["heart_rate"],
                "aggregation": "mean",
            },
        ]
        result = compute_indicators(df, indicators, "1d", "timestamp")
        # Should be ISO format: YYYY-MM-DDTHH:MM:SS
        assert result["timestamp"][0] == "2024-01-01T00:00:00"

    def test_timezone_aware_string_timestamps(self):
        """UTC-suffixed string timestamps should aggregate without parse errors."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    "2025-03-03T08:00:00Z",
                    "2025-03-03T12:00:00Z",
                    "2025-03-04T09:00:00Z",
                ],
                "heart_rate": [72.0, 84.0, 90.0],
            }
        )
        indicators = [
            {
                "id": "indicator:c21b43949b3712e734c8",
                "name": "avg_hr",
                "source_columns": ["heart_rate"],
                "aggregation": "mean",
            },
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert result["timestamp"].to_list() == ["2025-03-03T00:00:00", "2025-03-04T00:00:00"]
        assert [float(v) for v in result["value"].to_list()] == [78.0, 90.0]

    def test_hourly_clock(self):
        """Hourly model_clock produces every support tick in the observed span."""
        df = _make_raw_df()
        indicators = [
            {
                "id": "indicator:c21b43949b3712e734c8",
                "name": "avg_hr",
                "source_columns": ["heart_rate"],
                "aggregation": "mean",
            },
        ]
        result = compute_indicators(df, indicators, "1h", "timestamp")
        assert len(result) == 51
        assert result["timestamp"][0] == "2024-01-01T08:00:00"
        assert result["timestamp"][-1] == "2024-01-03T10:00:00"
        assert result["value"].null_count() == 45

    def test_indicator_specific_observation_window_overrides_model_clock(self):
        """Computed indicators bucket by their own support window, not the global clock."""
        df = _make_raw_df()
        indicators = [
            {
                "id": "indicator:f56b7b4807d8c6627ccb",
                "name": "weekly_steps",
                "source_columns": ["steps"],
                "aggregation": "sum",
                "observation_window": "1w",
            },
        ]

        result = compute_indicators(df, indicators, "1d", "timestamp")

        assert len(result) == 1
        assert result["timestamp"].to_list() == ["2024-01-01T00:00:00"]
        assert float(result["value"][0]) == 12000.0

    def test_col_name_parameter(self):
        """_build_agg_expr with custom col_name works correctly."""
        df = pl.DataFrame({"heart_rate": [72.0, 85.0, 68.0]})
        expr = _build_agg_expr("mean", "heart_rate")
        result = df.select(expr)
        assert abs(result["value"][0] - 75.0) < 0.01


@pytest.mark.parametrize(
    ("recording", "aggregation", "expected"),
    [
        ("samples", "last", [None, 4.0, None, 8.0]),
        ("changes", "last", [None, 4.0, 4.0, 8.0]),
        ("samples", "sum", [None, 4.0, None, 8.0]),
        ("events", "sum", [None, 4.0, 0.0, 8.0]),
        ("events", "count", [0.0, 1.0, 0.0, 1.0]),
    ],
)
def test_recording_semantics_resolve_only_declared_gaps(recording, aggregation, expected):
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec

    indicator = IndicatorSpec(
        id="indicator:record",
        name="record",
        how_to_measure="Read the recorded value",
        construct_polarity="positive",
        measurement_dtype="count",
        aggregation=aggregation,
        recording=recording,
        extraction_mode="computed",
        source_columns=("reading",),
    )
    raw = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 4), datetime(2026, 1, 1), datetime(2026, 1, 2)],
            "reading": [8.0, None, 4.0],
        }
    )
    result = compute_indicators(raw, [indicator.model_dump(mode="json")], "1d", "timestamp")
    assert result["value"].cast(pl.Float64).to_list() == expected
    # Conversions belong to the deterministic recipe and precede persistence.
    if recording == "changes":
        converted = indicator.model_copy(update={"computed_rule": "last(reading) / 2"})
        result = compute_indicators(raw, [converted.model_dump(mode="json")], "1d", "timestamp")
        assert result["value"].cast(pl.Float64).to_list() == [None, 2.0, 2.0, 4.0]
