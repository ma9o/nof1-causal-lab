"""Shared measurement-domain type aliases."""

from __future__ import annotations

from typing import Literal

type MeasurementDtype = Literal["continuous", "binary", "count", "ordinal", "categorical"]

type AggregationFunction = Literal[
    "mean",
    "sum",
    "min",
    "max",
    "std",
    "var",
    "last",
    "first",
    "count",
    "median",
    "p10",
    "p25",
    "p75",
    "p90",
    "p99",
    "skew",
    "kurtosis",
    "iqr",
    "range",
    "cv",
    "entropy",
    "instability",
    "trend",
    "n_unique",
]
