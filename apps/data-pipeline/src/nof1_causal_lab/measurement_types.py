"""Shared measurement-domain type aliases."""

from __future__ import annotations

from typing import Literal

type MeasurementDtype = Literal["continuous", "binary", "count", "ordinal", "categorical"]
