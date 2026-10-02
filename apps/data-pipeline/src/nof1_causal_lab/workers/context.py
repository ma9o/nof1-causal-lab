"""Extraction context composes owned definitions, fixed windows and source references."""

from __future__ import annotations

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_preparation import DataVariableSpec, FileSourceRef
from nof1_causal_lab.artifacts.duration import Duration


class MeasurementContext(Value):
    """One extraction selection, retaining its authored variables and source span."""

    source: FileSourceRef
    model_clock: Duration
    indicators: tuple[DataVariableSpec, ...] = Field(min_length=1)

    def window(self, indicator: DataVariableSpec) -> Duration:
        return indicator.observation_window or self.model_clock

    def select(self, indicator: DataVariableSpec) -> MeasurementContext:
        """A chunk carries the same source and one actual observation definition."""
        return MeasurementContext(
            source=self.source, model_clock=self.model_clock, indicators=(indicator,)
        )
