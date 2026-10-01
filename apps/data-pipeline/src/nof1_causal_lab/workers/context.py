"""Wire projections shared by extraction workers and stored extraction plans.

Domain definitions are validated by DataPreparationSpec/ModelSpec before projection.
Stored worker contexts are decoded with the adapter at their read boundary.
"""

from typing import Literal, NotRequired

from polars._typing import FillNullStrategy
from typing_extensions import TypedDict

from nof1_causal_lab.artifacts.identity import IndicatorId
from nof1_causal_lab.measurement_types import AggregationFunction, MeasurementDtype


class MeasurementIndicator(TypedDict):
    id: IndicatorId
    name: NotRequired[str]
    measurement_dtype: MeasurementDtype
    aggregation: AggregationFunction
    observation_window: NotRequired[str | None]
    how_to_measure: NotRequired[str]
    source_columns: NotRequired[list[str]]
    computed_rule: NotRequired[str | None]
    extraction_mode: NotRequired[Literal["computed", "semantic"]]
    fill_null: NotRequired[FillNullStrategy | float | None]
    fill_null_limit: NotRequired[int | None]
    ordinal_levels: NotRequired[list[str] | None]
    categorical_levels: NotRequired[list[str] | None]
    support_kind: NotRequired[str]
    summary_operator: NotRequired[str]
    anchor_policy: NotRequired[str]
    construct_id: NotRequired[str]
    construct_name: NotRequired[str]
    source_id: NotRequired[str]


class MeasurementContext(TypedDict):
    model_clock: str | None
    indicators: list[MeasurementIndicator]


class IndicatorMeasurementInfo(TypedDict):
    dtype: MeasurementDtype
    ordinal_levels: list[str] | None
    categorical_levels: list[str] | None
    support_kind: str
    summary_operator: str
    anchor_policy: str
    observation_window: str | None
