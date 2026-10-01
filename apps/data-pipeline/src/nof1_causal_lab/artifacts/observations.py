"""Observed-variable identity, codebooks and time-support semantics."""

from __future__ import annotations

from typing import Annotated, get_args

from polars._typing import (
    FillNullStrategy,
)
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from nof1_causal_lab.measurement_types import AggregationFunction, MeasurementDtype
from nof1_causal_lab.utils.observation_semantics import (
    AnchorPolicy,
    IndicatorObservationSemantics,
    SummaryOperator,
    SupportKind,
    derive_indicator_observation_semantics,
    supported_summary_operators_text,
)

from .duration import parse_duration_to_hours
from .identity import IndicatorId

VALID_AGGREGATIONS: set[str] = set(get_args(AggregationFunction.__value__))


class ObservationSpec(BaseModel):
    """A stable observed variable, reusable across scientific model definitions."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    id: IndicatorId = Field(description="Persistent identity. Preserve when revising or renaming.")
    name: str = Field(description="Indicator name (e.g., 'hrv', 'self_reported_stress')")
    measurement_dtype: MeasurementDtype = Field(
        description="'continuous', 'binary', 'count', 'ordinal', 'categorical'"
    )
    aggregation: AggregationFunction = Field(
        description=(
            "Aggregation function applied when bucketing raw extractions within the "
            "indicator support window. Measurement-structure support is currently limited to: "
            f"{supported_summary_operators_text()}. A computed_rule must produce this same summary. "
            f"Available parser operators: {', '.join(sorted(VALID_AGGREGATIONS))}"
        ),
    )
    observation_window: str | None = Field(
        default=None,
        description=(
            "Optional duration string describing the support window summarized by this "
            "indicator (for example '1mo' for a monthly average on a daily model clock). "
            "Resolved by the preparation window or the generative model clock."
        ),
    )
    fill_null: FillNullStrategy | Annotated[FiniteFloat, Field(strict=True)] | None = Field(
        default=None,
        description=(
            "Optional Polars null filling during preparation, after aggregation on the sorted "
            "time grid within the selected data span. Use forward, backward, min, max, mean, "
            "zero, one, or a numeric constant. Fills every null, including explicit unknown "
            "readings. Omitted leaves nulls unknown. Forward carries the last value and leaves "
            "leading nulls unknown."
        ),
    )
    fill_null_limit: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Maximum consecutive nulls filled by forward/backward; omitted is unlimited. "
            "Only valid when fill_null is forward or backward."
        ),
    )
    ordinal_levels: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Ordered list of level labels from lowest to highest for ordinal indicators "
            "(e.g., ['low', 'medium', 'high']). Required when measurement_dtype='ordinal' "
            "to ensure correct numeric encoding."
        ),
    )
    categorical_levels: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Exhaustive list of level labels for categorical indicators "
            "(e.g., ['home', 'work', 'other']). Required when "
            "measurement_dtype='categorical' to ensure correct numeric encoding."
        ),
    )

    @field_validator("observation_window")
    @classmethod
    def validate_observation_window(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parse_duration_to_hours(value)
        return value

    @model_validator(mode="after")
    def validate_fill_null_limit(self) -> ObservationSpec:
        if self.fill_null_limit is not None and self.fill_null not in {"forward", "backward"}:
            raise ValueError("fill_null_limit requires fill_null='forward' or 'backward'")
        return self

    @model_validator(mode="after")
    def validate_discrete_levels(self) -> ObservationSpec:
        """Require at least two unique labels for ordinal and categorical indicators."""
        if self.measurement_dtype not in {"ordinal", "categorical"}:
            return self

        field_name = f"{self.measurement_dtype}_levels"
        levels = (
            self.ordinal_levels if self.measurement_dtype == "ordinal" else self.categorical_levels
        )
        label_description = (
            "ordered level labels" if self.measurement_dtype == "ordinal" else "level labels"
        )
        if not levels:
            raise ValueError(
                f"{field_name} is required when measurement_dtype={self.measurement_dtype!r} "
                f"(provide at least 2 {label_description})"
            )
        if len(levels) < 2:
            raise ValueError(f"{field_name} must have at least 2 items, got {len(levels)}")
        if len(levels) != len(set(levels)):
            raise ValueError(f"{field_name} must not contain duplicate labels")
        return self

    @model_validator(mode="after")
    def validate_observation_semantics(self) -> ObservationSpec:
        """Reject aggregation/dtype combinations the measurement stack cannot model."""
        derive_indicator_observation_semantics(self.aggregation, self.measurement_dtype)
        return self

    def _observation_semantics(self) -> IndicatorObservationSemantics:
        return derive_indicator_observation_semantics(self.aggregation, self.measurement_dtype)

    @property
    def support_kind(self) -> SupportKind:
        """Whether this indicator is point-local or interval-summary."""
        return self._observation_semantics().support_kind

    @property
    def summary_operator(self) -> SummaryOperator:
        """Canonical summary operator used by extraction and likelihoods."""
        return self._observation_semantics().summary_operator

    @property
    def anchor_policy(self) -> AnchorPolicy:
        """Which support boundary receives the observation anchor."""
        return self._observation_semantics().anchor_policy

    @property
    def requires_interval_summary_measurement(self) -> bool:
        """Whether this indicator requires an interval-summary measurement equation."""
        return self.support_kind == SupportKind.INTERVAL
