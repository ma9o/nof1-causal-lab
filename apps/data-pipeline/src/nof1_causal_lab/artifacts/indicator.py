"""Observed indicators own their measurement choices and likelihoods."""

from __future__ import annotations

import ast
import logging
import re
from enum import StrEnum
from typing import Annotated, Literal, get_args, override

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nof1_causal_lab.distributions import VALID_LIKELIHOODS_FOR_DTYPE
from nof1_causal_lab.measurement_types import AggregationFunction, MeasurementDtype
from nof1_causal_lab.utils.aggregations import COMPUTED_RULE_FUNCTIONS
from nof1_causal_lab.utils.observation_semantics import (
    AnchorPolicy,
    IndicatorObservationSemantics,
    SummaryOperator,
    SupportKind,
    derive_indicator_observation_semantics,
    supported_summary_operators_text,
)

from .duration import parse_duration_to_hours
from .identity import IndicatorId  # noqa: TC001
from .likelihood import LikelihoodSpec  # noqa: TC001

logger = logging.getLogger(__name__)

VALID_AGGREGATIONS: set[str] = set(get_args(AggregationFunction.__value__))

_SEMANTIC_COLLISIONS: list[tuple[str, set[str], str]] = [
    (
        r"\bcount\b|\bnumber of\b|\bhow many\b",
        {"mean", "median", "std", "var"},
        "how_to_measure implies counting but aggregation computes a statistic",
    ),
    (
        r"\baverage\b|\bmean\b",
        {"sum", "first", "last", "count"},
        "how_to_measure implies averaging but aggregation is not mean/median",
    ),
    (
        r"\btotal\b|\bcumulative\b|\bsum\b",
        {"mean", "median", "first", "last"},
        "how_to_measure implies summing but aggregation is not sum",
    ),
    (
        r"\blast\b|\bmost recent\b|\bcurrent\b",
        {"mean", "sum", "median"},
        "how_to_measure implies point-in-time but aggregation is a window statistic",
    ),
]


def check_semantic_collisions(
    how_to_measure: str,
    aggregation: AggregationFunction,
) -> list[str]:
    """Check for inconsistencies between how_to_measure text and aggregation."""
    warnings: list[str] = []
    text_lower = how_to_measure.lower()
    for pattern, conflict_aggs, explanation in _SEMANTIC_COLLISIONS:
        match = re.search(pattern, text_lower)
        if aggregation in conflict_aggs and match:
            warnings.append(
                f"Semantic collision: {explanation}. "
                f"how_to_measure contains '{match.group()}' "
                f"but aggregation='{aggregation}'."
            )
    return warnings


class IndicatorPolarity(StrEnum):
    """Indicator polarity states whether a measurement increases or decreases with its
    construct.
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"  # noqa: V107 - native enum value construction


def _parse_computed_rule_expr(expr: str) -> ast.Expression:
    """Parse a computed-rule expression and surface a stable error."""
    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid computed_rule: {exc.msg}") from exc
    return parsed


def _computed_rule_source_names(expr: str) -> set[str]:
    """Collect source-column references from a computed-rule expression."""
    parsed = _parse_computed_rule_expr(expr)
    names: set[str] = set()

    class _NameCollector(ast.NodeVisitor):
        @override
        def visit_Call(self, node: ast.Call) -> None:
            if not isinstance(node.func, ast.Name):
                raise ValueError("computed_rule only supports simple function calls")
            if node.func.id not in COMPUTED_RULE_FUNCTIONS:
                available = ", ".join(sorted(COMPUTED_RULE_FUNCTIONS))
                raise ValueError(
                    f"Unsupported computed_rule function '{node.func.id}'. Available: {available}"
                )
            if node.keywords:
                raise ValueError("computed_rule does not support keyword arguments")
            for arg in node.args:
                self.visit(arg)

        @override
        def visit_Attribute(self, node: ast.Attribute) -> None:
            _ = node
            raise ValueError("computed_rule does not support attribute access")

        @override
        def visit_Name(self, node: ast.Name) -> None:
            if node.id not in COMPUTED_RULE_FUNCTIONS:
                names.add(node.id)

    _NameCollector().visit(parsed.body)
    return names


def _validate_window_expression(value: str) -> str:
    _computed_rule_source_names(value)
    return value


type WindowExpression = Annotated[
    str,
    AfterValidator(_validate_window_expression),
    Field(
        description=(
            "Deterministic support-window expression that returns one scalar per window. "
            "Use Python-like syntax over source_columns with arithmetic, comparisons, "
            "if/else, and helper functions such as any(), sum(), mean(), std(), "
            "first(), last(), count_true(), count_non_null(), lower(), contains(), "
            "and contains_any(). Use None for missing values."
        )
    ),
]


class IndicatorSpec(BaseModel):
    """A specification of a construct's observed measurement and how to extract it."""

    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")

    id: IndicatorId = Field(description="Persistent identity. Preserve when revising or renaming.")
    likelihood: LikelihoodSpec | None = None
    name: str = Field(description="Indicator name (e.g., 'hrv', 'self_reported_stress')")
    how_to_measure: str = Field(
        description="Instructions for workers on how to extract this from data"
    )
    construct_polarity: IndicatorPolarity = Field(
        description=(
            "Whether higher indicator values move in the same direction as the construct "
            "(`positive`) or the opposite direction (`negative`)."
        )
    )
    measurement_dtype: MeasurementDtype = Field(
        description="'continuous', 'binary', 'count', 'ordinal', 'categorical'"
    )
    aggregation: AggregationFunction = Field(
        description=(
            "Aggregation function applied when bucketing raw extractions within the "
            "indicator support window. Measurement-structure support is currently limited to: "
            f"{supported_summary_operators_text()}. Available parser operators: {', '.join(sorted(VALID_AGGREGATIONS))}"
        ),
    )
    recording: Literal["samples", "events", "changes"] = Field(
        default="samples",
        description=(
            "Source recording semantics within the raw dataset's covered time span. "
            "samples: absent readings are unknown. events: a complete event record; "
            "empty sum/count windows are zero. changes: a complete change record; "
            "the last recorded value persists, with leading gaps unknown. "
            "events and changes require computed extraction."
        ),
    )
    observation_window: str | None = Field(
        default=None,
        description=(
            "Optional duration string describing the support window summarized by this "
            "indicator (for example '1mo' for a monthly average on a daily model clock). "
            "If omitted, the support window defaults to the global model_clock."
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
    source_columns: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Raw data column names referenced by how_to_measure. "
            "Used to project chunks to only relevant columns before extraction."
        ),
    )
    computed_rule: WindowExpression | None = Field(
        default=None,
        description=(
            "Optional deterministic support-window expression for extraction_mode='computed'. "
            "Use this when a computed indicator needs formulas, thresholds, or multiple "
            "source columns instead of a direct single-column aggregation. "
            "The expression must return one scalar per support window."
        ),
    )
    extraction_mode: Literal["computed", "semantic"] = Field(
        default="semantic",
        description=(
            "'computed' (deterministic pipeline extraction) or 'semantic' (LLM extraction). "
            "Use 'computed' when the indicator can be derived deterministically either from "
            "a direct source-column aggregation or from a computed_rule support-window "
            "expression over the declared source_columns."
        ),
    )

    @model_validator(mode="after")
    def validate_recording(self) -> IndicatorSpec:
        if self.recording != "samples" and self.extraction_mode != "computed":
            raise ValueError("Complete event/change records require computed extraction")
        if self.recording == "events" and self.aggregation not in {"sum", "count"}:
            raise ValueError("Complete event records require sum/count aggregation")
        if self.recording == "changes" and self.aggregation != "last":
            raise ValueError("Change records require last aggregation")
        return self

    @field_validator("observation_window")
    @classmethod
    def validate_observation_window(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parse_duration_to_hours(value)
        return value

    @model_validator(mode="after")
    def validate_discrete_levels(self) -> IndicatorSpec:
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
    def warn_semantic_collisions(self) -> IndicatorSpec:
        """Log warnings when how_to_measure text conflicts with aggregation."""
        collisions = check_semantic_collisions(self.how_to_measure, self.aggregation)
        for warning in collisions:
            logger.warning("Indicator '%s': %s", self.name, warning)
        return self

    @model_validator(mode="after")
    def validate_computed_mode(self) -> IndicatorSpec:
        """Enforce constraints when extraction_mode='computed'."""
        if self.computed_rule is not None and self.extraction_mode != "computed":
            raise ValueError(
                f"Indicator '{self.name}' sets computed_rule but extraction_mode is "
                f"'{self.extraction_mode}'. computed_rule is only valid for "
                "extraction_mode='computed'."
            )
        if self.extraction_mode != "computed":
            return self
        if self.computed_rule is None and len(self.source_columns) != 1:
            raise ValueError(
                f"Computed indicator '{self.name}' currently requires exactly 1 direct "
                f"source_column, got {len(self.source_columns)}: {self.source_columns}"
            )
        if self.computed_rule is not None:
            if not self.source_columns:
                raise ValueError(
                    f"Computed indicator '{self.name}' with computed_rule must declare "
                    "at least 1 source_column."
                )
            referenced = _computed_rule_source_names(self.computed_rule)
            if not referenced:
                raise ValueError(
                    f"Computed indicator '{self.name}' has computed_rule "
                    "that does not reference any source_columns."
                )
            unknown = sorted(referenced - set(self.source_columns))
            if unknown:
                raise ValueError(
                    f"Computed indicator '{self.name}' computed_rule "
                    f"references undeclared source_columns: {unknown}. "
                    f"Declared source_columns: {self.source_columns}"
                )
        return self

    @model_validator(mode="after")
    def validate_observation_semantics(self) -> IndicatorSpec:
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

    @model_validator(mode="after")
    def validate_likelihood(self) -> IndicatorSpec:
        if (
            self.likelihood is not None
            and self.likelihood.law.family
            not in VALID_LIKELIHOODS_FOR_DTYPE[self.measurement_dtype]
        ):
            raise ValueError(
                f"Likelihood {self.likelihood.law.family.value!r} is incompatible with {self.measurement_dtype!r} indicator {self.id!r}"
            )
        return self
