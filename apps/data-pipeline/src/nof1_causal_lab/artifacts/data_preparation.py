"""Data-owned preparation instructions, source references and observation metadata."""

from __future__ import annotations

import ast
import re
from typing import Annotated, Literal, Self, override

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from nof1_causal_lab.json_types import JsonObject  # noqa: TC001
from nof1_causal_lab.measurement_types import AggregationFunction  # noqa: TC001
from nof1_causal_lab.utils.aggregations import COMPUTED_RULE_FUNCTIONS
from nof1_causal_lab.utils.observation_semantics import (
    IndicatorObservationSemantics,
    derive_indicator_observation_semantics,
)

from .duration import parse_duration_to_hours
from .identity import GitOid  # noqa: TC001
from .observations import ObservationSpec

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


class DataVariableSpec(ObservationSpec):
    """How to produce one observed variable, without any causal or latent model."""

    how_to_measure: str = Field(min_length=1, description="Scoring rubric and extraction instructions.")
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
    def validate_recording(self) -> DataVariableSpec:
        if self.recording != "samples" and self.extraction_mode != "computed":
            raise ValueError("Complete event/change records require computed extraction")
        if self.recording == "events" and self.aggregation not in {"sum", "count"}:
            raise ValueError("Complete event records require sum/count aggregation")
        if self.recording == "changes" and self.aggregation != "last":
            raise ValueError("Change records require last aggregation")
        return self

    @model_validator(mode="after")
    def validate_computed_mode(self) -> DataVariableSpec:
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
        self._observation_semantics()
        return self

    @override
    def _observation_semantics(self) -> IndicatorObservationSemantics:
        return derive_indicator_observation_semantics(
            self.aggregation, self.measurement_dtype, self.computed_rule
        )


class FileSourceRef(BaseModel):
    """Explicit uploaded filenames, relative to this study's input directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    files: tuple[str, ...] = Field(min_length=1)

    @field_validator("files")
    @classmethod
    def validate_filenames(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("Source filenames must be unique")
        if any(not value or value in {".", ".."} or "/" in value or "\\" in value for value in values):
            raise ValueError("Use uploaded filenames, without directory components")
        return values


class SimulationReplicateRef(BaseModel):
    """One replicate from a recorded, applied simulation in this study."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: GitOid
    replicate: int = Field(ge=0)


type DataSourceRef = FileSourceRef | SimulationReplicateRef


class DataPreparationSpec(BaseModel):
    """A versioned data definition supplied directly to prepare_data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_window: str
    variables: tuple[DataVariableSpec, ...] = Field(min_length=1)
    context: str = Field(default="", description="Optional context for interpreting the source data.")

    @field_validator("default_window")
    @classmethod
    def validate_window(cls, value: str) -> str:
        parse_duration_to_hours(value)
        return value

    @model_validator(mode="after")
    def unique_variables(self) -> Self:
        if len({item.id for item in self.variables}) != len(self.variables):
            raise ValueError("Prepared variables must have unique IDs")
        return self

    def observation_schema(self) -> tuple[ObservationSpec, ...]:
        return tuple(
            ObservationSpec.model_validate({
                **item.model_dump(include=set(ObservationSpec.model_fields)),
                "observation_window": item.observation_window or self.default_window,
            })
            for item in self.variables
        )

    def extraction_context(self) -> JsonObject:
        return {
            "model_clock": self.default_window,
            "indicators": [item.model_dump(mode="json") for item in self.variables],
        }


class PreparedDataMetadata(BaseModel):
    """Self-contained semantics and provenance of one prepared observation table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: DataSourceRef
    variables: tuple[ObservationSpec, ...] = Field(min_length=1)
    preparation: DataPreparationSpec | None = None

    @model_validator(mode="after")
    def resolved_variables(self) -> Self:
        if len({item.id for item in self.variables}) != len(self.variables):
            raise ValueError("Prepared variables must have unique IDs")
        if any(item.observation_window is None for item in self.variables):
            raise ValueError("Prepared variables must record their resolved observation windows")
        if isinstance(self.source, FileSourceRef) != (self.preparation is not None):
            raise ValueError("File sources require preparation instructions; simulation sources retain their recorded schema")
        if self.preparation is not None and self.variables != self.preparation.observation_schema():
            raise ValueError("Prepared schema must match its preparation instructions")
        return self
