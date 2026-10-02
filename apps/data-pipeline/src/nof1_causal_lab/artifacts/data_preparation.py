"""Data-owned preparation instructions, source references and observation metadata."""

from __future__ import annotations

import re
from datetime import date
from typing import TYPE_CHECKING, Annotated, Literal, Self

from polars._typing import FillNullStrategy
from pydantic import (
    AwareDatetime,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.utils.observation_semantics import (
    SummaryOperator,
    derive_indicator_observation_semantics,
)
from nof1_causal_lab.utils.window_expressions import WindowExpression

from .duration import Duration
from .identity import GitOid
from .observations import ObservationSpec

if TYPE_CHECKING:
    from nof1_causal_lab.workers.context import MeasurementContext

_SEMANTIC_COLLISIONS: tuple[tuple[str, frozenset[SummaryOperator], str], ...] = (
    (
        r"\bcount\b|\bnumber of\b|\bhow many\b",
        frozenset({SummaryOperator.MEAN, SummaryOperator.STD}),
        "how_to_measure implies counting but aggregation computes a statistic",
    ),
    (
        r"\baverage\b|\bmean\b",
        frozenset(
            {
                SummaryOperator.SUM,
                SummaryOperator.FIRST,
                SummaryOperator.LAST,
                SummaryOperator.COUNT,
            }
        ),
        "how_to_measure implies averaging but aggregation is not mean",
    ),
    (
        r"\btotal\b|\bcumulative\b|\bsum\b",
        frozenset({SummaryOperator.MEAN, SummaryOperator.FIRST, SummaryOperator.LAST}),
        "how_to_measure implies summing but aggregation is not sum",
    ),
    (
        r"\blast\b|\bmost recent\b|\bcurrent\b",
        frozenset({SummaryOperator.MEAN, SummaryOperator.SUM}),
        "how_to_measure implies point-in-time but aggregation is a window statistic",
    ),
)


def check_semantic_collisions(
    how_to_measure: str,
    aggregation: SummaryOperator,
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


class SemanticExtractionSpec(Value):
    """Interpret source records using an explicit measurement rubric."""

    kind: Literal["semantic"] = "semantic"
    how_to_measure: str = Field(
        min_length=1, description="Scoring rubric and extraction instructions."
    )
    source_columns: tuple[str, ...] = Field(
        default=(), description="Source columns exposed to the extraction worker."
    )


class ComputedExtractionSpec(Value):
    """Compute a deterministic support-window measurement from source columns."""

    kind: Literal["computed"] = "computed"
    how_to_measure: str = Field(
        min_length=1, description="Description of the deterministic measurement."
    )
    source_columns: tuple[str, ...] = Field(min_length=1)
    computed_rule: WindowExpression | None = Field(
        default=None,
        description=(
            "Optional deterministic support-window expression over the declared source columns. "
            "It must return one scalar per window with the observation's declared summary operator. "
            "Omitted uses a direct single-column aggregation."
        ),
    )
    fill_null: FillNullStrategy | Annotated[FiniteFloat, Field(strict=True)] | None = Field(
        default=None,
        description=(
            "Optional Polars null filling after aggregation on the sorted time grid within the "
            "selected data span. Use forward, backward, min, max, mean, zero, one, or a numeric "
            "constant. Fills every null, including explicit unknown readings. Omitted leaves "
            "nulls unknown. Forward carries the last value and leaves leading nulls unknown."
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

    @model_validator(mode="after")
    def validate_computation(self) -> Self:
        if self.fill_null_limit is not None and self.fill_null not in {"forward", "backward"}:
            raise ValueError("fill_null_limit requires fill_null='forward' or 'backward'")
        if self.computed_rule is None:
            if len(self.source_columns) != 1:
                raise ValueError("Direct computed extraction requires exactly 1 source_column")
        else:
            referenced = self.computed_rule.dependencies
            if not referenced:
                raise ValueError("computed_rule must reference at least 1 source_column")
            if unknown := sorted(referenced - set(self.source_columns)):
                raise ValueError(f"computed_rule references undeclared source_columns: {unknown}")
        return self


type ExtractionSpec = Annotated[
    ComputedExtractionSpec | SemanticExtractionSpec, Field(discriminator="kind")
]


class DataVariableSpec(Value):
    """Compose an observed variable with its data-owned extraction instructions."""

    observation: ObservationSpec
    extraction: ExtractionSpec

    @model_validator(mode="after")
    def validate_computed_summary(self) -> Self:
        if isinstance(self.extraction, ComputedExtractionSpec):
            derive_indicator_observation_semantics(
                self.observation.aggregation,
                self.observation.measurement_dtype,
                self.extraction.computed_rule,
            )
        return self


def _validate_uploaded_filename(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError("Use uploaded filenames, without directory components")
    return value


class FileSourceRef(Value):
    """Explicit uploaded filenames, relative to this study's input directory."""

    files: tuple[str, ...] = Field(min_length=1)
    start: date | None = Field(default=None, description="Inclusive UTC source-coverage date.")
    end: date | None = Field(default=None, description="Exclusive UTC source-coverage date.")

    @field_validator("files")
    @classmethod
    def validate_filenames(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(values)) != len(values):
            raise ValueError("Source filenames must be unique")
        return tuple(_validate_uploaded_filename(value) for value in values)

    @model_validator(mode="after")
    def ordered_bounds(self) -> Self:
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("Source coverage start must precede end")
        return self


class SimulationReplicateRef(Value):
    """One replicate from a recorded, applied simulation in this study."""

    revision: GitOid
    replicate: int = Field(ge=0)


type DataSourceRef = Annotated[
    FileSourceRef | SimulationReplicateRef,
    Field(description="Uploaded sources or one recorded simulation replicate."),
]


class DataPreparationSpec(Value):
    """A versioned data definition supplied directly to prepare_data."""

    default_window: Duration
    variables: tuple[DataVariableSpec, ...] = Field(min_length=1)
    context: str = Field(
        default="", description="Optional context for interpreting the source data."
    )

    @model_validator(mode="after")
    def unique_variables(self) -> Self:
        if len({item.observation.id for item in self.variables}) != len(self.variables):
            raise ValueError("Prepared variables must have unique IDs")
        return self

    def observation_schema(self) -> tuple[ObservationSpec, ...]:
        return tuple(
            item.observation.resolved(
                (item.observation.observation_window or self.default_window).source
            )
            for item in self.variables
        )


class FilePreparationSpec(Value):
    """Uploaded sources and the complete recipe for preparing their observations."""

    source: FileSourceRef
    definition: DataPreparationSpec

    def extraction_context(self) -> MeasurementContext:
        from nof1_causal_lab.workers.context import MeasurementContext

        return MeasurementContext(
            source=self.source,
            model_clock=self.definition.default_window,
            indicators=self.definition.variables,
        )


class PreparedDataMetadata(Value):
    """Self-contained semantics and provenance of one prepared observation table."""

    source: DataSourceRef
    variables: tuple[ObservationSpec, ...] = Field(min_length=1)
    preparation: DataPreparationSpec | None = None
    time_origin: AwareDatetime | None = Field(
        description="Calendar instant of model day zero; null denotes a calendar-free history."
    )

    @model_validator(mode="after")
    def resolved_variables(self) -> Self:
        if len({item.id for item in self.variables}) != len(self.variables):
            raise ValueError("Prepared variables must have unique IDs")
        if any(item.observation_window is None for item in self.variables):
            raise ValueError("Prepared variables must record their resolved observation windows")
        if isinstance(self.source, FileSourceRef) != (self.preparation is not None):
            raise ValueError(
                "File sources require preparation instructions; simulation "
                "sources retain their declared schema without extraction"
            )
        if self.preparation is not None and self.variables != self.preparation.observation_schema():
            raise ValueError("Prepared schema must match its preparation instructions")
        return self
