"""Read-only comparison of saved observation datasets and simulation replicates."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Annotated, Literal, Self

import numpy as np
import polars as pl
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from nof1_causal_lab.artifacts.duration import parse_duration_to_hours
from nof1_causal_lab.artifacts.effects import HistogramBin
from nof1_causal_lab.artifacts.identity import GitOid, IndicatorId
from nof1_causal_lab.artifacts.observations import ObservationSpec
from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorPredictiveChecks
from nof1_causal_lab.utils.histograms import histogram_draws

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime


class DataRef(BaseModel):
    """A saved panel or simulation; omitting replicate selects every saved simulation draw."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["panel", "simulation"]
    revision: GitOid
    replicate: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def source_coordinates(self) -> Self:
        if self.kind == "panel" and self.replicate is not None:
            raise ValueError("Panels already define their history and calendar timestamps")
        return self


type DataSelection = Annotated[
    DataRef | Annotated[tuple[DataRef, ...], Field(min_length=1)],
    Field(description="A data selection identifies one or more saved observation histories."),
]


class DataDiffRequest(BaseModel):
    """Compare two immutable data selections, each containing one or more histories."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    action: Literal["data_diff"] = "data_diff"
    left: DataSelection
    right: DataSelection


class DataPoint(BaseModel):
    """An observed anchor and support; dates are synthetic for a calendar-free series."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    anchor_time: AwareDatetime
    support_start: AwareDatetime | None
    support_end: AwareDatetime | None
    value: FiniteFloat | None


class DataSeries(BaseModel):
    """One variable's recorded measurements in one history; no pooling across replicas."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    variable: ObservationSpec | None
    time_origin: AwareDatetime | None = Field(
        description="Recorded calendar binding; null means the point dates are serialization coordinates, not real dates."
    )
    points: tuple[DataPoint, ...]


class DataPointChange(BaseModel):
    """An added, removed or revised measurement in a single-history comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    anchor_time: AwareDatetime
    change: Literal["added", "removed", "revised"]
    left: DataPoint | None
    right: DataPoint | None


class DataStatisticComparison(BaseModel):
    """The same descriptive statistic measured independently in every selected history."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    statistic: Literal["observed_count", "missing_count", "mean", "sd", "min", "max", "proportion"]
    level: str | None = None
    left: tuple[FiniteFloat | None, ...]
    right: tuple[FiniteFloat | None, ...]
    left_histogram: tuple[HistogramBin, ...]
    right_histogram: tuple[HistogramBin, ...]


class DataVariableDiff(BaseModel):
    """Definitions, histories and comparisons for one persistent observation identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    indicator_id: IndicatorId
    left: tuple[DataSeries, ...]
    right: tuple[DataSeries, ...]
    changes: tuple[DataPointChange, ...]
    statistics: tuple[DataStatisticComparison, ...]
    comparison_issues: tuple[str, ...]
    reference_side: Literal["left", "right"] | None
    predictive_checks: PosteriorPredictiveChecks | None
    predictive_unavailable_reason: str | None


class DataDiffReport(BaseModel):
    """Comparisons of existing data, preserving each history's immutable source reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    left: tuple[DataRef, ...]
    right: tuple[DataRef, ...]
    variables: tuple[DataVariableDiff, ...]


@dataclass(frozen=True)
class Dataset:
    """One loaded observation table with its own schema and exact source."""

    source: DataRef
    variables: tuple[ObservationSpec, ...]
    observations: pl.DataFrame
    time_origin: datetime | None


def read_data_diff(workspace_id: str, request: DataDiffRequest) -> DataDiffReport:
    """Load existing data only; never read a model for generation, fit, or write artifacts."""
    from nof1_causal_lab.actions.prepare_data import read_simulation_observations
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.lineage import read_data_metadata
    from nof1_causal_lab.study.store import ArtifactStore, read_model

    store = ArtifactStore(workspace_id)
    read_array = cache(store.read_array)
    input_indicators: set[IndicatorId] = set()

    @cache
    def load(source: DataRef) -> tuple[Dataset, ...]:
        if source.kind == "panel":
            metadata = read_data_metadata(store, source.revision)
            return (
                Dataset(
                    source,
                    metadata.variables,
                    store.read_parquet_file(
                        "panel",
                        source.revision,
                        "panel.parquet",
                    ),
                    metadata.time_origin,
                ),
            )
        record = StudyRepository(workspace_id).record(source.revision)
        if record.status != "applied" or record.action != "simulate":
            raise ValueError("Simulation data must select an applied simulation commit")
        report = SimulationReport.model_validate(record.diagnostics["report"])
        if report.model.workspace_id != workspace_id:
            raise ValueError("The simulation must belong to the selected study")
        model = read_model(store, report.model.revision)
        input_indicators.update(
            indicator.id
            for construct in model.constructs
            if construct.role == "exogenous"
            for indicator in construct.indicators
        )
        indices = range(report.draws) if source.replicate is None else (source.replicate,)
        result = []
        for replicate in indices:
            panel = read_simulation_observations(report, replicate, read_array=read_array)
            result.append(
                Dataset(
                    type(source).model_validate({**source.model_dump(), "replicate": replicate}),
                    report.observation_layout.variables,
                    panel,
                    report.time_origin,
                )
            )
        return tuple(result)

    def selection(value: DataSelection) -> tuple[Dataset, ...]:
        refs = (value,) if isinstance(value, DataRef) else value
        return tuple(dataset for source in refs for dataset in load(source))

    left, right = selection(request.left), selection(request.right)
    return data_diff(left, right, input_indicators=input_indicators)


def _series(dataset: Dataset) -> dict[IndicatorId, DataSeries]:
    from nof1_causal_lab.utils.observation_rows import validate_observation_rows

    frame = validate_observation_rows(dataset.observations, dataset.variables).with_columns(
        pl.col("anchor_time", "support_start", "support_end").dt.replace_time_zone("UTC")
    )
    return {
        variable.id: DataSeries(
            variable=variable,
            time_origin=dataset.time_origin,
            points=tuple(
                DataPoint.model_validate(row)
                for row in frame.filter(pl.col("indicator_id") == variable.id)
                .select("anchor_time", "support_start", "support_end", "value")
                .iter_rows(named=True)
            ),
        )
        for variable in dataset.variables
    }


def _semantics(variable: ObservationSpec):
    if variable.observation_window is None:
        raise ValueError("Dataset variables must define their measurement windows")
    return (
        variable.measurement_dtype,
        variable.aggregation,
        variable.support_kind,
        variable.summary_operator,
        variable.anchor_policy,
        variable.ordinal_levels,
        variable.categorical_levels,
        parse_duration_to_hours(variable.observation_window),
    )


def _statistics(
    left: tuple[DataSeries, ...], right: tuple[DataSeries, ...]
) -> tuple[DataStatisticComparison, ...]:
    def measure(series: DataSeries):
        values = np.asarray([point.value for point in series.points if point.value is not None])
        statistics = {
            ("observed_count", None): float(len(values)),
            ("missing_count", None): float(len(series.points) - len(values)),
        }
        variable = series.variable
        if variable is None:
            return statistics
        levels = variable.categorical_levels or variable.ordinal_levels
        if levels is not None:
            statistics.update(
                {
                    ("proportion", level): float(np.mean(values == index)) if len(values) else None
                    for index, level in enumerate(levels)
                }
            )
        else:
            for name, operation in (
                ("mean", np.mean),
                ("sd", np.std),
                ("min", np.min),
                ("max", np.max),
            ):
                statistics[name, None] = float(operation(values)) if len(values) else None
        return statistics

    a, b = tuple(map(measure, left)), tuple(map(measure, right))
    keys = sorted(set().union(*(item.keys() for item in (*a, *b))))
    result = []
    for statistic, level in keys:
        sides = [tuple(item.get((statistic, level)) for item in side) for side in (a, b)]
        finite = [[value for value in side if value is not None] for side in sides]
        result.append(
            DataStatisticComparison(
                statistic=statistic,
                level=level,
                left=sides[0],
                right=sides[1],
                left_histogram=tuple(histogram_draws(finite[0])) if finite[0] else (),
                right_histogram=tuple(histogram_draws(finite[1])) if finite[1] else (),
            )
        )
    return tuple(result)


def _predictive_comparison(
    identity: IndicatorId, left: tuple[DataSeries, ...], right: tuple[DataSeries, ...]
):
    if len(left) == len(right) == 1:
        return None, None, None
    if len(left) > 1 and len(right) > 1:
        return None, None, "Requires one reference history and multiple replicated histories"
    side, reference, replicas = (
        ("left", left[0], right) if len(left) == 1 else ("right", right[0], left)
    )
    variable = reference.variable
    # Shared input problems belong to comparison_issues, not a second PPC reason.
    if any((item.time_origin is None) != (reference.time_origin is None) for item in replicas):
        return side, None, None
    if variable is None or any(item.variable is None for item in replicas):
        return side, None, None
    if any(
        _semantics(item.variable) != _semantics(variable)
        for item in replicas
        if item.variable is not None
    ):
        return side, None, None
    if variable.measurement_dtype in {"categorical", "ordinal"}:
        return side, None, "Discrete codebooks use per-level proportions, not numeric PPC summaries"
    if not any(point.value is not None for point in reference.points):
        return side, None, "The reference history contains no observed values"
    aligned = []
    for series in replicas:
        lookup = {point.anchor_time: point for point in series.points}
        values = []
        for point in reference.points:
            candidate = lookup.get(point.anchor_time)
            if point.value is not None:
                if candidate is None or (candidate.support_start, candidate.support_end) != (
                    point.support_start,
                    point.support_end,
                ):
                    return side, None, None
                if candidate.value is None:
                    return side, None, "Replicas contain missing values at observed anchors"
            values.append(
                candidate.value if candidate is not None and candidate.value is not None else np.nan
            )
        aligned.append(values)
    import jax.numpy as jnp

    from nof1_causal_lab.models.posterior_predictive import measure_predictive_checks

    observed = [point.value if point.value is not None else np.nan for point in reference.points]
    checks = measure_predictive_checks(
        jnp.asarray(aligned)[:, :, None], jnp.asarray(observed)[:, None], (identity,)
    )
    return side, checks, None


def data_diff(
    left: Dataset | Sequence[Dataset],
    right: Dataset | Sequence[Dataset],
    *,
    input_indicators: set[IndicatorId] | frozenset[IndicatorId] = frozenset(),
) -> DataDiffReport:
    """Compare one or many saved histories on each side without pooling or resimulation."""
    sides = tuple(
        (value,) if isinstance(value, Dataset) else tuple(value) for value in (left, right)
    )
    for side in sides:
        if not side:
            raise ValueError("Each comparison side requires at least one dataset")
        if len(
            {
                (dataset.source.kind, dataset.source.revision, dataset.source.replicate)
                for dataset in side
            }
        ) != len(side):
            raise ValueError("A dataset cannot be counted twice within a comparison side")
    left_series, right_series = (tuple(map(_series, side)) for side in sides)
    variables = sorted(set().union(*(item.keys() for item in (*left_series, *right_series))))
    has_simulation = any(dataset.source.kind == "simulation" for side in sides for dataset in side)
    comparisons = []
    absent = DataSeries(variable=None, time_origin=None, points=())
    for identity in variables:
        if has_simulation and identity in input_indicators:
            continue
        a, b = (
            tuple(item.get(identity, absent) for item in side)
            for side in (left_series, right_series)
        )
        issues = []
        series = (*a, *b)
        definitions = [item.variable for item in series if item.variable is not None]
        mixed_calendars = (
            len({item.time_origin is None for item in series if item.variable is not None}) > 1
        )
        if mixed_calendars:
            issues.append("Calendar-free histories cannot be aligned to calendar-bound histories")
        if len(definitions) != len(series):
            issues.append("Variable is absent from one or more histories")
        if len({_semantics(item) for item in definitions}) > 1:
            issues.append(
                "Measurement definitions differ; statistics describe each side separately"
            )
        schedules = {
            tuple((p.anchor_time, p.support_start, p.support_end) for p in item.points)
            for item in series
        }
        if len(schedules) > 1:
            issues.append("Observation schedules or measurement windows differ")
        changes = []
        if len(a) == len(b) == 1 and not mixed_calendars:
            old, new = (
                {point.anchor_time: point for point in item.points} for item in (a[0], b[0])
            )
            for anchor in sorted(old.keys() | new.keys()):
                before, after = old.get(anchor), new.get(anchor)
                if before != after:
                    changes.append(
                        DataPointChange(
                            anchor_time=anchor,
                            change="added"
                            if before is None
                            else "removed"
                            if after is None
                            else "revised",
                            left=before,
                            right=after,
                        )
                    )
        reference, checks, reason = _predictive_comparison(identity, a, b)
        comparisons.append(
            DataVariableDiff(
                indicator_id=identity,
                left=a,
                right=b,
                changes=tuple(changes),
                statistics=_statistics(a, b),
                comparison_issues=tuple(issues),
                reference_side=reference,
                predictive_checks=checks,
                predictive_unavailable_reason=reason,
            )
        )
    return DataDiffReport(
        left=tuple(item.source for item in sides[0]),
        right=tuple(item.source for item in sides[1]),
        variables=tuple(comparisons),
    )
