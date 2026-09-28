"""Read-only comparison of saved observation datasets and simulation replicates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from typing import TYPE_CHECKING, Annotated, Literal, Self

import numpy as np
import polars as pl
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

from nof1_causal_lab.artifacts.duration import parse_duration_to_hours
from nof1_causal_lab.artifacts.effects import HistogramBin  # noqa: TC001
from nof1_causal_lab.artifacts.identity import GitOid, IndicatorId  # noqa: TC001
from nof1_causal_lab.artifacts.observations import ObservationSpec  # noqa: TC001
from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorPredictiveChecks  # noqa: TC001
from nof1_causal_lab.utils.histograms import histogram_draws

if TYPE_CHECKING:
    from collections.abc import Sequence


class DataRef(BaseModel):
    """A saved panel or simulation; omitting replicate selects every saved simulation draw."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["panel", "simulation"]
    revision: GitOid
    replicate: int | None = Field(default=None, ge=0)
    time_origin: AwareDatetime | None = Field(
        default=None,
        description="Calendar instant for simulation day zero; omitted uses 1970-01-01 UTC.",
    )

    @model_validator(mode="after")
    def source_coordinates(self) -> Self:
        if self.kind == "panel" and (self.replicate is not None or self.time_origin is not None):
            raise ValueError("Panels already define their history and calendar timestamps")
        return self


type DataSelection = Annotated[
    DataRef | Annotated[tuple[DataRef, ...], Field(min_length=1)],
    Field(description="A data selection identifies one or more saved observation histories."),
]


class DataDiffRequest(BaseModel):
    """Compare two immutable data selections, each containing one or more histories."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    left: DataSelection
    right: DataSelection


class DataPoint(BaseModel):
    """An observed value at an exact calendar anchor and measurement support."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    anchor_time: AwareDatetime
    support_start: AwareDatetime | None
    support_end: AwareDatetime | None
    value: FiniteFloat | None


class DataSeries(BaseModel):
    """One variable's recorded measurements in one history; no pooling across replicas."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    variable: ObservationSpec | None
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


def read_data_diff(workspace_id: str, request: DataDiffRequest) -> DataDiffReport:
    """Load existing data only; never read a model for generation, fit, or write artifacts."""
    from nof1_causal_lab.actions.data_checks import read_data_metadata
    from nof1_causal_lab.actions.prepare_data import prepare_simulation_panel
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.machine.history import StudyRepository
    from nof1_causal_lab.machine.store import ArtifactStore

    store = ArtifactStore(workspace_id)
    read_array = cache(store.read_array)

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
                ),
            )
        record = StudyRepository(workspace_id).record(source.revision)
        if record.status != "applied" or record.operation_id != "simulate":
            raise ValueError("Simulation data must select an applied simulation commit")
        report = SimulationReport.model_validate(record.diagnostics["report"])
        if report.model.workspace_id != workspace_id:
            raise ValueError("The simulation must belong to the selected study")
        indices = range(report.draws) if source.replicate is None else (source.replicate,)
        result = []
        for replicate in indices:
            panel = prepare_simulation_panel(report, replicate, read_array=read_array)
            if source.time_origin is not None:
                offset = source.time_origin.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
                panel = panel.with_columns(
                    pl.col(name) + offset
                    for name in (
                        "anchor_time",
                        "support_start",
                        "support_end",
                    )
                )
            result.append(
                Dataset(
                    source.model_copy(update={"replicate": replicate}),
                    report.observation_layout.variables,
                    panel,
                )
            )
        return tuple(result)

    def selection(value: DataSelection) -> tuple[Dataset, ...]:
        refs = (value,) if isinstance(value, DataRef) else value
        return tuple(dataset for source in refs for dataset in load(source))

    return data_diff(selection(request.left), selection(request.right))


def _series(dataset: Dataset) -> dict[IndicatorId, DataSeries]:
    variables = {item.id: item for item in dataset.variables}
    if len(variables) != len(dataset.variables):
        raise ValueError("Dataset variables must have unique identities")
    frame = dataset.observations
    unknown = set(frame["indicator_id"]) - variables.keys()
    if unknown:
        raise ValueError(f"Observations have no variable definition: {sorted(unknown)}")
    for name in ("anchor_time", "support_start", "support_end"):
        expression = pl.col(name)
        if frame.schema[name] == pl.String:
            expression = expression.str.to_datetime(time_zone="UTC")
        else:
            expression = expression.cast(pl.Datetime("us", time_zone="UTC"))
        frame = frame.with_columns(expression)
    frame = frame.with_columns(pl.col("value").cast(pl.Float64).fill_nan(None))
    result = {}
    for identity, variable in variables.items():
        selected = frame.filter(pl.col("indicator_id") == identity).sort("anchor_time")
        for field, expected in (
            ("support_kind", variable.support_kind.value),
            ("summary_operator", variable.summary_operator.value),
            ("anchor_policy", variable.anchor_policy.value),
        ):
            if any(value != expected for value in selected[field]):
                raise ValueError(f"Observation {identity} has inconsistent {field}")
        points = tuple(
            DataPoint.model_validate(row)
            for row in selected.select(
                "anchor_time",
                "support_start",
                "support_end",
                "value",
            ).iter_rows(named=True)
        )
        if len({point.anchor_time for point in points}) != len(points):
            raise ValueError(f"Observation {identity} has duplicate anchors within a history")
        if any(
            point.value is not None
            and (
                point.support_start is None
                or point.support_end is None
                or point.support_start > point.support_end
                or (
                    point.support_start
                    if variable.anchor_policy == "support_start"
                    else point.support_end
                )
                != point.anchor_time
            )
            for point in points
        ):
            raise ValueError(f"Observation {identity} has invalid measurement support")
        levels = variable.categorical_levels or variable.ordinal_levels
        if levels is not None and any(
            point.value is not None
            and (not point.value.is_integer() or not 0 <= point.value < len(levels))
            for point in points
        ):
            raise ValueError(f"Observation {identity} has values outside its codebook")
        result[identity] = DataSeries(variable=variable, points=points)
    return result


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
    if (len(left) == 1) == (len(right) == 1):
        return None, None, "Requires one reference history and multiple replicated histories"
    side, reference, replicas = (
        ("left", left[0], right) if len(left) == 1 else ("right", right[0], left)
    )
    variable = reference.variable
    if variable is None or any(item.variable is None for item in replicas):
        return side, None, "Variable is absent from one or more histories"
    if any(
        _semantics(item.variable) != _semantics(variable)
        for item in replicas
        if item.variable is not None
    ):
        return side, None, "Measurement definitions differ"
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
            if point.value is not None and (
                candidate is None
                or candidate.value is None
                or (candidate.support_start, candidate.support_end)
                != (point.support_start, point.support_end)
            ):
                return (
                    side,
                    None,
                    "Replicas do not cover the reference's observed anchors and measurement windows",
                )
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
    left: Dataset | Sequence[Dataset], right: Dataset | Sequence[Dataset]
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
    comparisons = []
    absent = DataSeries(variable=None, points=())
    for identity in variables:
        a, b = (
            tuple(item.get(identity, absent) for item in side)
            for side in (left_series, right_series)
        )
        issues = []
        series = (*a, *b)
        definitions = [item.variable for item in series if item.variable is not None]
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
        if len(a) == len(b) == 1:
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
