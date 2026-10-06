"""validation rule registry and reduction logic."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import polars as pl

from nof1_causal_lab.actions.validation.checks import (
    MIN_OBSERVATIONS,
    OBSERVATION_TIME_COLUMN,
    check_construct_correlations,
    check_dtype_range,
    check_hallucination_signals,
    check_time_coverage,
    check_timestamp_gaps,
    data_availability_issue,
    parse_timestamp_series,
    timestamp_issue_specs,
)
from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
    IndicatorAudit,
    IndicatorEmpiricalProfile,
    ValidationIssue,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from nof1_causal_lab.artifacts.construct import ConstructSpec
    from nof1_causal_lab.artifacts.identity import IndicatorId
    from nof1_causal_lab.artifacts.observations import (
        AuthoredObservationSpec,
        ResolvedObservationSpec,
    )

logger = logging.getLogger(__name__)


class HealthMetrics(TypedDict, total=False):
    """Optional measurements and per-check verdicts accumulated while auditing an indicator."""

    n_obs: int
    data_availability: bool | None
    variance: float | None
    n_unparseable_timestamps: int
    time_coverage_ratio: float | None
    max_gap_ratio: float | None
    dtype_violations: int | None
    duplicate_pct: float | None
    arithmetic_sequence_detected: bool
    cell_statuses: dict[str, Literal["ok", "warning", "error", "not_evaluated"]]


@dataclass(frozen=True)
class Issue:
    """Validation finding associated with an indicator and the check cell it should affect."""

    indicator: IndicatorId | None
    issue_type: str
    severity: Literal["error", "warning", "info"]
    message: str
    cell_key: str


@dataclass
class ValidationFindings:
    """Issues and measured quantities emitted by one validation rule."""

    issues: list[Issue] = field(default_factory=list)
    metrics: HealthMetrics = field(default_factory=HealthMetrics)


@dataclass
class IndicatorContext:
    """Parsed values, timestamps, and model context shared by an indicator's validation rules."""

    name: IndicatorId
    ind_data: pl.DataFrame
    values: pl.Series
    n_obs: int
    variance: float | None
    dtype: str | None
    is_time_invariant: bool
    model_clock_hours: float | None
    parsed_ts: pl.Series
    n_unparseable: int
    n_total_ts: int


@dataclass(frozen=True)
class IndicatorRuleInput:
    """One indicator's rows and optional parsed context; empty data has no context."""

    name: IndicatorId
    ind_data: pl.DataFrame
    ctx: IndicatorContext | None


@dataclass(frozen=True)
class ValidationContext:
    """Observation definitions and recorded rows shared across a validation pass."""

    combined: pl.DataFrame
    indicators: Sequence[AuthoredObservationSpec | ResolvedObservationSpec]
    indicator_ids: set[IndicatorId]
    indicator_lookup: Mapping[IndicatorId, AuthoredObservationSpec | ResolvedObservationSpec]
    construct_lookup: Mapping[str, ConstructSpec]
    model_clock_hours: float | None

    def iter_indicators(
        self,
    ) -> Iterator[tuple[IndicatorId, pl.DataFrame, IndicatorContext | None]]:
        """Yield each indicator's rows and parsed context in ID order, using ``None`` for empty rows."""
        for indicator_id in sorted(self.indicator_ids):
            ind_data = self.combined.filter(pl.col("indicator_id") == indicator_id)
            if ind_data.is_empty():
                yield indicator_id, ind_data, None
                continue
            yield (
                indicator_id,
                ind_data,
                _build_indicator_context(
                    indicator_id,
                    ind_data,
                    self.indicator_lookup,
                    self.construct_lookup,
                    self.model_clock_hours,
                ),
            )


@dataclass(frozen=True)
class ValidationRule[Input]:
    """Named validation operation producing issues and health measurements from its input."""

    name: str
    check: Callable[[Input], ValidationFindings]


def issue_payload(issue: Issue) -> ValidationIssue:
    """Project an internal finding into the persisted issue schema, omitting its UI check key."""
    return ValidationIssue(
        indicator_id=issue.indicator,
        issue_type=issue.issue_type,
        severity=issue.severity,
        message=issue.message,
    )


def issue_from_raw(raw_issue: ValidationIssue, *, cell_key: str) -> Issue:
    """Attach the check-cell identity used when reducing a persisted issue into verdicts."""
    return Issue(
        raw_issue.indicator_id,
        raw_issue.issue_type,
        raw_issue.severity,
        raw_issue.message,
        cell_key=cell_key,
    )


def issues_from_raw(
    raw_issues: list[ValidationIssue],
    *,
    cell_key: str | Callable[[ValidationIssue], str],
) -> list[Issue]:
    """Attach a fixed or issue-specific check-cell identity to each supplied issue."""
    issues: list[Issue] = []
    for raw_issue in raw_issues:
        resolved_cell_key = cell_key if isinstance(cell_key, str) else cell_key(raw_issue)
        issues.append(issue_from_raw(raw_issue, cell_key=resolved_cell_key))
    return issues


def no_data_validation_result() -> DataProfileArtifact:
    """Build a data profile containing the dataset-wide error for an empty extraction."""
    from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact

    return DataProfileArtifact(
        indicators={},
        dataset_issues=(
            ValidationIssue(
                indicator_id=None,
                issue_type="no_data",
                severity="error",
                message="No data extracted",
            ),
        ),
    )


def _rule_missing(entry: IndicatorRuleInput) -> ValidationFindings:
    issue = data_availability_issue(entry.name, 0 if entry.ctx is None else entry.ctx.n_obs)
    return ValidationFindings(
        issues=[issue_from_raw(issue, cell_key="")] if issue else [],
        metrics={"data_availability": None if issue else True},
    )


def _rule_no_numeric(entry: IndicatorRuleInput) -> ValidationFindings:
    if entry.ind_data.is_empty() or entry.ctx is not None:
        return ValidationFindings()
    return ValidationFindings(
        issues=[
            Issue(
                entry.name,
                "no_numeric",
                "error",
                "No numeric values extracted",
                cell_key="",
            )
        ]
    )


def _rule_timestamps(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    issues = [
        Issue(
            ctx.name,
            "unparseable_timestamps",
            severity,
            message,
            cell_key="n_unparseable_timestamps",
        )
        for severity, message in timestamp_issue_specs(ctx.n_total_ts, ctx.n_unparseable)
    ]
    return ValidationFindings(
        issues=issues,
        metrics={"n_unparseable_timestamps": ctx.n_unparseable},
    )


def _rule_sample_size(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    issues = []
    if ctx.n_obs < MIN_OBSERVATIONS:
        issues.append(
            Issue(
                ctx.name,
                "low_n",
                "warning",
                f"Only {ctx.n_obs} observations (recommend >= {MIN_OBSERVATIONS})",
                cell_key="n_obs",
            )
        )
    return ValidationFindings(issues=issues, metrics={"n_obs": ctx.n_obs})


def _rule_variance(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    if ctx.is_time_invariant:
        return ValidationFindings(metrics={"variance": ctx.variance})
    issues = []
    if ctx.variance is not None and ctx.variance == 0:
        issues.append(
            Issue(
                ctx.name,
                "no_variance",
                "warning",
                f"Zero variance (constant value = {ctx.values.first()})",
                cell_key="variance",
            )
        )
    return ValidationFindings(issues=issues, metrics={"variance": ctx.variance})


def _rule_dtype_range(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    if not ctx.dtype:
        return ValidationFindings(metrics={"dtype_violations": 0})
    raw_issues, violation_count = check_dtype_range(ctx.values, ctx.dtype, ctx.name)
    return ValidationFindings(
        issues=issues_from_raw(raw_issues, cell_key="dtype_violations"),
        metrics={"dtype_violations": violation_count},
    )


def _clock_metric_findings(
    entry: IndicatorRuleInput,
    *,
    check: Callable[[pl.Series, float, IndicatorId], tuple[list[ValidationIssue], float | None]],
    metric_key: Literal["time_coverage_ratio", "max_gap_ratio"],
) -> ValidationFindings:
    """Own clock-check eligibility and attach issues to the measured metric."""
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    if ctx.is_time_invariant or ctx.model_clock_hours is None:
        return ValidationFindings(metrics=cast("HealthMetrics", {metric_key: None}))
    raw_issues, ratio = check(ctx.parsed_ts, ctx.model_clock_hours, ctx.name)
    return ValidationFindings(
        issues=issues_from_raw(raw_issues, cell_key=metric_key),
        metrics=cast("HealthMetrics", {metric_key: ratio}),
    )


def _rule_time_coverage(entry: IndicatorRuleInput) -> ValidationFindings:
    return _clock_metric_findings(
        entry, check=check_time_coverage, metric_key="time_coverage_ratio"
    )


def _rule_timestamp_gaps(entry: IndicatorRuleInput) -> ValidationFindings:
    return _clock_metric_findings(entry, check=check_timestamp_gaps, metric_key="max_gap_ratio")


def _rule_hallucination_signals(entry: IndicatorRuleInput) -> ValidationFindings:
    ctx = entry.ctx
    if ctx is None:
        return ValidationFindings()
    raw_issues, duplicate_pct, arith_seq = check_hallucination_signals(
        ctx.values, ctx.dtype or "continuous", ctx.name
    )
    return ValidationFindings(
        issues=issues_from_raw(
            raw_issues,
            cell_key=lambda raw_issue: (
                "arithmetic_sequence_detected"
                if "arithmetic sequence" in raw_issue.message
                else "duplicate_pct"
            ),
        ),
        metrics={"duplicate_pct": duplicate_pct, "arithmetic_sequence_detected": arith_seq},
    )


def _rule_construct_correlations(ctx: ValidationContext) -> ValidationFindings:
    raw_issues = check_construct_correlations(ctx.combined, ctx.construct_lookup)
    return ValidationFindings(issues=issues_from_raw(raw_issues, cell_key=""))


DATA_RULES: list[ValidationRule[IndicatorRuleInput]] = [
    ValidationRule("missing", _rule_missing),
    ValidationRule("no_numeric", _rule_no_numeric),
    ValidationRule("timestamps", _rule_timestamps),
    ValidationRule("sample_size", _rule_sample_size),
    ValidationRule("variance", _rule_variance),
    ValidationRule("dtype_range", _rule_dtype_range),
    ValidationRule("time_coverage", _rule_time_coverage),
    ValidationRule("timestamp_gaps", _rule_timestamp_gaps),
    ValidationRule("hallucination_signals", _rule_hallucination_signals),
]

COMPATIBILITY_RULES: list[ValidationRule[ValidationContext]] = [
    ValidationRule("construct_correlations", _rule_construct_correlations),
]


CELL_STATUS_KEYS = frozenset(
    {
        "n_obs",
        "data_availability",
        "variance",
        "n_unparseable_timestamps",
        "time_coverage_ratio",
        "max_gap_ratio",
        "dtype_violations",
        "duplicate_pct",
        "arithmetic_sequence_detected",
    }
)


def reduce_findings(
    indicator_findings: dict[str, list[ValidationFindings]],
) -> tuple[list[ValidationIssue], dict[str, HealthMetrics]]:
    """Merge rule findings into issues and per-indicator verdicts, preserving error precedence."""
    all_issues: list[ValidationIssue] = []
    indicator_health: dict[str, HealthMetrics] = {}

    for indicator_id in sorted(indicator_findings):
        findings_list = indicator_findings[indicator_id]
        merged_metrics: HealthMetrics = {}
        ind_issues: list[Issue] = []
        for findings in findings_list:
            ind_issues.extend(findings.issues)
            merged_metrics.update(findings.metrics)

        for issue in ind_issues:
            all_issues.append(issue_payload(issue))

        cell_statuses: dict[str, Literal["ok", "warning", "error", "not_evaluated"]] = {
            key: "not_evaluated" if value is None else "ok"
            for key, value in merged_metrics.items()
            if key in CELL_STATUS_KEYS
        }
        for issue in ind_issues:
            if (
                issue.cell_key in cell_statuses
                and cell_statuses[issue.cell_key] != "error"
                and (issue.severity == "error" or issue.severity == "warning")
            ):
                cell_statuses[issue.cell_key] = issue.severity

        indicator_health[indicator_id] = {**merged_metrics, "cell_statuses": cell_statuses}

    return all_issues, indicator_health


def _build_indicator_context(
    indicator_id: IndicatorId,
    ind_data: pl.DataFrame,
    indicator_lookup: Mapping[IndicatorId, AuthoredObservationSpec | ResolvedObservationSpec],
    construct_lookup: Mapping[str, ConstructSpec],
    model_clock_hours: float | None,
) -> IndicatorContext | None:
    values_df = ind_data.select(pl.col("value").cast(pl.Float64, strict=False)).drop_nulls()
    n_obs = len(values_df)
    if n_obs == 0:
        return None

    values = values_df["value"]
    variance = cast("float | None", values.var())

    indicator_meta = indicator_lookup.get(indicator_id)
    dtype = indicator_meta.measurement_dtype if indicator_meta is not None else None
    if indicator_meta is not None and (window := indicator_meta.observation_window) is not None:
        model_clock_hours = window.seconds / 3600
    construct_meta = construct_lookup.get(indicator_id)
    is_time_invariant = (
        construct_meta is not None and construct_meta.temporal_status == "time_invariant"
    )

    timestamps = ind_data[OBSERVATION_TIME_COLUMN]
    parsed = parse_timestamp_series(timestamps)

    return IndicatorContext(
        name=indicator_id,
        ind_data=ind_data,
        values=values,
        n_obs=n_obs,
        variance=variance,
        dtype=dtype,
        is_time_invariant=is_time_invariant,
        model_clock_hours=model_clock_hours,
        parsed_ts=parsed.drop_nulls(),
        n_unparseable=parsed.null_count(),
        n_total_ts=len(timestamps),
    )


def _float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return None if math.isnan(value) else value


def _compute_empirical_profile(
    indicator_id: IndicatorId,
    model_data: pl.DataFrame,
) -> IndicatorEmpiricalProfile | None:
    ind_model = model_data.filter(pl.col("indicator_id") == indicator_id)
    values_df = ind_model.select(pl.col("value").cast(pl.Float64, strict=False)).drop_nulls()
    n_obs = len(values_df)
    if n_obs == 0:
        return None

    values = values_df["value"]
    return IndicatorEmpiricalProfile(
        n_obs=n_obs,
        min=_float_or_none(cast("float | None", values.min())),
        q25=_float_or_none(values.quantile(0.25)),
        q50=_float_or_none(values.quantile(0.50)),
        q75=_float_or_none(values.quantile(0.75)),
        max=_float_or_none(cast("float | None", values.max())),
        mean=_float_or_none(cast("float | None", values.mean())),
    )


def build_indicator_audits(
    *,
    indicator_ids: set[IndicatorId],
    model_data: pl.DataFrame,
    indicator_issues: list[ValidationIssue],
    indicator_health: dict[str, HealthMetrics],
) -> dict[IndicatorId, IndicatorAudit]:
    """Combine each selected indicator's empirical profile, issues, and check verdicts."""
    issues_by_indicator: dict[str, list[ValidationIssue]] = {name: [] for name in indicator_ids}
    for issue in indicator_issues:
        issue_indicator = issue.indicator_id
        if issue_indicator is not None and issue_indicator in issues_by_indicator:
            issues_by_indicator[issue_indicator].append(issue)

    audits: dict[IndicatorId, IndicatorAudit] = {}
    for indicator_id in sorted(indicator_ids):
        health_metrics = indicator_health.get(indicator_id, {})
        audits[indicator_id] = IndicatorAudit(
            profile=_compute_empirical_profile(
                indicator_id,
                model_data,
            ),
            issues=tuple(issues_by_indicator.get(indicator_id, [])),
            checks=dict(health_metrics.get("cell_statuses", {})),
        )
    return audits


def run_rules(
    ctx: ValidationContext,
    *,
    indicator_rules: Sequence[ValidationRule[IndicatorRuleInput]] = (),
    dataset_rules: Sequence[ValidationRule[ValidationContext]] = (),
) -> tuple[list[ValidationIssue], dict[str, HealthMetrics], list[ValidationIssue]]:
    """Evaluate indicator and dataset rules without mixing their findings.

    Args:
        ctx: Recorded observations and definitions supplying the validation pass.
        indicator_rules: Rules applied independently to every selected indicator.
        dataset_rules: Rules applied once to the complete dataset context.

    Returns:
        Indicator issues, per-indicator health metrics, and dataset-wide issues,
        in that order.
    """
    indicator_findings: dict[str, list[ValidationFindings]] = {}
    for indicator_id, ind_data, indicator_ctx in ctx.iter_indicators():
        rule_input = IndicatorRuleInput(indicator_id, ind_data, indicator_ctx)
        indicator_findings[indicator_id] = [rule.check(rule_input) for rule in indicator_rules]

    dataset_issues: list[ValidationIssue] = []
    for rule in dataset_rules:
        findings = rule.check(ctx)
        dataset_issues.extend(issue_payload(issue) for issue in findings.issues)

    indicator_issues, indicator_health = reduce_findings(indicator_findings)
    return indicator_issues, indicator_health, dataset_issues
