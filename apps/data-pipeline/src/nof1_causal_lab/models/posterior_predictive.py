"""Posterior Predictive Checks (PPCs) for fitted CT-SSM models.

Forward-simulates observations from posterior parameter draws and compares
them to the real data, producing per-variable diagnostics that flag
calibration, autocorrelation, and variance issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from nof1_causal_lab.artifacts.availability import Available, Evaluation, NotApplicable, Unavailable
from nof1_causal_lab.artifacts.checks import (
    Assessment,
    Evaluated,
    IndicatorCheckSubject,
    NotEvaluated,
    NumericCriterionEvidence,
)
from nof1_causal_lab.artifacts.identity import IndicatorId, IndicatorRef
from nof1_causal_lab.artifacts.posterior_diagnostics import (
    PosteriorPredictiveChecks,
    PPCOverlay,
    PPCTestStat,
)
from nof1_causal_lab.study.view_models import (
    Added,
    DataSeries,
    Dataset,
    DataStatistic,
    DataStatisticComparison,
    DataVariableDiff,
    PredictiveComparison,
    PredictiveComparisonResult,
    Removed,
    Revised,
)
from nof1_causal_lab.utils.histograms import histogram_draws
from nof1_causal_lab.utils.time_coordinates import ObservationInstant

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from datetime import datetime

    from nof1_causal_lab.artifacts.checks import IndicatorCheck


import jax.numpy as jnp

# ---------------------------------------------------------------------------
# PPC models
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------


def _indicator_columns(
    observations: jnp.ndarray,
    indicator_ids: Sequence[IndicatorId],
    *,
    check: IndicatorCheck,
) -> tuple[tuple[int, IndicatorCheckSubject, jnp.ndarray, jnp.ndarray], ...]:
    """Bind indicator identities and observed-position masks in column order."""

    def _column(
        j: int, name: IndicatorId
    ) -> tuple[int, IndicatorCheckSubject, jnp.ndarray, jnp.ndarray]:
        observed = observations[:, j]
        subject = IndicatorCheckSubject(check=check, target=IndicatorRef(id=name))
        return j, subject, observed, ~jnp.isnan(observed)

    return tuple(
        _column(j, name)
        for j, name in zip(range(observations.shape[1]), indicator_ids, strict=True)
    )


def _evaluated_indicator(
    subject: IndicatorCheckSubject,
    *,
    outcome: Literal["passed", "warning"],
    value: float,
    lower: float,
    upper: float,
    note: str,
) -> Evaluated[IndicatorCheckSubject, NumericCriterionEvidence]:
    """Construct the criterion evidence from its existing indicator subject."""
    return Evaluated(
        subject=subject,
        outcome=outcome,
        evidence=NumericCriterionEvidence(
            criterion=subject.check, value=value, lower=lower, upper=upper, note=note
        ),
    )


def _check_calibration(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[IndicatorId],
    low_threshold: float = 0.70,
    high_threshold: float = 0.98,
) -> tuple[Assessment[IndicatorCheckSubject, NumericCriterionEvidence], ...]:
    """Measure coverage of observations by pointwise 95% predictive intervals.

    Args:
        y_sim: Replicated observations with shape ``(draw, time, indicator)``.
        observations: Recorded values with shape ``(time, indicator)``; NaN marks
            missing observations.
        indicator_ids: Scientific identities in observation-column order.
        low_threshold: Smallest accepted fraction of observed values inside the
            predictive interval.
        high_threshold: Largest accepted fraction; higher coverage warns that the
            predictive distribution may be too diffuse.

    Returns:
        One assessment per indicator, including an explicit unevaluated result
        when fewer than two observations are available.
    """
    warnings: list[Assessment[IndicatorCheckSubject, NumericCriterionEvidence]] = []

    q025 = jnp.percentile(y_sim, 2.5, axis=0)  # (T, m)
    q975 = jnp.percentile(y_sim, 97.5, axis=0)  # (T, m)

    for j, subject, obs_j, valid in _indicator_columns(
        observations, indicator_ids, check="calibration"
    ):
        n_valid = jnp.sum(valid)
        if n_valid < 2:
            warnings.append(
                NotEvaluated(
                    subject=subject,
                    reason="INSUFFICIENT_OBSERVATIONS",
                    detail="Calibration requires at least two observations.",
                )
            )
            continue

        in_interval = valid & (obs_j >= q025[:, j]) & (obs_j <= q975[:, j])
        coverage = float(jnp.sum(in_interval) / n_valid)

        if coverage < low_threshold:
            outcome = "warning"
            note = f"Undercoverage: {coverage:.0%} of observations fall in 95% PPC interval (expected ~95%)"
        elif coverage > high_threshold:
            outcome = "warning"
            note = f"Overcoverage: {coverage:.0%} of observations fall in 95% PPC interval (model may be too diffuse)"
        else:
            outcome = "passed"
            note = f"95% CI coverage: {coverage:.1%} (expected ~95%)"
        warnings.append(
            _evaluated_indicator(
                subject,
                outcome=outcome,
                value=coverage,
                lower=low_threshold,
                upper=high_threshold,
                note=note,
            )
        )

    return tuple(warnings)


def _check_residual_autocorrelation(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[IndicatorId],
    threshold: float = 0.3,
) -> tuple[Assessment[IndicatorCheckSubject, NumericCriterionEvidence], ...]:
    """Assess lag-one correlation after subtracting the predictive mean.

    Args:
        y_sim: Replicated observations with shape ``(draw, time, indicator)``.
        observations: Recorded values with shape ``(time, indicator)``; NaN marks
            missing observations.
        indicator_ids: Scientific identities in observation-column order.
        threshold: Largest accepted absolute residual correlation.

    Returns:
        Per-indicator assessments using consecutive available residuals. Fewer
        than five observations or negligible residual variance yields an
        unevaluated assessment.
    """
    warnings: list[Assessment[IndicatorCheckSubject, NumericCriterionEvidence]] = []

    pp_mean = jnp.mean(y_sim, axis=0)  # (T, m)

    for j, subject, obs_j, valid in _indicator_columns(
        observations, indicator_ids, check="autocorrelation"
    ):
        # Build valid residuals
        residuals = jnp.where(valid, obs_j - pp_mean[:, j], 0.0)
        n_valid = int(jnp.sum(valid))
        if n_valid < 5:
            warnings.append(
                NotEvaluated(
                    subject=subject,
                    reason="INSUFFICIENT_OBSERVATIONS",
                    detail="Autocorrelation requires at least five observations.",
                )
            )
            continue

        # Compute lag-1 autocorrelation on valid residuals
        # Extract valid residuals using masking
        valid_idx = jnp.where(valid, size=n_valid)[0]
        valid_res = residuals[valid_idx]

        mean_r = jnp.mean(valid_res)
        centered = valid_res - mean_r
        var_r = jnp.mean(centered**2)

        if var_r < 1e-12:
            warnings.append(
                NotEvaluated(
                    subject=subject,
                    reason="ZERO_RESIDUAL_VARIANCE",
                    detail="Residual autocorrelation is undefined with zero residual variance.",
                )
            )
            continue

        autocov = jnp.mean(centered[:-1] * centered[1:])
        rho = float(autocov / var_r)

        passed = abs(rho) <= threshold
        warnings.append(
            _evaluated_indicator(
                subject,
                outcome="passed" if passed else "warning",
                value=rho,
                lower=-threshold,
                upper=threshold,
                note=f"Residual autocorrelation at lag 1: {rho:.2f}"
                + ("" if passed else f" (|rho| > {threshold})"),
            )
        )

    return tuple(warnings)


def _check_variance_ratio(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[IndicatorId],
    high_ratio: float = 3.0,
    low_ratio: float = 1.0 / 3.0,
) -> tuple[Assessment[IndicatorCheckSubject, NumericCriterionEvidence], ...]:
    """Compare replicated temporal standard deviations with the observed standard deviation.

    Args:
        y_sim: Replicated observations with shape ``(draw, time, indicator)``.
        observations: Recorded values with shape ``(time, indicator)``; NaN marks
            missing observations.
        indicator_ids: Scientific identities in observation-column order.
        high_ratio: Largest accepted predicted-to-observed standard-deviation ratio.
        low_ratio: Smallest accepted predicted-to-observed standard-deviation ratio.

    Returns:
        Per-indicator assessments of the mean replicated standard deviation divided
        by the observed standard deviation, using the same observed time points.
        Insufficient observations or negligible observed variance is unevaluated.
    """
    warnings: list[Assessment[IndicatorCheckSubject, NumericCriterionEvidence]] = []

    for j, subject, obs_j, valid in _indicator_columns(
        observations, indicator_ids, check="variance"
    ):
        n_valid = int(jnp.sum(valid))
        if n_valid < 3:
            warnings.append(
                NotEvaluated(
                    subject=subject,
                    reason="INSUFFICIENT_OBSERVATIONS",
                    detail="Variance comparison requires at least three observations.",
                )
            )
            continue

        valid_idx = jnp.where(valid, size=n_valid)[0]
        obs_std = float(jnp.std(obs_j[valid_idx]))
        if obs_std < 1e-12:
            warnings.append(
                NotEvaluated(
                    subject=subject,
                    reason="ZERO_OBSERVED_VARIANCE",
                    detail="Variance ratio is undefined with zero observed variance.",
                )
            )
            continue

        # Compare temporal variation on the same observed schedule. Latent-only
        # support boundaries have no emission and must not enter this reduction.
        per_draw_std = jnp.std(y_sim[:, valid_idx, j], axis=1)
        predicted_std = float(jnp.mean(per_draw_std))
        ratio = predicted_std / obs_std

        if ratio > high_ratio:
            outcome = "warning"
            note = f"PPC variance too high: simulated std / observed std = {ratio:.1f}"
        elif ratio < low_ratio:
            outcome = "warning"
            note = f"PPC variance too low: simulated std / observed std = {ratio:.1f}"
        else:
            outcome = "passed"
            note = f"Predicted variance {predicted_std:.3f} vs observed {obs_std:.3f} (ratio {ratio:.2f})"
        warnings.append(
            _evaluated_indicator(
                subject, outcome=outcome, value=ratio, lower=low_ratio, upper=high_ratio, note=note
            )
        )

    return tuple(warnings)


# ---------------------------------------------------------------------------
# Overlay and test statistic computations
# ---------------------------------------------------------------------------


def _predictive_value(value: jnp.ndarray) -> float | None:
    """A missing scheduled emission is an absent plot value, including in JSON."""
    return float(value) if jnp.isfinite(value) else None


def _compute_overlays(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[IndicatorId],
    *,
    times: tuple[float, ...],
    time_origin: datetime | None,
    standardized: tuple[bool, ...],
) -> list[PPCOverlay]:
    """Compose observed values, predictive medians, and retained trajectories for plotting.

    Args:
        y_sim: Replicated observations with shape ``(draw, time, indicator)``.
        observations: Recorded values with shape ``(time, indicator)``; NaN marks
            missing observations.
        indicator_ids: Scientific identities in observation-column order.
        times: Model-day coordinates aligned with the time axis.
        time_origin: Calendar instant of model day zero, or ``None`` for an undated
            model-time axis.
        standardized: Per-indicator flags describing the scale already used by
            both arrays; this function does not transform their values.

    Returns:
        One overlay per indicator, retaining every supplied predictive trajectory
        and representing missing plot values as ``None``.
    """
    overlays = []
    n_manifest = observations.shape[1]
    n_draws = y_sim.shape[0]

    q50 = jnp.percentile(y_sim, 50.0, axis=0)  # (T, m)

    for j, name in zip(range(n_manifest), indicator_ids, strict=True):
        obs_j = observations[:, j]
        observed = [None if jnp.isnan(v) else float(v) for v in obs_j]

        # Spaghetti: individual draw trajectories for this variable
        spaghetti = [
            [_predictive_value(v) for v in y_sim[int(idx), :, j]] for idx in range(n_draws)
        ]

        overlays.append(
            PPCOverlay(
                indicator_id=name,
                times=times,
                time_origin=time_origin,
                standardized=standardized[j],
                observed=tuple(observed),
                median=tuple(_predictive_value(v) for v in q50[:, j]),
                spaghetti_draws=tuple(tuple(draw) for draw in spaghetti),
            )
        )

    return overlays


def _compute_test_stats(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[IndicatorId],
) -> list[PPCTestStat]:
    """Compute test statistic distributions across y_rep draws.

    For each variable and each stat (mean, sd, min, max), computes the
    statistic across all y_rep draws and compares to the observed value.
    This is Gabry's ppc_stat plot data.

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        indicator_ids: scientific indicator IDs in observation-column order

    Returns:
        Observed and per-replicate mean, standard deviation, minimum, and maximum
        for indicators with at least three observations. Each replicate is reduced
        on exactly the observed schedule.
    """
    test_stats = []
    n_manifest = observations.shape[1]

    stat_fns: dict[Literal["mean", "sd", "min", "max"], Callable[..., jnp.ndarray]] = {
        "mean": jnp.nanmean,
        "sd": lambda x, **kw: jnp.nanstd(x, **kw),
        "min": jnp.nanmin,
        "max": jnp.nanmax,
    }

    for j, name in zip(range(n_manifest), indicator_ids, strict=True):
        obs_j = observations[:, j]
        valid = ~jnp.isnan(obs_j)
        n_valid = int(jnp.sum(valid))
        if n_valid < 3:
            continue

        valid_idx = jnp.where(valid, size=n_valid)[0]
        obs_valid = obs_j[valid_idx]

        for stat_name, stat_fn in stat_fns.items():
            obs_stat = float(stat_fn(obs_valid))

            # Compute stat for each y_rep draw (over time axis)
            rep_stats = []
            for i in range(y_sim.shape[0]):
                y_rep_j = y_sim[i, :, j]
                # Use same valid mask as observed
                y_rep_valid = y_rep_j[valid_idx]
                rep_stats.append(float(stat_fn(y_rep_valid)))

            test_stats.append(
                PPCTestStat(
                    indicator_id=name,
                    stat_name=stat_name,
                    observed_value=obs_stat,
                    rep_values=tuple(rep_stats),
                    p_value=sum(value >= obs_stat for value in rep_stats) / len(rep_stats),
                    histogram=tuple(histogram_draws(rep_stats, max_bins=12)),
                )
            )

    return test_stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def measure_predictive_checks(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[IndicatorId],
    *,
    times: tuple[float, ...],
    time_origin: datetime | None,
    standardized: tuple[bool, ...],
) -> PosteriorPredictiveChecks:
    """Measure an existing predictive batch without generating more trajectories."""
    if y_sim.ndim != 3 or observations.shape != y_sim.shape[1:]:
        raise ValueError("Predictive checks require aligned draw/time/indicator arrays")
    if len(indicator_ids) != observations.shape[1] or len(set(indicator_ids)) != len(indicator_ids):
        raise ValueError("Predictive checks require one distinct ID per observation column")
    comparable = jnp.where(jnp.isfinite(observations)[None, :, :], y_sim, 0.0)
    if not bool(jnp.isfinite(comparable).all()):
        raise ValueError("Predictive comparisons require finite draws at observed positions")
    warnings = (
        *_check_calibration(y_sim, observations, indicator_ids),
        *_check_residual_autocorrelation(y_sim, observations, indicator_ids),
        *_check_variance_ratio(y_sim, observations, indicator_ids),
    )

    overlays = _compute_overlays(
        y_sim,
        observations,
        indicator_ids,
        times=times,
        time_origin=time_origin,
        standardized=standardized,
    )
    test_stats = _compute_test_stats(y_sim, observations, indicator_ids)

    return PosteriorPredictiveChecks(
        per_variable_warnings=warnings,
        n_subsample=int(y_sim.shape[0]),
        overlays=tuple(overlays),
        test_stats=tuple(test_stats),
    )


def _statistics(
    left: tuple[DataSeries, ...], right: tuple[DataSeries, ...]
) -> tuple[DataStatisticComparison, ...]:
    def measure(series: DataSeries) -> dict[tuple[DataStatistic, str | None], float | None]:
        values = np.asarray([point.value for point in series.points if point.value is not None])
        statistics: dict[tuple[DataStatistic, str | None], float | None] = {
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
            statistics[("mean", None)] = float(np.mean(values)) if len(values) else None
            statistics[("sd", None)] = float(np.std(values)) if len(values) else None
            statistics[("min", None)] = float(np.min(values)) if len(values) else None
            statistics[("max", None)] = float(np.max(values)) if len(values) else None
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
    identity: IndicatorId, left: Sequence[DataSeries], right: Sequence[DataSeries]
) -> PredictiveComparisonResult:
    if len(left) == len(right) == 1:
        return NotApplicable(reason="A predictive comparison requires replicated histories.")
    if len(left) > 1 and len(right) > 1:
        return Unavailable(
            reason="Requires one reference history and multiple replicated histories"
        )
    side: Literal["left", "right"]
    if len(left) == 1:
        (reference,) = left
        side, replicas = "left", right
    else:
        (reference,) = right
        side, replicas = "right", left
    variable = reference.variable

    def _comparison(evaluation: Evaluation[PosteriorPredictiveChecks]) -> PredictiveComparison:
        return PredictiveComparison(reference_side=side, evaluation=evaluation)

    # Shared input problems belong to comparison_issues, not a second PPC reason.
    if any((item.time_origin is None) != (reference.time_origin is None) for item in replicas):
        return _comparison(NotApplicable(reason="Comparison inputs are incompatible."))
    if variable is None or any(item.variable is None for item in replicas):
        return _comparison(NotApplicable(reason="Comparison inputs are incompatible."))
    if any(
        item.variable.definition != variable.definition
        for item in replicas
        if item.variable is not None
    ):
        return _comparison(NotApplicable(reason="Comparison inputs are incompatible."))
    if variable.measurement_dtype in {"categorical", "ordinal"}:
        return _comparison(
            Unavailable(
                reason="Discrete codebooks use per-level proportions, not numeric PPC summaries"
            )
        )
    if not any(point.value is not None for point in reference.points):
        return _comparison(Unavailable(reason="The reference history contains no observed values"))
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
                    return _comparison(NotApplicable(reason="Comparison inputs are incompatible."))
                if candidate.value is None:
                    return _comparison(
                        Unavailable(reason="Replicas contain missing values at observed anchors")
                    )
            values.append(
                candidate.value if candidate is not None and candidate.value is not None else np.nan
            )
        aligned.append(values)
    import jax.numpy as jnp

    observed = [point.value if point.value is not None else np.nan for point in reference.points]
    checks = measure_predictive_checks(
        jnp.asarray(aligned)[:, :, None],
        jnp.asarray(observed)[:, None],
        (identity,),
        times=tuple(
            ObservationInstant(point.anchor_time)
            .relative_to(ObservationInstant.origin(reference.time_origin))
            .days
            for point in reference.points
        ),
        time_origin=reference.time_origin,
        standardized=(False,),
    )
    return _comparison(Available(value=checks))


def compare_data_variables(
    left: Dataset | Sequence[Dataset],
    right: Dataset | Sequence[Dataset],
    *,
    input_indicators: set[IndicatorId] | frozenset[IndicatorId] = frozenset(),
) -> tuple[DataVariableDiff, ...]:
    """Compare one or many saved histories on each side without pooling or resimulation."""
    sides = tuple(
        (value,) if isinstance(value, Dataset) else tuple(value) for value in (left, right)
    )
    for side in sides:
        if not side:
            raise ValueError("Each comparison side requires at least one dataset")
        if len(
            {
                (
                    dataset.source.revision,
                    dataset.source.replicate_index,
                )
                for dataset in side
            }
        ) != len(side):
            raise ValueError("A dataset cannot be counted twice within a comparison side")
    left_series, right_series = (tuple(dataset.series for dataset in side) for side in sides)
    variables = sorted(set().union(*(item.keys() for item in (*left_series, *right_series))))
    comparisons = []
    absent = DataSeries(variable=None, time_origin=None, points=())
    for identity in variables:
        if identity in input_indicators:
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
        if any(item.definition != definitions[0].definition for item in definitions[1:]):
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
                        Added(after=new[anchor])
                        if anchor not in old
                        else Removed(before=old[anchor])
                        if anchor not in new
                        else Revised(before=old[anchor], after=new[anchor])
                    )
        comparisons.append(
            DataVariableDiff(
                indicator_id=identity,
                left=a,
                right=b,
                changes=tuple(changes),
                statistics=_statistics(a, b),
                comparison_issues=tuple(issues),
                predictive=_predictive_comparison(identity, a, b),
            )
        )
    return tuple(comparisons)
