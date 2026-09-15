"""Posterior Predictive Checks (PPCs) for fitted CT-SSM models.

Forward-simulates observations from posterior parameter draws and compares
them to the real data, producing per-variable diagnostics that flag
calibration, autocorrelation, and variance issues.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.artifacts.posterior_diagnostics import (
    PosteriorPredictiveChecks,
    PPCOverlay,
    PPCTestStat,
    PPCWarning,
)
from nof1_causal_lab.utils.histograms import histogram_draws

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


import jax.numpy as jnp

# ---------------------------------------------------------------------------
# PPC models
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------


def _check_calibration(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[str],
    low_threshold: float = 0.70,
    high_threshold: float = 0.98,
) -> list[PPCWarning]:
    """Check calibration: % of timepoints where obs falls in [2.5th, 97.5th].

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        indicator_ids: scientific indicator IDs in observation-column order
    """
    warnings = []
    n_manifest = observations.shape[1]

    q025 = jnp.percentile(y_sim, 2.5, axis=0)  # (T, m)
    q975 = jnp.percentile(y_sim, 97.5, axis=0)  # (T, m)

    for j, name in zip(range(n_manifest), indicator_ids, strict=True):
        obs_j = observations[:, j]
        valid = ~jnp.isnan(obs_j)
        n_valid = jnp.sum(valid)
        if n_valid < 2:
            continue

        in_interval = valid & (obs_j >= q025[:, j]) & (obs_j <= q975[:, j])
        coverage = float(jnp.sum(in_interval) / n_valid)

        if coverage < low_threshold:
            warnings.append(
                PPCWarning(
                    indicator_id=name,
                    check_type="calibration",
                    message=f"Undercoverage: {coverage:.0%} of observations fall in 95% PPC interval (expected ~95%)",
                    value=coverage,
                    passed=False,
                )
            )
        elif coverage > high_threshold:
            warnings.append(
                PPCWarning(
                    indicator_id=name,
                    check_type="calibration",
                    message=f"Overcoverage: {coverage:.0%} of observations fall in 95% PPC interval (model may be too diffuse)",
                    value=coverage,
                    passed=False,
                )
            )
        else:
            warnings.append(
                PPCWarning(
                    indicator_id=name,
                    check_type="calibration",
                    message=f"95% CI coverage: {coverage:.1%} (expected ~95%)",
                    value=coverage,
                    passed=True,
                )
            )

    return warnings


def _check_residual_autocorrelation(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[str],
    threshold: float = 0.3,
) -> list[PPCWarning]:
    """Check lag-1 autocorrelation of residuals (obs - posterior predictive mean).

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        indicator_ids: scientific indicator IDs in observation-column order
    """
    warnings = []
    n_manifest = observations.shape[1]

    pp_mean = jnp.mean(y_sim, axis=0)  # (T, m)

    for j, name in zip(range(n_manifest), indicator_ids, strict=True):
        obs_j = observations[:, j]
        valid = ~jnp.isnan(obs_j)

        # Build valid residuals
        residuals = jnp.where(valid, obs_j - pp_mean[:, j], 0.0)
        n_valid = int(jnp.sum(valid))
        if n_valid < 5:
            continue

        # Compute lag-1 autocorrelation on valid residuals
        # Extract valid residuals using masking
        valid_idx = jnp.where(valid, size=n_valid)[0]
        valid_res = residuals[valid_idx]

        mean_r = jnp.mean(valid_res)
        centered = valid_res - mean_r
        var_r = jnp.mean(centered**2)

        if var_r < 1e-12:
            continue

        autocov = jnp.mean(centered[:-1] * centered[1:])
        rho = float(autocov / var_r)

        passed = abs(rho) <= threshold
        warnings.append(
            PPCWarning(
                indicator_id=name,
                check_type="autocorrelation",
                message=f"Residual autocorrelation at lag 1: {rho:.2f}"
                + ("" if passed else f" (|rho| > {threshold})"),
                value=rho,
                passed=passed,
            )
        )

    return warnings


def _check_variance_ratio(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[str],
    high_ratio: float = 3.0,
    low_ratio: float = 1.0 / 3.0,
) -> list[PPCWarning]:
    """Check posterior predictive std / observed std ratio.

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        indicator_ids: scientific indicator IDs in observation-column order
    """
    warnings = []
    n_manifest = observations.shape[1]

    for j, name in zip(range(n_manifest), indicator_ids, strict=True):
        obs_j = observations[:, j]
        valid = ~jnp.isnan(obs_j)
        n_valid = int(jnp.sum(valid))
        if n_valid < 3:
            continue

        valid_idx = jnp.where(valid, size=n_valid)[0]
        obs_std = float(jnp.std(obs_j[valid_idx]))
        if obs_std < 1e-12:
            continue

        # Compare temporal variation on the same observed schedule. Latent-only
        # support boundaries have no emission and must not enter this reduction.
        per_draw_std = jnp.std(y_sim[:, valid_idx, j], axis=1)
        predicted_std = float(jnp.mean(per_draw_std))
        ratio = predicted_std / obs_std

        if ratio > high_ratio:
            warnings.append(
                PPCWarning(
                    indicator_id=name,
                    check_type="variance",
                    message=f"PPC variance too high: simulated std / observed std = {ratio:.1f}",
                    value=ratio,
                    passed=False,
                )
            )
        elif ratio < low_ratio:
            warnings.append(
                PPCWarning(
                    indicator_id=name,
                    check_type="variance",
                    message=f"PPC variance too low: simulated std / observed std = {ratio:.1f}",
                    value=ratio,
                    passed=False,
                )
            )
        else:
            warnings.append(
                PPCWarning(
                    indicator_id=name,
                    check_type="variance",
                    message=f"Predicted variance {predicted_std:.3f} vs observed {obs_std:.3f} (ratio {ratio:.2f})",
                    value=ratio,
                    passed=True,
                )
            )

    return warnings


# ---------------------------------------------------------------------------
# Overlay and test statistic computations
# ---------------------------------------------------------------------------


def _predictive_value(value: jnp.ndarray) -> float | None:
    """A missing scheduled emission is an absent plot value, including in JSON."""
    return float(value) if jnp.isfinite(value) else None


def _compute_overlays(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[str],
    n_spaghetti: int = 20,
) -> list[PPCOverlay]:
    """Compute per-variable quantile bands and spaghetti draws for PPC plots.

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        indicator_ids: scientific indicator IDs in observation-column order
        n_spaghetti: number of individual y_rep draws to include for spaghetti plots
    """
    overlays = []
    n_manifest = observations.shape[1]
    n_draws = y_sim.shape[0]

    q025 = jnp.percentile(y_sim, 2.5, axis=0)  # (T, m)
    q25 = jnp.percentile(y_sim, 25.0, axis=0)
    q50 = jnp.percentile(y_sim, 50.0, axis=0)
    q75 = jnp.percentile(y_sim, 75.0, axis=0)
    q975 = jnp.percentile(y_sim, 97.5, axis=0)

    # Select evenly-spaced spaghetti draws
    n_spag = min(n_spaghetti, n_draws)
    spag_indices = jnp.linspace(0, n_draws - 1, n_spag).astype(int)

    for j, name in zip(range(n_manifest), indicator_ids, strict=True):
        obs_j = observations[:, j]
        observed = [None if jnp.isnan(v) else float(v) for v in obs_j]

        # Spaghetti: individual draw trajectories for this variable
        spaghetti = [[_predictive_value(v) for v in y_sim[int(idx), :, j]] for idx in spag_indices]

        overlays.append(
            PPCOverlay(
                indicator_id=name,
                observed=observed,
                q025=[_predictive_value(v) for v in q025[:, j]],
                q25=[_predictive_value(v) for v in q25[:, j]],
                median=[_predictive_value(v) for v in q50[:, j]],
                q75=[_predictive_value(v) for v in q75[:, j]],
                q975=[_predictive_value(v) for v in q975[:, j]],
                spaghetti_draws=spaghetti,
            )
        )

    return overlays


def _compute_test_stats(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[str],
) -> list[PPCTestStat]:
    """Compute test statistic distributions across y_rep draws.

    For each variable and each stat (mean, sd, min, max), computes the
    statistic across all y_rep draws and compares to the observed value.
    This is Gabry's ppc_stat plot data.

    Args:
        y_sim: (n_subsample, T, n_manifest)
        observations: (T, n_manifest)
        indicator_ids: scientific indicator IDs in observation-column order
    """
    test_stats = []
    n_manifest = observations.shape[1]

    _StatName = Literal["mean", "sd", "min", "max"]
    stat_fns: dict[_StatName, Callable[..., jnp.ndarray]] = {
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
                    rep_values=rep_stats,
                    p_value=sum(value >= obs_stat for value in rep_stats) / len(rep_stats),
                    histogram=histogram_draws(rep_stats, max_bins=12),
                )
            )

    return test_stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def measure_predictive_checks(
    y_sim: jnp.ndarray,
    observations: jnp.ndarray,
    indicator_ids: Sequence[str],
) -> PosteriorPredictiveChecks:
    """Measure an existing predictive batch without generating more trajectories."""
    if y_sim.ndim != 3 or observations.shape != y_sim.shape[1:]:
        raise ValueError("Predictive checks require aligned draw/time/indicator arrays")
    if len(indicator_ids) != observations.shape[1] or len(set(indicator_ids)) != len(indicator_ids):
        raise ValueError("Predictive checks require one distinct ID per observation column")
    comparable = jnp.where(jnp.isfinite(observations)[None, :, :], y_sim, 0.0)
    if not bool(jnp.isfinite(comparable).all()):
        raise ValueError("Predictive comparisons require finite draws at observed positions")
    warnings: list[PPCWarning] = []
    warnings.extend(_check_calibration(y_sim, observations, indicator_ids))
    warnings.extend(_check_residual_autocorrelation(y_sim, observations, indicator_ids))
    warnings.extend(_check_variance_ratio(y_sim, observations, indicator_ids))

    overlays = _compute_overlays(y_sim, observations, indicator_ids)
    test_stats = _compute_test_stats(y_sim, observations, indicator_ids)

    return PosteriorPredictiveChecks(
        per_variable_warnings=warnings,
        checked=True,
        n_subsample=int(y_sim.shape[0]),
        overlays=overlays,
        test_stats=test_stats,
    )
