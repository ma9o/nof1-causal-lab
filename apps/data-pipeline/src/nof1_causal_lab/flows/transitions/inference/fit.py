"""posterior: Bayesian inference and diagnostics."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Unpack

import jax.numpy as jnp

from nof1_causal_lab.flows.model_spec_compile_cache import restore_model_spec_compile_cache
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.runtime import (
    PreparedModelRuntime,
    prepare_model_runtime,
)

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior
    from nof1_causal_lab.sampler_config import (
        MarginalParticleGibbsOptions,
        SamplerConfigInput,
    )

logger = logging.getLogger(__name__)


def fit_prepared_model(
    runtime: PreparedModelRuntime,
    **kwargs: Unpack[MarginalParticleGibbsOptions],
) -> ParticleMCMCPosterior:
    """Apply application sampler configuration to a prepared executable model."""
    from nof1_causal_lab.models.ssm.inference import fit

    sampler_config = {**runtime.sampler_config, **kwargs}
    method = sampler_config.get("method", "marginal_particle_gibbs")
    fit_kwargs = {key: value for key, value in sampler_config.items() if key != "method"}
    return fit(
        runtime.model,
        observations=runtime.observations,
        times=runtime.times,
        method=method,
        **fit_kwargs,
    )


def _fit_elapsed_seconds(start: float) -> float:
    return time.monotonic() - start


def _format_name_preview(names: list[str], limit: int = 4) -> str:
    if not names:
        return "none"
    preview = ", ".join(names[:limit])
    if len(names) > limit:
        preview = f"{preview}, ..."
    return preview


def _observed_cell_counts(observations: jnp.ndarray) -> tuple[int, int]:
    total_cells = int(observations.size)
    observed_cells = int(jnp.sum(~jnp.isnan(observations)).item())
    return observed_cells, total_cells


def _time_span_days(times: jnp.ndarray) -> float:
    if times.size <= 1:
        return 0.0
    return float((times[-1] - times[0]).item())


def _support_summary(runtime: PreparedModelRuntime) -> str:
    support = runtime.observation_support
    if support is None:
        return "none"
    if not support.requires_interval_summary_handling:
        return "point-only"
    return (
        f"interval({len(support.interval_summary_manifest_names)}: "
        f"{_format_name_preview(support.interval_summary_manifest_names)}) "
        f"max_active_windows={support.max_active_windows}"
    )


def fit_model(
    model_spec: ModelSpec,
    data_for_model: pl.DataFrame,
    sampler_config: SamplerConfigInput | None = None,
    model: Any = None,
    workspace_id: str | None = None,
    wait_for_compile_cache: bool = False,
    compute_loo_diagnostics: bool = True,
) -> Any:
    """Fit the SSM model to data.

    Args:
        model_spec: Complete scientific definition pinned to this fit
        data_for_model: Canonical observation rows (indicator, value, anchor_time, support metadata)
        sampler_config: Override sampler configuration (None uses config defaults)
        model: Optional pre-built SSMModel

    Returns:
        Fitted model results

    NOTE: Uses NumPyro SSM implementation.
    """
    logger.info(
        "Fitting model: rows=%d indicators=%d sampler=%s model_provided=%s",
        len(data_for_model),
        data_for_model["indicator_id"].n_unique()
        if "indicator_id" in data_for_model.columns
        else 0,
        (sampler_config or {}).get("method", "config default"),
        model is not None,
    )
    t0 = time.monotonic()

    cache_restored = restore_model_spec_compile_cache(
        workspace_id,
        model_spec,
        wait_for_pending=wait_for_compile_cache,
    )
    logger.info(
        "Compile cache restore: restored=%s wait_for_pending=%s workspace_id=%s",
        cache_restored,
        wait_for_compile_cache,
        workspace_id or "none",
    )

    try:
        prep_t0 = time.monotonic()
        runtime = prepare_model_runtime(
            data_for_model=data_for_model,
            model_spec=model_spec,
            sampler_config=sampler_config,
            model=model,
        )
        observed_cells, total_cells = _observed_cell_counts(runtime.observations)
        logger.info(
            "Prepared runtime in %.1fs: wide_rows=%d timepoints=%d manifest_vars=%d "
            "observed_cells=%d/%d time_span_days=%.2f support=%s",
            _fit_elapsed_seconds(prep_t0),
            len(runtime.wide_data),
            len(runtime.times),
            len(numeric.observation_names(runtime.spec)),
            observed_cells,
            total_cells,
            _time_span_days(runtime.times),
            _support_summary(runtime),
        )
        logger.info(
            "Manifest order: %s",
            _format_name_preview(numeric.observation_names(runtime.spec), limit=6),
        )

        inference_structure = runtime.inference_structure
        logger.info(
            "Inference route: requested_method=%s resolved_method=%s structural_backend=%s "
            "method_override=%s",
            (sampler_config or {}).get("method", "config default"),
            inference_structure.resolved_method,
            inference_structure.structural_backend,
            inference_structure.method_override or "none",
        )

        # Fit the model — returns a production particle posterior.
        logger.info("Starting inference kernel...")
        fit_t0 = time.monotonic()
        result = fit_prepared_model(runtime)
        logger.info(
            "Inference kernel complete in %.1fs: method=%s wide_rows=%d manifest_vars=%d",
            _fit_elapsed_seconds(fit_t0),
            result.method,
            len(runtime.wide_data),
            len(numeric.observation_names(runtime.spec)),
        )

        # The engine owns the telemetry payload.
        logger.info("Collecting sampler diagnostics...")
        inference_diagnostics = result.get_inference_diagnostics()

        loo_diag = None
        if compute_loo_diagnostics:
            logger.info("Computing leave-one-measurement-row-out diagnostics...")
            loo_diag = result.get_loo_diagnostics(observations=runtime.observations)
        else:
            logger.info("Skipping LOO diagnostics by configuration.")

        # Posterior marginals and pairs
        logger.info("Extracting posterior summaries...")
        posterior_marginals = result.get_posterior_marginals()
        posterior_pairs = result.get_posterior_pairs()
        from nof1_causal_lab.flows.transitions.inference.subjects import (
            reference_posterior_findings,
        )

        posterior_marginals, posterior_pairs = reference_posterior_findings(
            model_spec, posterior_marginals, posterior_pairs
        )
        samples = result.get_samples()
        n_samples = (
            int(next(iter(samples.values())).shape[0])
            if isinstance(samples, dict) and samples
            else 0
        )
        logger.info(
            "Posterior summaries ready in %.1fs: n_samples=%d",
            _fit_elapsed_seconds(t0),
            n_samples,
        )

        return {
            "fitted": True,
            "inference_type": result.method,
            "n_samples": n_samples,
            "duration_seconds": _fit_elapsed_seconds(t0),
            "result": result,
            "spec": runtime.spec,
            "runtime": runtime,
            "times": runtime.times,
            "inference_diagnostics": inference_diagnostics,
            "loo_diagnostics": loo_diag,
            "posterior_marginals": posterior_marginals,
            "posterior_pairs": posterior_pairs,
        }

    except NotImplementedError:
        logger.warning("SSM implementation not available for model fitting")
        return {
            "fitted": False,
            "error": "SSM implementation not available",
            "duration_seconds": _fit_elapsed_seconds(t0),
        }
