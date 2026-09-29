"""Condition a model on observations without launching predictive simulation."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.errors import ModelFitError
from nof1_causal_lab.models.ssm.inference.persistence import condition_model

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import polars as pl

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.numpyro_json import ArrayLoader
    from nof1_causal_lab.sampler_config import SamplerConfig


def build_sampler_config() -> SamplerConfig:
    """Resolve the configured sampler and retain the fitted state paths."""
    from nof1_causal_lab.utils.config import get_config

    config = get_config()
    sampler_config = config.inference.to_sampler_config()
    sampler_config["retain_latent_paths"] = True
    return sampler_config


def fit(
    *,
    model_spec: ModelSpec,
    data_for_model: pl.DataFrame,
    sampler_config: SamplerConfig,
    array_writer: Callable[[np.ndarray], str],
    array_loader: ArrayLoader,
    workspace_id: str | None,
    compute_loo_diagnostics: bool,
) -> UncheckedJsonObject:
    """Fit the model from materialized model-spec/2 artifacts and shape posterior."""
    from nof1_causal_lab.flows.transitions.inference.fit import fit_model

    fitted_result = fit_model(
        model_spec,
        data_for_model,
        sampler_config=sampler_config,
        workspace_id=workspace_id,
        wait_for_compile_cache=True,
        compute_loo_diagnostics=compute_loo_diagnostics,
    )
    if not fitted_result["fitted"]:
        raise ModelFitError(
            fitted_result["error"],
            transition_id="posterior",
            diagnostics={
                "inference_metadata": {
                    "method": sampler_config.get("method", "unknown"),
                    "n_samples": 0,
                    "duration_seconds": fitted_result["duration_seconds"],
                },
                "inference_diagnostics": None,
            },
        )

    result = fitted_result["result"]
    inference_metadata = {
        "method": result.method,
        "n_samples": result.draws.describe().n_draws,
        "duration_seconds": fitted_result["duration_seconds"],
    }

    conditioned = condition_model(
        model_spec,
        result,
        times=fitted_result["runtime"].times,
        array_writer=array_writer,
        array_loader=array_loader,
    )

    return {
        "_model": conditioned,
        "engine_evidence": asdict(result.evidence),
        "inference_metadata": inference_metadata,
        "inference_diagnostics": fitted_result["inference_diagnostics"],
        "loo_diagnostics": fitted_result["loo_diagnostics"],
        "posterior_marginals": fitted_result["posterior_marginals"],
        "posterior_pairs": fitted_result["posterior_pairs"],
    }
