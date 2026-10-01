"""Condition a model on observations without launching predictive simulation."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, TypedDict

from nof1_causal_lab.actions.errors import ModelFitError
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceMetadata
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.models.ssm.inference.persistence import condition_model
from nof1_causal_lab.models.ssm.inference.types import PosteriorDiagnostics

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    import numpy as np
    import polars as pl

    from nof1_causal_lab.numpyro_json import ArrayLoader
    from nof1_causal_lab.sampler_config import SamplerConfig


class FitResult(TypedDict):
    """Serialized fit report plus the conditioned model and engine evidence."""

    time_origin: str | None
    _model: ModelSpec
    engine_evidence: JsonObject
    inference_metadata: JsonObject
    inference_diagnostics: PosteriorDiagnostics
    loo_diagnostics: JsonObject | None
    posterior_marginals: list[JsonObject]
    posterior_pairs: list[JsonObject]


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
    time_origin: datetime | None,
    sampler_config: SamplerConfig,
    array_writer: Callable[[np.ndarray], str],
    array_loader: ArrayLoader,
    compute_loo_diagnostics: bool,
) -> FitResult:
    """Fit the model from materialized model-spec/2 artifacts and shape posterior."""
    from nof1_causal_lab.actions.inference.fit import fit_model

    fitted_result = fit_model(
        model_spec,
        data_for_model,
        time_origin=time_origin,
        sampler_config=sampler_config,
        compute_loo_diagnostics=compute_loo_diagnostics,
    )
    if not fitted_result["fitted"]:
        raise ModelFitError(
            fitted_result["error"],
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
    inference_metadata = InferenceMetadata(
        method=result.method,
        n_samples=result.draws.describe().n_draws,
        duration_seconds=fitted_result["duration_seconds"],
    )

    conditioned = condition_model(
        fitted_result["runtime"].model.inputs,
        result,
        times=fitted_result["runtime"].times,
        array_writer=array_writer,
        array_loader=array_loader,
    )

    return {
        "time_origin": time_origin.isoformat() if time_origin is not None else None,
        "_model": conditioned,
        "engine_evidence": asdict(result.evidence),
        "inference_metadata": inference_metadata.model_dump(mode="json"),
        "inference_diagnostics": fitted_result["inference_diagnostics"],
        "loo_diagnostics": fitted_result["loo_diagnostics"],
        "posterior_marginals": fitted_result["posterior_marginals"],
        "posterior_pairs": fitted_result["posterior_pairs"],
    }
