"""Condition a model on observations without launching predictive simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from nof1_causal_lab.actions.errors import ModelFitError
from nof1_causal_lab.artifacts.checks import Evaluated
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceMetadata, InferenceReport
from nof1_causal_lab.artifacts.posterior_diagnostics import ParticleMCMCEvidence
from nof1_causal_lab.models.ssm.inference.convergence import parameter_convergence
from nof1_causal_lab.models.ssm.inference.persistence import condition_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    import numpy as np
    import polars as pl

    from nof1_causal_lab.numpyro_json import ArrayLoader
    from nof1_causal_lab.sampler_config import SamplerSpec


class FitResult(TypedDict):
    """Serialized fit report plus the conditioned model and engine evidence."""

    _model: ModelSpec
    engine_evidence: ParticleMCMCEvidence
    report: InferenceReport


def fit(
    *,
    model_spec: ModelSpec,
    data_for_model: pl.DataFrame,
    time_origin: datetime | None,
    sampler: SamplerSpec,
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
        sampler=sampler,
        compute_loo_diagnostics=compute_loo_diagnostics,
    )
    if "error" not in fitted_result:
        result = fitted_result["result"]
        inference_metadata = InferenceMetadata(
            method=result.method,
            n_samples=result.draws.describe().n_draws,
            duration_seconds=fitted_result["duration_seconds"],
        )

        conditioned = condition_model(
            model_spec,
            fitted_result["panel"].model,
            result,
            times=fitted_result["panel"].times,
            array_writer=array_writer,
            array_loader=array_loader,
        )

        return {
            "_model": conditioned,
            "engine_evidence": result.evidence,
            "report": InferenceReport(
                time_origin=time_origin,
                inference_metadata=inference_metadata,
                engine=Evaluated(
                    subject="production_engine", outcome="passed", evidence=result.evidence
                ),
                inference_diagnostics=fitted_result["inference_diagnostics"],
                sampler_diagnostics=result.diagnostics.marginal_particle_gibbs,
                convergence=parameter_convergence(fitted_result["inference_diagnostics"]),
                loo_diagnostics=fitted_result["loo_diagnostics"][0]
                if fitted_result["loo_diagnostics"]
                else None,
                posterior_marginals=fitted_result["posterior_marginals"],
                detail=fitted_result["detail"],
            ),
        }
    raise ModelFitError(
        fitted_result["error"],
        diagnostics={
            "inference_metadata": {
                "method": "marginal_particle_gibbs",
                "n_samples": 0,
                "duration_seconds": fitted_result["duration_seconds"],
            },
            "inference_diagnostics": None,
        },
    )
