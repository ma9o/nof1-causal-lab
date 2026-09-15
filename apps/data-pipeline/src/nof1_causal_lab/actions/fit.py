"""Condition a model on observations without launching predictive simulation."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.errors import ModelFitError
from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.persistence import condition_model

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.sampler_config import SamplerConfig


def build_sampler_config(inference_method: str | None) -> SamplerConfig:
    """Resolve the sampler configuration from config + optional override."""
    from nof1_causal_lab.utils.config import get_config

    config = get_config()
    sampler_config = config.inference.to_sampler_config(method_override=inference_method)
    if sampler_config.get("method") in {
        "aux_kalman_mcmc",
        "pit_particle_mgrad",
        "marginal_particle_gibbs",
    }:
        sampler_config["retain_latent_paths"] = True
    return sampler_config


def fit(
    *,
    model_spec: ModelSpec,
    data_for_model: Any,
    sampler_config: SamplerConfig,
    array_writer,
    array_loader,
    workspace_id: str,
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
    inf_method = fitted_result.get("inference_type") or sampler_config.get("method", "unknown")
    inference_metadata = {
        "method": inf_method,
        "n_samples": int(fitted_result.get("n_samples", 0)),
        "duration_seconds": float(fitted_result.get("duration_seconds", 0.0)),
    }

    if not fitted_result.get("fitted", False):
        raise ModelFitError(
            fitted_result.get("error") or "model fit failed",
            transition_id="posterior",
            diagnostics={
                "inference_metadata": inference_metadata,
                "inference_diagnostics": fitted_result.get("inference_diagnostics"),
            },
        )

    result = fitted_result["result"]
    if not isinstance(result, ParticleMCMCPosterior):
        raise ModelFitError(
            "Production inference did not return a particle-MCMC posterior",
            transition_id="posterior",
        )

    conditioned = condition_model(
        model_spec,
        result,
        times=fitted_result["times"],
        array_writer=array_writer,
        array_loader=array_loader,
    )

    return {
        "_model": conditioned,
        "engine_evidence": asdict(result.evidence),
        "inference_metadata": inference_metadata,
        "inference_diagnostics": fitted_result["inference_diagnostics"],
        "loo_diagnostics": fitted_result.get("loo_diagnostics"),
        "posterior_marginals": fitted_result.get("posterior_marginals"),
        "posterior_pairs": fitted_result.get("posterior_pairs"),
    }
