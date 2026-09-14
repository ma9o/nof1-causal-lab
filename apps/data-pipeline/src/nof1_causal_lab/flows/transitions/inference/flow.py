"""posterior orchestration."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.errors import ModelFitError
from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.persistence import condition_model

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.sampler_config import SamplerConfig


def _log_ppc(ppc_result: UncheckedJsonObject) -> None:
    logger.info("--- Posterior Predictive Checks ---")
    if not ppc_result.get("checked", False):
        logger.info("  Skipped: %s", ppc_result.get("error", "unknown"))
        return

    warnings = ppc_result.get("per_variable_warnings", [])
    if warnings:
        logger.warning("  %d warning(s):", len(warnings))
        for warning in warnings:
            logger.warning("    - %s: %s", warning["indicator_id"], warning["message"])
        return

    logger.info("  All checks passed")


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


def run_inference_with_data(
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
    from .fit import fit_model, run_ppc

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

    ppc_result = run_ppc(fitted_result)
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

    _log_ppc(ppc_result)

    return {
        "_model": conditioned,
        "engine_evidence": asdict(result.evidence),
        "inference_metadata": inference_metadata,
        "inference_diagnostics": fitted_result["inference_diagnostics"],
        "assessment": {
            "ppc": ppc_result,
            "loo_diagnostics": fitted_result.get("loo_diagnostics"),
        },
        "posterior_marginals": fitted_result.get("posterior_marginals"),
        "posterior_pairs": fitted_result.get("posterior_pairs"),
    }
