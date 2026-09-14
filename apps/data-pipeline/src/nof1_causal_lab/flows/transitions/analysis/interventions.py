"""analysis intervention task."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import jax.numpy as jnp

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.models.ssm import numerics as numeric

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nof1_causal_lab.models.causal_proofs import CertifiedCausalAnalysis


def run_interventions(
    analysis: CertifiedCausalAnalysis,
) -> list[UncheckedJsonObject]:
    """Run interventions only after identification and engine proofs are joined."""
    from nof1_causal_lab.models.ssm.counterfactual import compute_interventions
    from nof1_causal_lab.models.ssm.dynamics import posterior_dynamics_from_samples

    treatments = analysis.treatments
    outcome = analysis.outcome
    logger.info(
        "Running interventions: treatments=%d outcome=%s fitted=%s",
        len(treatments),
        outcome or "unknown",
        True,
    )

    from nof1_causal_lab.models.ssm.inference.persistence import model_draws

    spec = analysis.model
    samples = model_draws(spec).parameters

    latent_names = numeric.state_names(spec)
    if latent_names is None:
        latent_names = numeric.observation_names(spec) or []

    manifest_names = numeric.observation_names(spec) or []

    posterior_dynamics = posterior_dynamics_from_samples(spec, samples)
    lambda_draws = samples.get("lambda")
    lambda_mean = None
    if lambda_draws is not None:
        lambda_mean = lambda_draws.mean(axis=0) if lambda_draws.ndim == 3 else lambda_draws
    results = compute_interventions(
        param_samples=posterior_dynamics.param_samples,
        vector_field=posterior_dynamics.vector_field,
        treatments=treatments,
        outcome=outcome,
        latent_names=latent_names,
        measurement_clock=analysis.model.measurement_clock,
        manifest_names=manifest_names,
        times=jnp.asarray(spec.time_points),
        lambda_mean=lambda_mean,
    )
    construct_ids = {item.name: item.id for item in analysis.model.constructs}
    for entry in results:
        entry["treatment_id"] = construct_ids[entry["treatment"]]
    logger.info("Interventions complete: ranked_treatments=%d", len(results))
    return results
