"""Builds the per-step :class:`SmootherContext` for MPGibbs latent smoothers."""

# References:
#   https://arxiv.org/abs/2505.04611 — Corenflos (2025), arXiv:2505.04611:
#     the per-step posterior mixture over the parameter ensemble that the smoothers
#     condition on.
#   Särkkä (2013), Bayesian Filtering and Smoothing,
#     for the Gaussian transition / initial-state moments assembled here.

from __future__ import annotations

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    MPGibbsStatic,
    SmootherContext,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _normalize_log_probs,
)


def build_smoother_context(
    static: MPGibbsStatic,
    state,
    parameter_particles: jnp.ndarray,
    label_correction: jnp.ndarray,
) -> SmootherContext:
    """Assemble per-step latent contexts and shared smoother helper closures."""
    latent_context_runtime_fn = static.latent_context_runtime_fn
    log_prior_unc_fn = static.log_prior_unc_fn
    obs_increment_fn = static.obs_increment_fn
    trajectory_log_prob_fn = static.trajectory_log_prob_fn
    runtime_observations = static.runtime_observations
    runtime_times = static.runtime_times
    num_parameter_particles = static.num_parameter_particles
    transition_initial_log_prob_fn = static.transition_initial_log_prob_fn
    transition_log_prob_fn = static.transition_log_prob_fn
    transition_log_probs_for_pairs_fn = static.transition_log_probs_for_pairs_fn
    transition_pairwise_log_probs_fn = static.transition_pairwise_log_probs_fn

    x_ref = state.latent_trajectory
    latent_dtype = x_ref.dtype
    traj_dtype = state.trajectory_log_prob.dtype
    num_steps = int(x_ref.shape[0])
    num_free_particles = static.num_particles - 1

    with jax.named_scope("build_contexts"):
        contexts = jax.vmap(lambda z: latent_context_runtime_fn(z, runtime_times))(
            parameter_particles
        )
        parameter_log_probs = (
            jax.vmap(log_prior_unc_fn)(parameter_particles) + label_correction
        ).astype(traj_dtype)
        initial_label_log_probs = _normalize_log_probs(parameter_log_probs)

    def _initial_value_grad_by_param(particle0: jnp.ndarray):
        def _one_context(context):
            return jax.value_and_grad(
                lambda particle: transition_initial_log_prob_fn(context, particle)
            )(particle0)

        log_prob, grad = jax.vmap(_one_context)(contexts)
        return log_prob.astype(traj_dtype), grad.astype(latent_dtype)

    def _transition_current_value_grad_by_param(
        prev_particle: jnp.ndarray,
        particle_t: jnp.ndarray,
        time_idx: jnp.ndarray,
    ):
        def _one_context(context):
            return jax.value_and_grad(
                lambda particle: transition_log_prob_fn(
                    context,
                    prev_particle,
                    particle,
                    time_idx,
                )
            )(particle_t)

        log_prob, grad_current = jax.vmap(_one_context)(contexts)
        return log_prob.astype(traj_dtype), grad_current.astype(latent_dtype)

    def _transition_next_value_grad_by_param(
        particle_t: jnp.ndarray,
        next_particle: jnp.ndarray,
        next_time_idx: jnp.ndarray,
    ):
        def _one_context(context):
            return jax.value_and_grad(
                lambda particle: transition_log_prob_fn(
                    context,
                    particle,
                    next_particle,
                    next_time_idx,
                )
            )(particle_t)

        log_prob, grad_prev = jax.vmap(_one_context)(contexts)
        return log_prob.astype(traj_dtype), grad_prev.astype(latent_dtype)

    def _selected_transition_log_probs(
        prev_particles: jnp.ndarray,
        next_particles: jnp.ndarray,
        seam: jnp.ndarray,
    ) -> jnp.ndarray:
        seam_clamped = jnp.minimum(seam, num_steps - 1)
        real_seam = seam < num_steps
        per_param = jax.vmap(
            lambda context: transition_log_probs_for_pairs_fn(
                context,
                prev_particles,
                next_particles,
                seam_clamped,
            )
        )(contexts)
        return jnp.where(real_seam, jnp.swapaxes(per_param, 0, 1), 0.0).astype(traj_dtype)

    def _pairwise_transition_log_probs(
        prev_particles: jnp.ndarray,
        next_particles: jnp.ndarray,
        seam: jnp.ndarray,
    ) -> jnp.ndarray:
        seam_clamped = jnp.minimum(seam, num_steps - 1)
        real_seam = seam < num_steps
        per_param = jax.vmap(
            lambda context: transition_pairwise_log_probs_fn(
                context,
                prev_particles,
                next_particles,
                seam_clamped,
            )
        )(contexts)
        transition_lp = jnp.moveaxis(per_param, 0, -1)
        return jnp.where(real_seam, transition_lp, 0.0).astype(traj_dtype)

    def _trajectory_label_log_probs(path: jnp.ndarray) -> jnp.ndarray:
        def _one_context(context):
            return trajectory_log_prob_fn(
                context,
                path,
                runtime_observations,
            )

        trajectory_log_probs = jax.vmap(_one_context)(contexts).astype(traj_dtype)
        return _normalize_log_probs(parameter_log_probs + trajectory_log_probs).astype(traj_dtype)

    return SmootherContext(
        contexts=contexts,
        initial_label_log_probs=initial_label_log_probs,
        num_steps=num_steps,
        num_free_particles=num_free_particles,
        num_parameter_particles=num_parameter_particles,
        latent_dtype=latent_dtype,
        traj_dtype=traj_dtype,
        obs_increment_fn=obs_increment_fn,
        runtime_observations=runtime_observations,
        latent_free_mask=static.latent_free_mask,
        amala_delta=jnp.asarray(state.latent_delta, dtype=latent_dtype),
        amala_kappa=static.amala_kappa,
        amala_grad_clip=static.amala_grad_clip,
        dsmc_leaf_proposal=static.dsmc_leaf_proposal,
        latent_block_coords=static.latent_block_coords,
        paid_mix_z_weight=static.paid_mix_z_weight,
        paid_mix_pilot_weight=static.paid_mix_pilot_weight,
        pilot_means=(
            None
            if static.pilot_means is None
            else jnp.asarray(static.pilot_means, dtype=latent_dtype)
        ),
        pilot_vars=(
            None
            if static.pilot_vars is None
            else jnp.asarray(static.pilot_vars, dtype=latent_dtype)
        ),
        pilot_wide_vars=(
            None
            if static.pilot_wide_vars is None
            else jnp.asarray(static.pilot_wide_vars, dtype=latent_dtype)
        ),
        initial_value_grad_by_param=_initial_value_grad_by_param,
        transition_current_value_grad_by_param=_transition_current_value_grad_by_param,
        transition_next_value_grad_by_param=_transition_next_value_grad_by_param,
        selected_transition_log_probs=_selected_transition_log_probs,
        pairwise_transition_log_probs=_pairwise_transition_log_probs,
        trajectory_label_log_probs=_trajectory_label_log_probs,
    )
