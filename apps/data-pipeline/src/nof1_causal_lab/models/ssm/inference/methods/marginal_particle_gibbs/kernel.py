"""Marginalized Particle Gibbs joint parameter/trajectory kernel.

Implements the M=2-by-default collapsed Particle Gibbs construction from
Corenflos (2025), "Particle Gibbs without the Gibbs bit", for directly
evaluable SSM potentials. The parameter proposal is formed in unconstrained
space via the auxiliary decomposition

    u | theta  ~ N(theta + 2 delta Sigma g(theta), 2 delta Sigma)
    theta' | u ~ N(u, 2 delta Sigma)

where ``g`` is a fixed, parameter-only gradient oracle. The default computes
the complete-data gradient on a fixed pilot path chosen at construction; it
must not use the changing reference trajectory during the chain. ``parameter_proposal``
selects the drift: ``random_walk`` sets ``g = 0`` (the symmetric special case),
while the default ``pseudo_langevin`` (Corenflos 2025, §3.1) drifts the
theta->u half by ``g`` so the two halves reproduce preconditioned MALA. The
resulting asymmetry is corrected exactly in the Barker label-selection weights
(identically zero in the random-walk case). The latent trajectory is updated by
conditional SMC against the posterior mixture over the parameter ensemble.

Step-size convention. ``delta`` here (``param_step_size``) is the variance
coefficient of each half, not the conventional MALA step. The combined kernel
has variance ``4 delta Sigma`` (and drift ``2 delta Sigma g`` for pseudo-Langevin),
so ``param_step_size`` maps to MALA's ``h`` and to the RW variance coefficient
via ``h = 4 * param_step_size`` in both branches. Tune accordingly: a familiar
MALA step of ``h = 0.1`` corresponds to ``param_step_size = 0.025`` here.
"""

# References:
#   https://arxiv.org/abs/2505.04611 — Corenflos (2025), "Particle Gibbs
#     without the Gibbs bit" (arXiv:2505.04611): the collapsed M-ensemble parameter
#     proposal (§3.1 pseudo-Langevin) and Barker label-selection weights live here.

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCState, _clip_scale
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._context import (
    build_smoother_context,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    _DSMC_LEAF_PROPOSAL_PAID_MIX,
    _DSMC_LEAF_PROPOSALS,
    _LATENT_SMOOTHER_DSMC,
    DSMCLeafProposal,
    MPGibbsLatentSmoother,
    MPGibbsStatic,
    _resolve_latent_smoother,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _masked_mean,
    _select_pytree,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.diagnostics import (
    build_mpgibbs_diagnostic_flags,
    resolve_mpgibbs_diagnostic_metrics,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.smoothers.dsmc import (
    step as dsmc_step,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx.inference.particle_runtime import ParticleRuntime

    from nof1_causal_lab.models.ssm.inference.conditioning import ExactStateConstraints


class SignFlipSpec(NamedTuple):
    """Coordinate and parameter masks for an exact joint reflection move."""

    coords: jnp.ndarray
    masks: jnp.ndarray


_DEFAULT_MIN_SCALE = 1e-6
_DEFAULT_MAX_SCALE = 1e3
_DEFAULT_AMALA_DELTA_INIT = 1e-2
_DEFAULT_AMALA_DELTA_MIN = 1e-5
_DEFAULT_AMALA_DELTA_MAX = 1e1
_DEFAULT_AMALA_TARGET_ACCEPT = 0.75
_DEFAULT_AMALA_ADAPTATION_WINDOW = 100
_DEFAULT_AMALA_ADAPTATION_TOLERANCE = 0.05
_DEFAULT_AMALA_ADAPTATION_RHO = 0.5
_DEFAULT_AMALA_ADAPTATION_RHO_MIN = 1e-3
_DEFAULT_AMALA_ADAPTATION_GAMMA = -0.5
_DEFAULT_AMALA_GRAD_CLIP = float("inf")
# Target acceptance for the M=2 ensemble Barker selection. Its move-rate is
# bounded above by ~0.5 -- the step->0 coin-flip limit, (M-1)/M for M ensemble
# candidates -- so MALA-style optima (~0.574) do NOT transfer: a target above
# that ceiling makes dual averaging chase an unreachable rate and collapse the
# step size to the floor (observed empirically at 0.57). 0.35 sits safely below
# the ceiling, is reachable under dual averaging, and matches the long-standing
# baseline. The ceiling is a property of the M=2 selection, not of the proposal
# drift, so this default is shared by random_walk and pseudo_langevin.
_DEFAULT_PARAM_TARGET_ACCEPT = 0.35


def _uses_amala_delta(latent_smoother: MPGibbsLatentSmoother) -> bool:
    # Every dsmc leaf is z-anchored on the reference with scale delta/2, so the
    # per-time delta adaptation always applies.
    return latent_smoother.name == _LATENT_SMOOTHER_DSMC


class MarginalParticleGibbsKernel(NamedTuple):
    """Callable joint kernel and static metadata."""

    step_fn: Callable[
        [TrajectoryMCMCState, jnp.ndarray], tuple[TrajectoryMCMCState, dict[str, jnp.ndarray]]
    ]
    num_particles: int
    num_parameter_particles: int
    initial_param_step_size: float
    target_accept: float
    min_scale: float
    max_scale: float
    preconditioned: bool
    latent_smoother: MPGibbsLatentSmoother
    latent_delta: float
    amala_delta_init: float
    amala_delta_min: float
    amala_delta_max: float
    amala_target_accept: float
    amala_adaptation_window: int
    amala_adaptation_tolerance: float
    amala_adaptation_rho: float
    amala_adaptation_rho_min: float
    amala_adaptation_gamma: float
    adapt_amala_delta: bool
    amala_kappa: float
    amala_grad_clip: float
    dsmc_leaf_proposal: DSMCLeafProposal
    latent_block_coords: int | None
    diagnostic_metrics: frozenset[str]
    exact_constraints: ExactStateConstraints | None


def build_marginal_particle_gibbs_kernel(
    target: ParticleRuntime,
    *,
    num_particles: int,
    num_parameter_particles: int,
    param_step_size: float,
    target_accept: float | None = None,
    min_scale: float = _DEFAULT_MIN_SCALE,
    max_scale: float = _DEFAULT_MAX_SCALE,
    parameter_preconditioner_chol: jnp.ndarray | None = None,
    parameter_proposal: str = "pseudo_langevin",
    latent_smoother: str = _LATENT_SMOOTHER_DSMC,
    latent_delta: float = 0.2,
    amala_delta_init: float = _DEFAULT_AMALA_DELTA_INIT,
    amala_delta_min: float = _DEFAULT_AMALA_DELTA_MIN,
    amala_delta_max: float = _DEFAULT_AMALA_DELTA_MAX,
    amala_target_accept: float = _DEFAULT_AMALA_TARGET_ACCEPT,
    amala_adaptation_window: int = _DEFAULT_AMALA_ADAPTATION_WINDOW,
    amala_adaptation_tolerance: float = _DEFAULT_AMALA_ADAPTATION_TOLERANCE,
    amala_adaptation_rho: float = _DEFAULT_AMALA_ADAPTATION_RHO,
    amala_adaptation_rho_min: float = _DEFAULT_AMALA_ADAPTATION_RHO_MIN,
    amala_adaptation_gamma: float = _DEFAULT_AMALA_ADAPTATION_GAMMA,
    amala_kappa: float = 0.75,
    amala_grad_clip: float = _DEFAULT_AMALA_GRAD_CLIP,
    dsmc_leaf_proposal: DSMCLeafProposal = "amala_exact",
    latent_block_coords: int | None = None,
    paid_mix_z_weight: float = 0.85,
    paid_mix_pilot_weight: float = 0.10,
    pilot_means: jnp.ndarray | None = None,
    pilot_vars: jnp.ndarray | None = None,
    pilot_wide_vars: jnp.ndarray | None = None,
    sign_flip_spec: SignFlipSpec | None = None,
    parameter_reference_path: jnp.ndarray | None = None,
    exact_constraints: ExactStateConstraints | None = None,
    diagnostic_metrics_all: bool = False,
    diagnostic_metrics: tuple[str, ...] | list[str] | None = None,
) -> MarginalParticleGibbsKernel:
    """Build a marginalized Particle Gibbs joint state update."""
    latent_smoother_spec = _resolve_latent_smoother(latent_smoother)
    resolved_diagnostic_metrics = resolve_mpgibbs_diagnostic_metrics(
        diagnostic_metrics_all=diagnostic_metrics_all,
        diagnostic_metrics=diagnostic_metrics,
    )
    diagnostic_flags = build_mpgibbs_diagnostic_flags(
        diagnostic_metrics=resolved_diagnostic_metrics,
    )
    if num_particles < 2:
        raise ValueError("marginal_particle_gibbs requires num_particles >= 2.")
    if num_parameter_particles < 2:
        raise ValueError("marginal_particle_gibbs requires num_parameter_particles >= 2.")
    if min_scale > max_scale:
        raise ValueError(
            "marginal_particle_gibbs min_scale must be <= max_scale; "
            f"got {min_scale} > {max_scale}."
        )
    if parameter_proposal not in ("random_walk", "pseudo_langevin"):
        raise ValueError(
            "marginal_particle_gibbs parameter_proposal must be 'random_walk' or "
            f"'pseudo_langevin'; got {parameter_proposal!r}."
        )
    if amala_delta_init <= 0.0:
        raise ValueError(
            f"marginal_particle_gibbs amala_delta_init must be positive; got {amala_delta_init}."
        )
    if amala_delta_min <= 0.0 or amala_delta_max <= 0.0 or amala_delta_min > amala_delta_max:
        raise ValueError(
            "marginal_particle_gibbs requires 0 < amala_delta_min <= amala_delta_max; "
            f"got {amala_delta_min} and {amala_delta_max}."
        )
    if not (amala_delta_min <= amala_delta_init <= amala_delta_max):
        raise ValueError(
            "marginal_particle_gibbs requires amala_delta_init inside "
            f"[amala_delta_min, amala_delta_max]; got {amala_delta_init}."
        )
    if not (0.0 < amala_target_accept < 1.0):
        raise ValueError(
            "marginal_particle_gibbs amala_target_accept must be in (0, 1); "
            f"got {amala_target_accept}."
        )
    if amala_adaptation_window < 1:
        raise ValueError(
            "marginal_particle_gibbs amala_adaptation_window must be positive; "
            f"got {amala_adaptation_window}."
        )
    if amala_adaptation_tolerance < 0.0:
        raise ValueError(
            "marginal_particle_gibbs amala_adaptation_tolerance must be non-negative; "
            f"got {amala_adaptation_tolerance}."
        )
    if amala_adaptation_rho <= 0.0 or amala_adaptation_rho_min <= 0.0:
        raise ValueError(
            "marginal_particle_gibbs amala adaptation rates must be positive; "
            f"got rho={amala_adaptation_rho}, rho_min={amala_adaptation_rho_min}."
        )
    if amala_kappa < 0.0:
        raise ValueError(
            f"marginal_particle_gibbs amala_kappa must be non-negative; got {amala_kappa}."
        )
    if amala_grad_clip <= 0.0:
        raise ValueError(
            f"marginal_particle_gibbs amala_grad_clip must be positive; got {amala_grad_clip}."
        )
    if dsmc_leaf_proposal not in _DSMC_LEAF_PROPOSALS:
        allowed = ", ".join(repr(candidate) for candidate in _DSMC_LEAF_PROPOSALS)
        raise ValueError(
            "marginal_particle_gibbs dsmc_leaf_proposal must be one of "
            f"{allowed}; got {dsmc_leaf_proposal!r}."
        )
    if latent_block_coords is not None and latent_block_coords < 1:
        raise ValueError(
            "marginal_particle_gibbs latent_block_coords must be a positive coordinate "
            f"count or None (all coordinates); got {latent_block_coords}."
        )
    if dsmc_leaf_proposal == _DSMC_LEAF_PROPOSAL_PAID_MIX:
        if not (0.0 < paid_mix_z_weight < 1.0) or not (0.0 < paid_mix_pilot_weight < 1.0):
            raise ValueError(
                "marginal_particle_gibbs paid_mix weights must lie in (0, 1); got "
                f"z={paid_mix_z_weight}, pilot={paid_mix_pilot_weight}."
            )
        if paid_mix_z_weight + paid_mix_pilot_weight >= 1.0:
            raise ValueError(
                "marginal_particle_gibbs paid_mix weights must leave a positive wide-tail "
                f"share; got z + pilot = {paid_mix_z_weight + paid_mix_pilot_weight}."
            )
        if pilot_means is None or pilot_vars is None or pilot_wide_vars is None:
            raise ValueError(
                "marginal_particle_gibbs dsmc_leaf_proposal='paid_mix' requires pilot "
                "moments (pilot_means, pilot_vars, pilot_wide_vars) from a fixed pilot initializer."
            )
    use_gradient_drift = parameter_proposal == "pseudo_langevin"
    if target_accept is None:
        target_accept = _DEFAULT_PARAM_TARGET_ACCEPT

    latent_context_runtime_fn = target.context
    log_prior_unc_fn = target.log_prior
    obs_increment_fn = target.observation_increment
    trajectory_log_prob_fn = target.path_log_prob
    runtime_observations = target.observations
    runtime_times = target.times

    fixed_path = (
        target.initial_path(target.context(target.initial_position, runtime_times))
        if parameter_reference_path is None
        else jnp.asarray(parameter_reference_path)
    )
    if exact_constraints is not None:
        fixed_path = exact_constraints.project(fixed_path)
    latent_free_mask = (
        jnp.ones(fixed_path.shape, dtype=bool)
        if exact_constraints is None
        else exact_constraints.free_mask
    )

    def _theta_logpost_grad(z: jnp.ndarray) -> jnp.ndarray:
        # q(u | theta) uses a theta-only oracle, fixed for every kernel call.
        # Conditioning this oracle on the current path invalidates the label correction.
        return jax.grad(
            lambda zz: target.log_posterior(zz, fixed_path, runtime_observations, runtime_times)
        )(z)

    preconditioner = (
        None
        if parameter_preconditioner_chol is None
        else jnp.asarray(
            parameter_preconditioner_chol,
            dtype=target.initial_position.dtype,
        )
    )

    def _propose_parameter_ensemble(
        current_position: jnp.ndarray,
        key: jnp.ndarray,
        step_size: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        dim = int(current_position.shape[0])
        step = jnp.asarray(step_size, dtype=current_position.dtype)
        proposal_scale = jnp.sqrt(jnp.asarray(2.0, dtype=current_position.dtype) * step)
        if preconditioner is None:
            chol = jnp.eye(dim, dtype=current_position.dtype)
        else:
            chol = jnp.asarray(preconditioner, dtype=current_position.dtype)
        aux_key, proposal_key = random.split(key)

        # Pseudo-Langevin (Corenflos 2025, §3.1): drift the θ→u half by 2·step·Σ·g(θ),
        # with Σ = chol·cholᵀ and g the fixed parameter-only gradient oracle. The θ'←u
        # half stays a plain Gaussian, so only u carries the drift. Combined, the two
        # halves reproduce preconditioned MALA; random-walk is the zero-drift case.
        if use_gradient_drift:
            with jax.named_scope("propose_grad_drift"):
                grad_ref = _theta_logpost_grad(current_position)
                drift = (jnp.asarray(2.0, dtype=step.dtype) * step) * (chol @ (chol.T @ grad_ref))
        else:
            grad_ref = jnp.zeros_like(current_position)
            drift = jnp.zeros_like(current_position)

        u = (
            current_position
            + drift
            + proposal_scale
            * (
                random.normal(aux_key, current_position.shape, dtype=current_position.dtype)
                @ chol.T
            )
        )
        proposal_eps = random.normal(
            proposal_key,
            (num_parameter_particles - 1, dim),
            dtype=current_position.dtype,
        )
        proposed = u[None, :] + proposal_scale * (proposal_eps @ chol.T)
        ensemble = jnp.concatenate([current_position[None, :], proposed], axis=0)

        # Barker label-prior correction Δ_l = log q(u|θˡ) − log q(θˡ|u). With the drift
        # only in q(u|·), Δ_l = g(θˡ)·(u−θˡ) − step·‖cholᵀ g(θˡ)‖². It is identically 0
        # in the random-walk (symmetric) case, recovering eq. (14). Added to the label
        # prior so the marginalized-PGibbs selection stays exact under the asymmetry.
        if use_gradient_drift:
            with jax.named_scope("propose_grad_barker"):
                grads = jnp.concatenate(
                    [
                        grad_ref[None, :],
                        jax.vmap(lambda th: _theta_logpost_grad(th))(proposed),
                    ],
                    axis=0,
                )
                whitened_sq = jnp.sum((grads @ chol) ** 2, axis=1)
                drift_dot = jnp.sum(grads * (u[None, :] - ensemble), axis=1)
                label_correction = drift_dot - step * whitened_sq
        else:
            label_correction = jnp.zeros((num_parameter_particles,), dtype=current_position.dtype)

        return ensemble, label_correction

    static = MPGibbsStatic(
        latent_context_runtime_fn=latent_context_runtime_fn,
        log_prior_unc_fn=log_prior_unc_fn,
        obs_increment_fn=obs_increment_fn,
        trajectory_log_prob_fn=trajectory_log_prob_fn,
        runtime_observations=runtime_observations,
        runtime_times=runtime_times,
        latent_free_mask=latent_free_mask,
        num_particles=num_particles,
        num_parameter_particles=num_parameter_particles,
        latent_delta=latent_delta,
        amala_kappa=amala_kappa,
        amala_grad_clip=amala_grad_clip,
        dsmc_leaf_proposal=dsmc_leaf_proposal,
        latent_block_coords=latent_block_coords,
        paid_mix_z_weight=paid_mix_z_weight,
        paid_mix_pilot_weight=paid_mix_pilot_weight,
        pilot_means=pilot_means,
        pilot_vars=pilot_vars,
        pilot_wide_vars=pilot_wide_vars,
        transition_initial_log_prob_fn=target.initial_log_prob,
        transition_log_prob_fn=target.transition_log_prob,
        transition_log_probs_for_pairs_fn=target.aligned_transition_log_prob,
        transition_pairwise_log_probs_fn=target.pairwise_transition_log_prob,
        diagnostic_metrics=resolved_diagnostic_metrics,
    )

    def _step_fn(state: TrajectoryMCMCState, key: jnp.ndarray):
        if sign_flip_spec is not None:
            param_key, block_key, label_key, flip_choice_key, flip_accept_key = random.split(key, 5)
            flip_keys = (flip_choice_key, flip_accept_key)
        else:
            param_key, block_key, label_key = random.split(key, 3)
            flip_keys = None
        x_ref = state.latent_trajectory
        traj_dtype = state.trajectory_log_prob.dtype
        complete_dtype = state.complete_log_posterior.dtype

        with jax.named_scope("propose_parameters"):
            parameter_particles, label_correction = _propose_parameter_ensemble(
                state.position,
                param_key,
                _clip_scale(
                    state.param_step_size,
                    min_scale=min_scale,
                    max_scale=max_scale,
                ),
            )

        with jax.named_scope("build_context"):
            ctx = build_smoother_context(static, state, parameter_particles, label_correction)
        contexts = ctx.contexts
        with jax.named_scope(latent_smoother_spec.name + "_smoother_full"):
            smoother_result = dsmc_step(ctx, block_key, x_ref)

        with jax.named_scope("postprocess"):
            latent_path = smoother_result.latent_path
            final_label_log_probs = smoother_result.final_label_log_probs
            origin_path = smoother_result.origin_path
            zero_scalar = jnp.asarray(0.0, dtype=traj_dtype)
            amala_grad_norm_mean = smoother_result.diagnostics.get(
                "amala_grad_norm_mean",
                zero_scalar,
            )
            amala_grad_norm_max = smoother_result.diagnostics.get(
                "amala_grad_norm_max",
                zero_scalar,
            )
            selected_label = random.categorical(label_key, final_label_log_probs).astype(jnp.int32)
            next_position = parameter_particles[selected_label]
            next_context = _select_pytree(contexts, selected_label)
            next_traj_lp = jnp.asarray(
                trajectory_log_prob_fn(
                    next_context,
                    latent_path,
                    runtime_observations,
                ),
                dtype=traj_dtype,
            )
            next_complete = jnp.asarray(log_prior_unc_fn(next_position), dtype=complete_dtype)
            next_complete = next_complete + next_traj_lp.astype(complete_dtype)
            sign_flip_diagnostics: dict[str, jnp.ndarray] = {}

            if sign_flip_spec is not None and flip_keys is not None:
                # Joint (latent coordinate, loading column) sign-flip MH move composed
                # after the smoother sweep: the flip is a state-independent involution,
                # so the acceptance is the exact joint posterior ratio — which prices
                # the sign-asymmetric loading prior and any drift coupling. This is
                # the escape route between the factor-sign mirror basins that the
                # alternating conditionals cannot cross on their own.
                flip_choice_key, flip_accept_key = flip_keys
                n_flippable = int(sign_flip_spec.coords.shape[0])
                flip_idx = random.randint(flip_choice_key, (), 0, n_flippable)
                flip_coord = sign_flip_spec.coords[flip_idx]
                position_mask = sign_flip_spec.masks[flip_idx]
                flipped_position = jnp.where(position_mask, -next_position, next_position)
                latent_dim = int(latent_path.shape[1])
                coord_sign = jnp.where(
                    jnp.arange(latent_dim, dtype=jnp.int32) == flip_coord,
                    -jnp.ones((latent_dim,), dtype=latent_path.dtype),
                    jnp.ones((latent_dim,), dtype=latent_path.dtype),
                )
                flipped_latent = latent_path * coord_sign[None, :]
                flipped_context = latent_context_runtime_fn(flipped_position, runtime_times)
                flipped_complete, flipped_traj_lp = target.log_posterior_from_context(
                    flipped_position,
                    flipped_context,
                    flipped_latent,
                    runtime_observations,
                )
                flip_delta = flipped_complete.astype(complete_dtype) - next_complete
                flip_accepted = (
                    jnp.log(random.uniform(flip_accept_key, dtype=flip_delta.dtype)) < flip_delta
                )
                next_position = jnp.where(flip_accepted, flipped_position, next_position)
                latent_path = jnp.where(flip_accepted, flipped_latent, latent_path)
                next_traj_lp = jnp.where(
                    flip_accepted,
                    jnp.asarray(flipped_traj_lp, dtype=traj_dtype),
                    next_traj_lp,
                )
                next_complete = jnp.where(
                    flip_accepted,
                    flipped_complete.astype(complete_dtype),
                    next_complete,
                )
                next_context = jax.tree_util.tree_map(
                    lambda flipped, kept: jnp.where(flip_accepted, flipped, kept),
                    flipped_context,
                    next_context,
                )
                sign_flip_diagnostics = {
                    "sign_flip_accepted": flip_accepted.astype(state.position.dtype),
                    "sign_flip_delta": flip_delta.astype(jnp.float32),
                }

            latent_move = latent_path - x_ref
            latent_move_rms_per_t = jnp.sqrt(
                _masked_mean(latent_move * latent_move, latent_free_mask, axis=-1)
            )
            latent_move_rms = jnp.sqrt(_masked_mean(latent_move * latent_move, latent_free_mask))
            latent_move_max_abs = jnp.max(jnp.abs(latent_move))
            parameter_accepted = (selected_label != 0).astype(state.position.dtype)
            latent_updated = ((origin_path != 0) & jnp.any(latent_free_mask, axis=-1)).astype(
                state.position.dtype
            )
            # Per-(t, d) exact-equality freeze indicator. A completely stuck chain has
            # zero autocovariance at every lag and therefore reports PERFECT ESS under
            # initial-positive-sequence estimators — this counter is the direct gauge
            # standard diagnostics structurally cannot provide. With coordinate-block
            # proposals it also resolves per-coordinate freezing that the per-t
            # `latent_accepted` trace cannot see.
            latent_frozen = (latent_path == x_ref).astype(state.position.dtype)
            latent_frozen_frac = _masked_mean(latent_frozen, latent_free_mask)
            latent_frozen_frac_by_d = _masked_mean(latent_frozen, latent_free_mask, axis=0)

            step_info = {
                "parameter_accepted": parameter_accepted,
                "latent_accepted": latent_updated,
                "latent_frozen_frac": latent_frozen_frac,
                "latent_frozen_frac_by_d": latent_frozen_frac_by_d,
                "selected_label": selected_label.astype(jnp.float32),
                "final_particle": origin_path[-1].astype(jnp.float32),
                "latent_move_rms": latent_move_rms,
                "latent_move_max_abs": latent_move_max_abs,
                "latent_move_rms_per_t": latent_move_rms_per_t,
                "final_label_log_probs": final_label_log_probs.astype(jnp.float32),
                "amala_grad_norm_mean": amala_grad_norm_mean.astype(jnp.float32),
                "amala_grad_norm_max": amala_grad_norm_max.astype(jnp.float32),
            }
            step_info.update(sign_flip_diagnostics)
            if diagnostic_flags.parameter_movement:
                parameter_jump = next_position - state.position
                step_info["parameter_jump_rms"] = jnp.sqrt(
                    jnp.mean(parameter_jump * parameter_jump)
                )
            if diagnostic_flags.particle_identity:
                reference_path_hit_rate = jnp.mean((origin_path == 0).astype(state.position.dtype))
                particle_ids = jnp.arange(num_particles, dtype=origin_path.dtype)
                selected_particle_unique_count = jnp.sum(
                    jnp.any(origin_path[:, None] == particle_ids[None, :], axis=0)
                ).astype(traj_dtype)
                step_info.update(
                    {
                        "selected_particle_per_t": origin_path.astype(jnp.float32),
                        "reference_path_hit_rate": reference_path_hit_rate.astype(jnp.float32),
                        "selected_particle_unique_count": (
                            selected_particle_unique_count.astype(jnp.float32)
                        ),
                    }
                )

        return (
            state._replace(
                position=next_position,
                latent_context=next_context,
                latent_trajectory=latent_path,
                trajectory_log_prob=next_traj_lp,
                complete_log_posterior=next_complete,
            ),
            step_info,
        )

    return MarginalParticleGibbsKernel(
        step_fn=_step_fn,
        num_particles=num_particles,
        num_parameter_particles=num_parameter_particles,
        initial_param_step_size=param_step_size,
        target_accept=target_accept,
        min_scale=min_scale,
        max_scale=max_scale,
        preconditioned=parameter_preconditioner_chol is not None,
        latent_smoother=latent_smoother_spec,
        latent_delta=latent_delta,
        amala_delta_init=amala_delta_init,
        amala_delta_min=amala_delta_min,
        amala_delta_max=amala_delta_max,
        amala_target_accept=amala_target_accept,
        amala_adaptation_window=amala_adaptation_window,
        amala_adaptation_tolerance=amala_adaptation_tolerance,
        amala_adaptation_rho=amala_adaptation_rho,
        amala_adaptation_rho_min=amala_adaptation_rho_min,
        amala_adaptation_gamma=amala_adaptation_gamma,
        adapt_amala_delta=_uses_amala_delta(latent_smoother_spec),
        amala_kappa=amala_kappa,
        amala_grad_clip=amala_grad_clip,
        dsmc_leaf_proposal=dsmc_leaf_proposal,
        latent_block_coords=latent_block_coords,
        diagnostic_metrics=resolved_diagnostic_metrics,
        exact_constraints=exact_constraints,
    )
