"""Support-aware Laplace solvers for interval-summary observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optimistix as optx

from nof1_causal_lab.models.ssm.execution.contracts import (
    LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
    ObservationLaws,
    build_likelihood_eval_aux,
)
from nof1_causal_lab.models.ssm.execution.observation_operator import (
    accumulate_support_statistics,
    expected_observation_mean,
    get_point_like_mask,
    get_support_kind_codes,
    trajectory_observation_log_prob,
)

from .shared import (
    GaussianTrajectoryPriorTerms,
    SupportObservationWindowBatch,
    _build_prior_banded_system,
    _factor_block_banded_cholesky,
    _factor_block_profile_cholesky,
    _predictive_latent_init,
    _prepare_linearized_path,
    _solve_block_banded_from_cholesky,
    _solve_block_profile_from_cholesky,
    build_gaussian_trajectory_prior_terms,
    precision_logdet,
    trajectory_prior_log_prob_from_terms,
)
from .solvers import solve_latent_mode

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx import StochasticContinuousTimeStateEvolution

    from nof1_causal_lab.models.ssm.execution.observation_model import (
        CompiledObservationModel,
        EmissionLogProbFn,
        ObservationKernel,
    )
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

type SupportWindowDerivatives = Callable[..., tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]


def _assemble_support_aware_observation_system(
    z_est: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel: ObservationKernel,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    point_like_mask: jnp.ndarray,
    window_derivatives: tuple[SupportWindowDerivatives, ...],
    bandwidth: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble exact Newton observation terms in block-banded form."""
    T, D = z_est.shape
    diag = jnp.zeros((T, D, D), dtype=z_est.dtype)
    upper = jnp.zeros((bandwidth, T, D, D), dtype=z_est.dtype)
    rhs = jnp.zeros((T, D), dtype=z_est.dtype)

    clean_obs = jnp.nan_to_num(observations, nan=0.0)
    point_mask = obs_mask.astype(z_est.dtype) * point_like_mask[None, :]

    local_grads, local_hess = jax.vmap(
        lambda y_t, z_t, mask_t: obs_kernel.latent_grad_hess_fn(y_t, z_t, H, d, R, mask_t)
    )(clean_obs, z_est, point_mask)
    diag = diag + local_hess
    point_rhs: jnp.ndarray = jax.vmap(lambda j_t, z_t, g_t: j_t @ z_t + g_t)(
        local_hess, z_est, local_grads
    )
    rhs = rhs + point_rhs

    if len(support_window_batches) == 0:
        return diag, upper, rhs

    max_support_state_len = max(batch.max_state_len for batch in support_window_batches)
    padded_z = jnp.pad(z_est, ((0, max_support_state_len - 1), (0, 0)))

    for support_windows, batch_window_derivatives in zip(
        support_window_batches,
        window_derivatives,
        strict=True,
    ):
        segment_states = padded_z[support_windows.padded_state_indices]
        segment_flat = segment_states.reshape(segment_states.shape[0], -1)
        anchor_obs = clean_obs[support_windows.anchor_indices]
        grad_blocks, jac_blocks, mean_info = batch_window_derivatives(
            segment_flat,
            support_windows.state_lens,
            support_windows.mask_full.astype(z_est.dtype),
            support_windows.prev_coeffs.astype(z_est.dtype),
            support_windows.curr_coeffs.astype(z_est.dtype),
            support_windows.weights.astype(z_est.dtype),
            anchor_obs,
            H,
            d,
            R,
        )
        diag_updates = jnp.einsum("gmid,gmn,gnie->gide", jac_blocks, mean_info, jac_blocks)
        taylor_rhs = grad_blocks + jnp.einsum("gide,gie->gid", diag_updates, segment_states)

        diag = diag.at[support_windows.time_indices.reshape(-1)].add(
            (
                diag_updates * support_windows.valid_diag.astype(z_est.dtype)[..., None, None]
            ).reshape(-1, D, D)
        )

        batch_bandwidth = support_windows.cross_time_indices.shape[0]
        for offset in range(1, batch_bandwidth + 1):
            cross_len = support_windows.max_state_len - offset
            left_jac = jac_blocks[:, :, :cross_len, :]
            right_jac = jac_blocks[:, :, offset:, :]
            cross_updates = jnp.einsum("gmid,gmn,gnie->gide", left_jac, mean_info, right_jac)

            left_states = segment_states[:, :cross_len, :]
            right_states = segment_states[:, offset:, :]
            taylor_rhs = taylor_rhs.at[:, :cross_len, :].add(
                jnp.einsum("gide,gie->gid", cross_updates, right_states)
            )
            taylor_rhs = taylor_rhs.at[:, offset:, :].add(
                jnp.einsum("gide,gid->gie", cross_updates, left_states)
            )

            valid_cross = support_windows.valid_cross[offset - 1, :, :cross_len]
            upper_times = support_windows.cross_time_indices[offset - 1, :, :cross_len]
            upper = upper.at[offset - 1, upper_times.reshape(-1)].add(
                (cross_updates * valid_cross.astype(z_est.dtype)[..., None, None]).reshape(-1, D, D)
            )

        rhs = rhs.at[support_windows.time_indices.reshape(-1)].add(
            (taylor_rhs * support_windows.valid_diag.astype(z_est.dtype)[..., None]).reshape(
                -1,
                D,
            )
        )

    return diag, upper, rhs


def _make_support_window_derivatives(
    *,
    max_state_len: int,
    n_latent: int,
    n_manifest: int,
    summary_operator_codes: jnp.ndarray,
    obs_kernel: ObservationKernel,
    mean_log_prob_fn: EmissionLogProbFn | None,
) -> SupportWindowDerivatives:
    """Build support-window derivatives with Gauss-Newton curvature in mean space."""
    assert mean_log_prob_fn is not None

    def _window_expected_mean_single(
        segment_flat_single: jnp.ndarray,
        state_len_single: jnp.ndarray,
        mask_full_single: jnp.ndarray,
        prev_coeffs_single: jnp.ndarray,
        curr_coeffs_single: jnp.ndarray,
        weights_single: jnp.ndarray,
        anchor_obs_single: jnp.ndarray,
        H: jnp.ndarray,
        d: jnp.ndarray,
        R: jnp.ndarray,
    ) -> jnp.ndarray:
        states = segment_flat_single.reshape(max_state_len, n_latent)
        responses = jax.vmap(lambda z_t: obs_kernel.response_fn(H @ z_t + d))(states)
        last_response = jax.lax.dynamic_index_in_dim(
            responses,
            jnp.maximum(state_len_single - 1, 0),
            axis=0,
            keepdims=False,
        )

        def _single_step_window(_):
            return last_response, last_response**2, mask_full_single

        def _multi_step_window(_):
            zeros = jnp.zeros((n_manifest, 1), dtype=responses.dtype)

            def _scan_step(carry, inputs):
                response_prev, accum_sum, accum_sumsq, accum_weight = carry
                response_t, prev_coeff_t, curr_coeff_t, weight_t = inputs
                obs_sum, obs_sumsq, obs_weight = accumulate_support_statistics(
                    response_prev,
                    accum_sum,
                    accum_sumsq,
                    accum_weight,
                    response_t,
                    prev_coeff_t,
                    curr_coeff_t,
                    weight_t,
                )
                return (response_t, obs_sum, obs_sumsq, obs_weight), None

            final_carry, _ = jax.lax.scan(
                _scan_step,
                (responses[0], zeros, zeros, zeros),
                (
                    responses[1:],
                    prev_coeffs_single[..., None],
                    curr_coeffs_single[..., None],
                    weights_single[..., None],
                ),
            )
            _response_last, obs_sum, obs_sumsq, obs_weight = final_carry
            return obs_sum.squeeze(-1), obs_sumsq.squeeze(-1), obs_weight.squeeze(-1)

        obs_sum, obs_sumsq, obs_weight = jax.lax.cond(
            state_len_single == 1,
            _single_step_window,
            _multi_step_window,
            operand=None,
        )

        expected_mean = expected_observation_mean(
            last_response,
            obs_sum,
            obs_sumsq,
            obs_weight,
            summary_operator_codes,
        )
        del anchor_obs_single, R
        return expected_mean

    def _window_mean_log_prob_single(
        expected_mean_single: jnp.ndarray,
        anchor_obs_single: jnp.ndarray,
        mask_full_single: jnp.ndarray,
        R: jnp.ndarray,
    ) -> jnp.ndarray:
        return mean_log_prob_fn(anchor_obs_single, expected_mean_single, R, mask_full_single)

    window_expected_mean_jacobian: Callable[..., jnp.ndarray] = jax.jacrev(
        _window_expected_mean_single
    )
    mean_log_prob_grad: Callable[..., jnp.ndarray] = jax.grad(_window_mean_log_prob_single)
    mean_log_prob_hessian: Callable[..., jnp.ndarray] = jax.hessian(_window_mean_log_prob_single)

    def _batched_window_derivatives(
        segment_flat: jnp.ndarray,
        state_lens: jnp.ndarray,
        mask_full: jnp.ndarray,
        prev_coeffs: jnp.ndarray,
        curr_coeffs: jnp.ndarray,
        weights: jnp.ndarray,
        anchor_obs: jnp.ndarray,
        H: jnp.ndarray,
        d: jnp.ndarray,
        R: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        in_axes = (0, 0, 0, 0, 0, 0, 0, None, None, None)
        expected_mean = jax.vmap(_window_expected_mean_single, in_axes=in_axes)(
            segment_flat,
            state_lens,
            mask_full,
            prev_coeffs,
            curr_coeffs,
            weights,
            anchor_obs,
            H,
            d,
            R,
        )
        mean_grad = jax.vmap(mean_log_prob_grad, in_axes=(0, 0, 0, None))(
            expected_mean,
            anchor_obs,
            mask_full,
            R,
        )
        mean_hessian = jax.vmap(mean_log_prob_hessian, in_axes=(0, 0, 0, None))(
            expected_mean,
            anchor_obs,
            mask_full,
            R,
        )
        mean_info = -0.5 * (mean_hessian + jnp.swapaxes(mean_hessian, -1, -2))
        jac_flat = jax.vmap(window_expected_mean_jacobian, in_axes=in_axes)(
            segment_flat,
            state_lens,
            mask_full,
            prev_coeffs,
            curr_coeffs,
            weights,
            anchor_obs,
            H,
            d,
            R,
        )
        jac_blocks = jac_flat.reshape(-1, n_manifest, max_state_len, n_latent)
        grad_blocks = jnp.einsum("gmid,gm->gid", jac_blocks, mean_grad)
        return grad_blocks, jac_blocks, mean_info

    return _batched_window_derivatives


def _support_aware_joint_log_prob(
    z_est: jnp.ndarray,
    *,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    cd: jnp.ndarray,
    prior_terms: GaussianTrajectoryPriorTerms,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel: ObservationKernel,
    mean_log_prob_fn: EmissionLogProbFn | None,
    observation_support: ObservationSupportRuntime,
) -> jnp.ndarray:
    """Exact latent joint log-density used for support-aware step acceptance."""
    return trajectory_prior_log_prob_from_terms(z_est, Ad, cd, prior_terms) + (
        trajectory_observation_log_prob(
            z_est,
            observations,
            obs_mask,
            H,
            d,
            R,
            obs_kernel,
            mean_log_prob_fn,
            observation_support,
        )
    )


def _support_aware_posterior_system(
    z_est: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel: ObservationKernel,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    point_like_mask: jnp.ndarray,
    window_derivatives: tuple[SupportWindowDerivatives, ...],
    bandwidth: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble the exact support-aware posterior Newton system at `z_est`."""
    prior_diag, prior_upper, prior_rhs = _build_prior_banded_system(
        Ad,
        Qd,
        cd,
        init_mean,
        init_cov,
        bandwidth,
    )
    obs_diag, obs_upper, obs_rhs = _assemble_support_aware_observation_system(
        z_est,
        observations,
        obs_mask,
        H,
        d,
        R,
        obs_kernel,
        support_window_batches,
        point_like_mask,
        window_derivatives,
        bandwidth,
    )
    return prior_diag + obs_diag, prior_upper + obs_upper, prior_rhs + obs_rhs


def _support_latent_mode(
    observations,
    obs_mask,
    initial,
    parameters,
    objects,
    observation_support,
    support_window_batches,
    bandwidth,
    row_upper,
    row_lower,
    n_iters,
    *,
    factor=_factor_block_profile_cholesky,
    solve=_solve_block_profile_from_cholesky,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    point_mask = get_point_like_mask(
        get_support_kind_codes(observation_support), observations.dtype
    )

    def log_joint(z, args):
        A, Q, c, H, d, R, mean, covariance, kernel, mean_density, _derivatives = objects(args)
        prior = build_gaussian_trajectory_prior_terms(A, Q, c, mean, covariance)
        return _support_aware_joint_log_prob(
            z,
            observations=observations,
            obs_mask=obs_mask,
            Ad=A,
            cd=c,
            prior_terms=prior,
            H=H,
            d=d,
            R=R,
            obs_kernel=kernel,
            mean_log_prob_fn=mean_density,
            observation_support=observation_support,
        )

    def system(z, args):
        A, Q, c, H, d, R, mean, covariance, kernel, _mean_density, derivatives = objects(args)
        return _support_aware_posterior_system(
            z,
            observations,
            obs_mask,
            A,
            Q,
            c,
            H,
            d,
            R,
            mean,
            covariance,
            kernel,
            support_window_batches,
            point_mask,
            derivatives,
            bandwidth,
        )

    return solve_latent_mode(
        log_joint,
        system,
        initial,
        parameters,
        bandwidth=bandwidth,
        row_upper=row_upper,
        row_lower=row_lower,
        max_steps=n_iters,
        factor=factor,
        solve=solve,
    )


def _support_aware_laplace_terms_from_mode(
    z_mode: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel: ObservationKernel,
    mean_log_prob_fn: EmissionLogProbFn | None,
    observation_support: ObservationSupportRuntime,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    point_like_mask: jnp.ndarray,
    window_derivatives: tuple[SupportWindowDerivatives, ...],
    bandwidth: int,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    factor_block_cholesky_fn=_factor_block_profile_cholesky,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate the Laplace log-likelihood terms at a fixed latent mode."""
    prior_terms = build_gaussian_trajectory_prior_terms(
        Ad,
        Qd,
        cd,
        init_mean,
        init_cov,
    )
    mode_log_joint = _support_aware_joint_log_prob(
        z_mode,
        observations=observations,
        obs_mask=obs_mask,
        Ad=Ad,
        cd=cd,
        prior_terms=prior_terms,
        H=H,
        d=d,
        R=R,
        obs_kernel=obs_kernel,
        mean_log_prob_fn=mean_log_prob_fn,
        observation_support=observation_support,
    )
    system_diag, system_upper, _system_rhs = _support_aware_posterior_system(
        z_mode,
        observations,
        obs_mask,
        Ad,
        Qd,
        cd,
        H,
        d,
        R,
        init_mean,
        init_cov,
        obs_kernel,
        support_window_batches,
        point_like_mask,
        window_derivatives,
        bandwidth,
    )
    with jax.named_scope("map/support_aware_final_hessian"):
        chol_diag, _lower = factor_block_cholesky_fn(
            system_diag,
            system_upper,
            row_upper_bandwidths,
            row_lower_bandwidths,
        )

    flat_dim = observations.shape[0] * init_mean.shape[0]
    laplace_logdet = precision_logdet(
        system_diag, system_upper, row_upper_bandwidths, row_lower_bandwidths
    )
    min_chol_diag = jnp.min(jnp.diagonal(chol_diag, axis1=1, axis2=2))
    log_lik = mode_log_joint + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi) - 0.5 * laplace_logdet
    return log_lik, mode_log_joint, laplace_logdet, min_chol_diag


def _support_dynamic_transition_ieks_laplace(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    dynamics: StochasticContinuousTimeStateEvolution,
    time_intervals: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel: ObservationKernel,
    mean_log_prob_fn: EmissionLogProbFn | None,
    observation_support: ObservationSupportRuntime,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    bandwidth: int,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    window_derivatives: tuple[SupportWindowDerivatives, ...],
    n_ieks_iters: int,
    *,
    z_init: jnp.ndarray | None = None,
    factor_block_cholesky_fn=_factor_block_banded_cholesky,
    solve_block_from_cholesky_fn=_solve_block_banded_from_cholesky,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Optimistix fixed points over support-aware local initialization modes."""
    point_mask = get_point_like_mask(
        get_support_kind_codes(observation_support), observations.dtype
    )
    transitions_at, initial = _prepare_linearized_path(
        dynamics,
        time_intervals,
        init_mean,
        z_init=z_init,
        dtype=observations.dtype,
    )

    def objects(parameters):
        return (*parameters, obs_kernel, mean_log_prob_fn, window_derivatives)

    def update(path, _args):
        A, Q, c = transitions_at(path)
        mode, _aux = _support_latent_mode(
            observations,
            obs_mask,
            path,
            (A, Q, c, H, d, R, init_mean, init_cov),
            objects,
            observation_support,
            support_window_batches,
            bandwidth,
            row_upper_bandwidths,
            row_lower_bandwidths,
            1,
            factor=factor_block_cholesky_fn,
            solve=solve_block_from_cholesky_fn,
        )
        return mode

    solution = optx.fixed_point(
        update,
        optx.FixedPointIteration(rtol=1e-3, atol=1e-3),
        initial,
        max_steps=max(n_ieks_iters, 1),
        throw=False,
    )
    mode = jax.lax.stop_gradient(jnp.asarray(solution.value))
    A, Q, c = transitions_at(mode)
    log_lik, mode_log_joint, logdet, min_chol = _support_aware_laplace_terms_from_mode(
        mode,
        observations,
        obs_mask,
        A,
        Q,
        c,
        H,
        d,
        R,
        init_mean,
        init_cov,
        obs_kernel,
        mean_log_prob_fn,
        observation_support,
        support_window_batches,
        point_mask,
        window_derivatives,
        bandwidth,
        row_upper_bandwidths,
        row_lower_bandwidths,
        factor_block_cholesky_fn=factor_block_cholesky_fn,
    )
    A0, Q0, c0 = transitions_at(initial)
    prior0 = build_gaussian_trajectory_prior_terms(A0, Q0, c0, init_mean, init_cov)
    aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
        n_iterations=solution.stats["num_steps"],
        init_log_joint=_support_aware_joint_log_prob(
            initial,
            observations=observations,
            obs_mask=obs_mask,
            Ad=A0,
            cd=c0,
            prior_terms=prior0,
            H=H,
            d=d,
            R=R,
            obs_kernel=obs_kernel,
            mean_log_prob_fn=mean_log_prob_fn,
            observation_support=observation_support,
        ),
        final_log_joint=mode_log_joint,
        laplace_logdet=logdet,
        min_chol_diag=min_chol,
    )
    return log_lik, mode, {**aux, "latent_mode": mode}


def _support_aware_ieks_laplace_core(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel: ObservationKernel,
    mean_log_prob_fn: EmissionLogProbFn | None,
    observation_support: ObservationSupportRuntime,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    bandwidth: int,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    window_derivatives: tuple[SupportWindowDerivatives, ...],
    build_measurement_objects: Callable[
        [jnp.ndarray, ObservationLaws],
        tuple[CompiledObservationModel, tuple[SupportWindowDerivatives, ...]],
    ],
    observation_laws: ObservationLaws,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
    final_factor_block_cholesky_fn=_factor_block_profile_cholesky,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Sparse support-aware Newton mode and implicit gradients owned by Optimistix."""
    del obs_kernel, mean_log_prob_fn, window_derivatives
    point_mask = get_point_like_mask(
        get_support_kind_codes(observation_support), observations.dtype
    )

    def objects(parameters):
        measurement, derivatives = build_measurement_objects(parameters[5], parameters[8])
        return (*parameters[:8], measurement.kernel, measurement.mean_log_prob_fn, derivatives)

    parameters = (Ad, Qd, cd, H, d, R, init_mean, init_cov, observation_laws)
    initial = _predictive_latent_init(Ad, cd, init_mean) if z_init is None else jnp.asarray(z_init)
    initial = initial.astype(jnp.result_type(observations, Ad, H, R, init_mean, init_cov))
    mode, mode_aux = _support_latent_mode(
        observations,
        obs_mask,
        initial,
        parameters,
        objects,
        observation_support,
        support_window_batches,
        bandwidth,
        row_upper_bandwidths,
        row_lower_bandwidths,
        n_ieks_iters,
    )
    A, Q, c, loading, intercept, variance, mean, covariance, kernel, mean_density, derivatives = (
        objects(parameters)
    )
    log_lik, mode_log_joint, logdet, min_chol = _support_aware_laplace_terms_from_mode(
        mode,
        observations,
        obs_mask,
        A,
        Q,
        c,
        loading,
        intercept,
        variance,
        mean,
        covariance,
        kernel,
        mean_density,
        observation_support,
        support_window_batches,
        point_mask,
        derivatives,
        bandwidth,
        row_upper_bandwidths,
        row_lower_bandwidths,
        factor_block_cholesky_fn=final_factor_block_cholesky_fn,
    )
    aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS,
        final_log_joint=mode_log_joint,
        laplace_logdet=logdet,
        min_chol_diag=min_chol,
        **mode_aux,
    )
    return log_lik, mode, {**aux, "latent_mode": mode}


def _support_aware_ieks_laplace(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H: jnp.ndarray,
    d: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel: ObservationKernel,
    mean_log_prob_fn: EmissionLogProbFn | None,
    observation_support: ObservationSupportRuntime,
    support_window_batches: tuple[SupportObservationWindowBatch, ...],
    bandwidth: int,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    window_derivatives: tuple[SupportWindowDerivatives, ...],
    build_measurement_objects: Callable[
        [jnp.ndarray, ObservationLaws],
        tuple[CompiledObservationModel, tuple[SupportWindowDerivatives, ...]],
    ],
    observation_laws: ObservationLaws,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Support-aware IEKS solve plus Laplace log-likelihood."""
    return _support_aware_ieks_laplace_core(
        observations,
        obs_mask,
        Ad,
        Qd,
        cd,
        H,
        d,
        R,
        init_mean,
        init_cov,
        obs_kernel,
        mean_log_prob_fn,
        observation_support,
        support_window_batches,
        bandwidth,
        row_upper_bandwidths,
        row_lower_bandwidths,
        window_derivatives,
        build_measurement_objects,
        observation_laws,
        n_ieks_iters,
        z_init=z_init,
        final_factor_block_cholesky_fn=_factor_block_profile_cholesky,
    )
