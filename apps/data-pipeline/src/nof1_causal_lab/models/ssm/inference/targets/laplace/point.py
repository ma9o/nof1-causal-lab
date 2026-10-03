"""Point-observation and linear-summary Laplace solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from nof1_causal_lab.models.ssm.covariance_utils import symmetrize
from nof1_causal_lab.models.ssm.execution.contracts import (
    LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
    LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
    ObservationLaws,
    build_likelihood_eval_aux,
)
from nof1_causal_lab.models.ssm.execution.observation_operator import (
    row_observation_log_prob as _row_observation_log_prob,
)
from nof1_causal_lab.models.ssm.execution.observation_operator import (
    trajectory_observation_log_prob,
)

from .shared import (
    GaussianTrajectoryPriorTerms,
    _build_ieks_system_from_prior,
    _build_prior_tridiagonal_system,
    _compute_profile_lower_bandwidths,
    _factor_block_profile_cholesky,
    _predictive_latent_init,
    _prepare_linearized_path,
    build_gaussian_trajectory_prior_terms,
    precision_logdet,
    trajectory_prior_log_prob_from_terms,
)
from .solvers import solve_latent_mode

if TYPE_CHECKING:
    from dynestyx import StochasticContinuousTimeStateEvolution

    from nof1_causal_lab.models.ssm.execution.observation_model import (
        EmissionLogProbFn,
        ObservationKernel,
    )


def _row_joint_log_prob(
    latent_trajectory: jnp.ndarray,
    *,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    cd: jnp.ndarray,
    prior_terms: GaussianTrajectoryPriorTerms,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel: ObservationKernel,
) -> jnp.ndarray:
    """Exact latent joint log-density for per-row observation operators."""
    return trajectory_prior_log_prob_from_terms(latent_trajectory, Ad, cd, prior_terms) + (
        _row_observation_log_prob(
            latent_trajectory,
            observations,
            obs_mask,
            H_rows,
            d_rows,
            R,
            obs_kernel,
        )
    )


def _point_profile_bandwidths(n_time: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return profile bandwidth vectors for a block-tridiagonal system."""
    row_upper = np.zeros((n_time,), dtype=np.int32)
    if n_time > 1:
        row_upper[:-1] = 1
    row_lower = _compute_profile_lower_bandwidths(row_upper.astype(np.int64)).astype(np.int32)
    return jnp.asarray(row_upper, dtype=jnp.int32), jnp.asarray(row_lower, dtype=jnp.int32)


def _point_linearize(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    obs_kernel: ObservationKernel,
    z_estimate: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-time emission gradients and negative Hessians."""
    obs_mask_float = obs_mask.astype(observations.dtype)
    grads_and_hess = jax.vmap(
        lambda y_t, z_t, mask_t, H_t, d_t: obs_kernel.latent_grad_hess_fn(
            y_t,
            z_t,
            H_t,
            d_t,
            R,
            mask_t,
        )
    )(
        observations,
        z_estimate,
        obs_mask_float,
        H_rows,
        d_rows,
    )
    return grads_and_hess[0], grads_and_hess[1]


def _point_posterior_system(
    z_est: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel: ObservationKernel,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the point-observation IEKS linear system at a latent iterate."""
    prior_lower, prior_diag, prior_upper, prior_rhs = _build_prior_tridiagonal_system(
        Ad,
        Qd,
        cd,
        init_mean,
        init_cov,
    )
    grads, J_t = _point_linearize(
        observations,
        obs_mask,
        H_rows,
        d_rows,
        R,
        obs_kernel,
        z_est,
    )
    tilde_y = jax.vmap(lambda J, z, g: J @ z + g)(J_t, z_est, grads)
    _lower, diag, upper, rhs = _build_ieks_system_from_prior(
        prior_lower,
        prior_diag,
        prior_upper,
        prior_rhs,
        J_t,
        tilde_y,
    )
    return diag, upper, rhs


def _point_laplace_terms_from_mode(
    z_mode: jnp.ndarray,
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel: ObservationKernel,
    *,
    factor_block_cholesky_fn=_factor_block_profile_cholesky,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate the point-observation Laplace terms at a fixed latent mode."""
    T, D = z_mode.shape
    prior_terms = build_gaussian_trajectory_prior_terms(
        Ad,
        Qd,
        cd,
        init_mean,
        init_cov,
    )
    mode_log_joint = _row_joint_log_prob(
        z_mode,
        observations=observations,
        obs_mask=obs_mask,
        Ad=Ad,
        cd=cd,
        prior_terms=prior_terms,
        H_rows=H_rows,
        d_rows=d_rows,
        R=R,
        obs_kernel=obs_kernel,
    )
    system_diag, system_upper, _system_rhs = _point_posterior_system(
        z_mode,
        observations,
        obs_mask,
        Ad,
        Qd,
        cd,
        H_rows,
        d_rows,
        R,
        init_mean,
        init_cov,
        obs_kernel,
    )
    row_upper_bandwidths, row_lower_bandwidths = _point_profile_bandwidths(T)
    chol_diag, _lower = factor_block_cholesky_fn(
        system_diag,
        system_upper[None, ...],
        row_upper_bandwidths,
        row_lower_bandwidths,
    )
    flat_dim = T * D
    laplace_logdet = precision_logdet(
        system_diag, system_upper[None], row_upper_bandwidths, row_lower_bandwidths
    )
    min_chol_diag = jnp.min(jnp.diagonal(chol_diag, axis1=1, axis2=2))
    log_lik = mode_log_joint + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi) - 0.5 * laplace_logdet
    return log_lik, mode_log_joint, laplace_logdet, min_chol_diag


def _point_ieks_laplace_core(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel: ObservationKernel,
    *,
    build_measurement_objects=None,
    observation_laws: ObservationLaws = (),
    solver_kind: int,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Sparse Newton mode and implicit gradients owned by Optimistix."""
    row_upper, row_lower = _point_profile_bandwidths(observations.shape[0])

    def objects(parameters):
        kernel = (
            obs_kernel
            if build_measurement_objects is None
            else build_measurement_objects(parameters[5], parameters[8]).kernel
        )
        return (*parameters[:8], kernel)

    def log_joint(z, parameters):
        A, Q, c, H, d, variance, mean, covariance, kernel = objects(parameters)
        prior = build_gaussian_trajectory_prior_terms(A, Q, c, mean, covariance)
        return _row_joint_log_prob(
            z,
            observations=observations,
            obs_mask=obs_mask,
            Ad=A,
            cd=c,
            prior_terms=prior,
            H_rows=H,
            d_rows=d,
            R=variance,
            obs_kernel=kernel,
        )

    def system(z, parameters):
        A, Q, c, H, d, variance, mean, covariance, kernel = objects(parameters)
        diag, upper, rhs = _point_posterior_system(
            z, observations, obs_mask, A, Q, c, H, d, variance, mean, covariance, kernel
        )
        return diag, upper[None], rhs

    parameters = (Ad, Qd, cd, H_rows, d_rows, R, init_mean, init_cov)
    if build_measurement_objects is not None:
        parameters = (*parameters, observation_laws)
    initial = (
        _predictive_latent_init(Ad, cd, init_mean)
        if z_init is None
        else jnp.asarray(z_init, dtype=observations.dtype)
    )
    z_mode, mode_aux = solve_latent_mode(
        log_joint,
        system,
        initial,
        parameters,
        bandwidth=1,
        row_upper=row_upper,
        row_lower=row_lower,
        max_steps=n_ieks_iters,
    )
    A, Q, c, loading, intercept, variance, mean, covariance, kernel = objects(parameters)
    log_lik, mode_log_joint, logdet, min_chol = _point_laplace_terms_from_mode(
        z_mode,
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
    )
    aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=solver_kind,
        final_log_joint=mode_log_joint,
        laplace_logdet=logdet,
        min_chol_diag=min_chol,
        **mode_aux,
    )
    return z_mode, log_lik, {**aux, "latent_mode": z_mode}


def _point_dynamic_transition_ieks_laplace(
    observations: jnp.ndarray,
    obs_mask: jnp.ndarray,
    dynamics: StochasticContinuousTimeStateEvolution,
    time_intervals: jnp.ndarray,
    H_rows: jnp.ndarray,
    d_rows: jnp.ndarray,
    R: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    obs_kernel: ObservationKernel,
    *,
    n_ieks_iters: int,
    z_init: jnp.ndarray | None = None,
    solver_kind: int = LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Optimistix fixed-point iteration over locally linearized initialization modes."""
    transitions_at, initial = _prepare_linearized_path(
        dynamics,
        time_intervals,
        init_mean,
        z_init=z_init,
        dtype=observations.dtype,
    )

    def update(path, _args):
        A, Q, c = transitions_at(path)
        mode, _likelihood, _aux = _point_ieks_laplace_core(
            observations,
            obs_mask,
            A,
            Q,
            c,
            H_rows,
            d_rows,
            R,
            init_mean,
            init_cov,
            obs_kernel,
            solver_kind=solver_kind,
            n_ieks_iters=1,
            z_init=path,
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
    log_lik, mode_log_joint, logdet, min_chol = _point_laplace_terms_from_mode(
        mode,
        observations,
        obs_mask,
        A,
        Q,
        c,
        H_rows,
        d_rows,
        R,
        init_mean,
        init_cov,
        obs_kernel,
    )
    A0, Q0, c0 = transitions_at(initial)
    prior0 = build_gaussian_trajectory_prior_terms(A0, Q0, c0, init_mean, init_cov)
    aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=solver_kind,
        n_iterations=solution.stats["num_steps"],
        init_log_joint=_row_joint_log_prob(
            initial,
            observations=observations,
            obs_mask=obs_mask,
            Ad=A0,
            cd=c0,
            prior_terms=prior0,
            H_rows=H_rows,
            d_rows=d_rows,
            R=R,
            obs_kernel=obs_kernel,
        ),
        final_log_joint=mode_log_joint,
        laplace_logdet=logdet,
        min_chol_diag=min_chol,
    )
    return mode, log_lik, {**aux, "latent_mode": mode}


def _ieks_smooth(
    observations,
    obs_mask,
    Ad,
    Qd,
    cd,
    H_rows,
    d_rows,
    R,
    init_mean,
    init_cov,
    obs_kernel: ObservationKernel,
    *,
    solver_kind: int = LIKELIHOOD_SOLVER_KIND_POINT_IEKS,
    n_ieks_iters=5,
    z_init: jnp.ndarray | None = None,
    build_measurement_objects=None,
    observation_laws: ObservationLaws = (),
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, jnp.ndarray]]:
    """Run the Iterated Extended Kalman Smoother to find the MAP state trajectory."""
    return _point_ieks_laplace_core(
        observations,
        obs_mask,
        Ad,
        Qd,
        cd,
        H_rows,
        d_rows,
        R,
        init_mean,
        init_cov,
        obs_kernel,
        build_measurement_objects=build_measurement_objects,
        observation_laws=observation_laws,
        solver_kind=solver_kind,
        n_ieks_iters=n_ieks_iters,
        z_init=z_init,
    )


def _dense_latent_mode(log_joint, initial, max_steps):
    """Whiten local curvature; Optimistix owns dense search and differentiation."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(jax.hessian(lambda y: -log_joint(y))(initial))
    scale = jax.lax.stop_gradient(eigenvectors / jnp.sqrt(jnp.maximum(eigenvalues, 1e-4))[None, :])
    solution = optx.minimise(
        lambda position, _args: -log_joint(initial + scale @ position),
        optx.BFGS(rtol=1e-5, atol=1e-5),
        jnp.zeros_like(initial),
        max_steps=max(max_steps, 1) + 1,
        throw=False,
    )
    return initial + scale @ solution.value, solution.stats["num_steps"] - 1


def _dense_support_laplace_log_lik(
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
    observation_support,
    n_newton_iters: int,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Dense Laplace approximation for interval-summary observation semantics."""
    T, D = observations.shape[0], init_mean.shape[0]
    flat_dim = T * D

    with jax.named_scope("map/dense_support_init"):
        z_init = _predictive_latent_init(Ad, cd, init_mean)
        prior_terms = build_gaussian_trajectory_prior_terms(
            Ad,
            Qd,
            cd,
            init_mean,
            init_cov,
        )

    def _joint_log_prob(z_flat: jnp.ndarray) -> jnp.ndarray:
        z = z_flat.reshape(T, D)
        prior_ll = trajectory_prior_log_prob_from_terms(z, Ad, cd, prior_terms)
        obs_ll = trajectory_observation_log_prob(
            z,
            observations,
            obs_mask,
            H,
            d,
            R,
            obs_kernel,
            mean_log_prob_fn,
            observation_support,
        )
        return prior_ll + obs_ll

    def _neg_log_prob(z_flat: jnp.ndarray) -> jnp.ndarray:
        return -_joint_log_prob(z_flat)

    z_flat = z_init.reshape(-1)
    init_log_joint = _joint_log_prob(z_flat)
    with jax.named_scope("map/dense_support_mode"):
        z_flat, n_steps = _dense_latent_mode(_joint_log_prob, z_flat, n_newton_iters)

    with jax.named_scope("map/dense_support_curvature"):
        mode_log_joint = _joint_log_prob(z_flat)
        hess = jax.hessian(_neg_log_prob)(z_flat)
        hess = symmetrize(hess)
        eigvals = jnp.linalg.eigvalsh(hess)
        min_eig = jnp.min(eigvals)
        logdet = jnp.sum(jnp.log(jnp.maximum(eigvals, 1e-6)))
    log_lik = mode_log_joint + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi) - 0.5 * logdet
    inner_eval_aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
        n_iterations=jnp.asarray(n_steps, dtype=jnp.int32),
        init_log_joint=init_log_joint,
        final_log_joint=mode_log_joint,
        final_rel_change=(
            jnp.linalg.norm(z_flat - z_init.reshape(-1))
            / (1.0 + jnp.linalg.norm(z_init.reshape(-1)))
        ),
        laplace_logdet=logdet,
        min_chol_diag=jnp.sqrt(jnp.maximum(min_eig, 0.0)),
    )
    inner_eval_aux["latent_mode"] = z_flat.reshape(T, D)
    return log_lik, inner_eval_aux


def _dense_dynamic_support_laplace_log_lik(
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
    observation_support,
    n_newton_iters: int,
    *,
    z_init: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Dense interval-support Laplace path with local dynamics linearization."""
    T, D = observations.shape[0], init_mean.shape[0]
    flat_dim = T * D

    def _joint_log_prob_fixed(
        z_flat_eval: jnp.ndarray,
        Ad: jnp.ndarray,
        Qd: jnp.ndarray,
        cd: jnp.ndarray,
    ) -> jnp.ndarray:
        z = z_flat_eval.reshape(T, D)
        prior_terms = build_gaussian_trajectory_prior_terms(
            Ad,
            Qd,
            cd,
            init_mean,
            init_cov,
        )
        prior_ll = trajectory_prior_log_prob_from_terms(z, Ad, cd, prior_terms)
        obs_ll = trajectory_observation_log_prob(
            z,
            observations,
            obs_mask,
            H,
            d,
            R,
            obs_kernel,
            mean_log_prob_fn,
            observation_support,
        )
        return prior_ll + obs_ll

    with jax.named_scope("map/dense_dynamic_support_init"):
        _transitions_at, initial_path = _prepare_linearized_path(
            dynamics,
            time_intervals,
            init_mean,
            z_init=z_init,
            dtype=observations.dtype,
        )
        z_flat = initial_path.reshape(-1)
        Ad_curr, Qd_curr, cd_curr = _transitions_at(initial_path)
        init_log_joint = _joint_log_prob_fixed(z_flat, Ad_curr, Qd_curr, cd_curr)

    def update(path, _args):
        A, Q, c = _transitions_at(path)
        mode, _steps = _dense_latent_mode(
            lambda flat: _joint_log_prob_fixed(flat, A, Q, c), path.reshape(-1), 1
        )
        return mode.reshape(T, D)

    with jax.named_scope("map/dense_dynamic_support_mode"):
        solution = optx.fixed_point(
            update,
            optx.FixedPointIteration(rtol=1e-3, atol=1e-3),
            initial_path,
            max_steps=max(n_newton_iters, 1),
            throw=False,
        )
        z_flat = jnp.asarray(solution.value).reshape(-1)

    with jax.named_scope("map/dense_dynamic_support_curvature"):
        Ad_final, Qd_final, cd_final = _transitions_at(z_flat.reshape(T, D))

        def _final_neg_log_prob_fixed(z_flat_eval: jnp.ndarray) -> jnp.ndarray:
            return -_joint_log_prob_fixed(z_flat_eval, Ad_final, Qd_final, cd_final)

        mode_log_joint = _joint_log_prob_fixed(z_flat, Ad_final, Qd_final, cd_final)
        hess = jax.hessian(_final_neg_log_prob_fixed)(z_flat)
        hess = symmetrize(hess)
        eigvals = jnp.linalg.eigvalsh(hess)
        min_eig = jnp.min(eigvals)
        logdet = jnp.sum(jnp.log(jnp.maximum(eigvals, 1e-6)))

    log_lik = mode_log_joint + 0.5 * flat_dim * jnp.log(2.0 * jnp.pi) - 0.5 * logdet
    inner_eval_aux = build_likelihood_eval_aux(
        observations.dtype,
        solver_kind=LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT,
        n_iterations=solution.stats["num_steps"],
        init_log_joint=init_log_joint,
        final_log_joint=mode_log_joint,
        laplace_logdet=logdet,
        min_chol_diag=jnp.sqrt(jnp.maximum(min_eig, 0.0)),
    )
    inner_eval_aux["latent_mode"] = z_flat.reshape(T, D)
    return log_lik, inner_eval_aux
