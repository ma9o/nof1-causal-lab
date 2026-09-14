"""MAP orchestration on a two-coordinate quadratic, without building or fitting an SSM."""

from types import SimpleNamespace
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest
from dynestyx.inference.particle_runtime import Parameterization

from nof1_causal_lab.models.ssm.inference.types import WarmupProposal
from nof1_causal_lab.models.ssm.inference.warmup import map as map_warmup

_INNER_DIAGNOSTICS = {
    "solver_kind": 1,
    "n_iterations": 2,
    "n_accepted_steps": 2,
    "init_log_joint": -3.0,
    "final_log_joint": -1.0,
    "final_rel_change": 1e-3,
    "final_damping": 1e-4,
    "final_step_alpha": 1.0,
    "final_step_norm": 0.1,
    "laplace_logdet": 2.0,
    "min_chol_diag": 0.5,
}


def _log_posterior(z, observations, times, latent_mode_init=None):
    del observations, times, latent_mode_init
    return -jnp.sum((z - jnp.array([1.0, -2.0])) ** 2) - 0.1 * jnp.sum(z**2)


def _negative_log_posterior(z, observations, times, latent_mode_init=None):
    return -_log_posterior(z, observations, times, latent_mode_init)


def _objective_with_aux(z, observations, times, latent_mode_init=None):
    log_post = _log_posterior(z, observations, times, latent_mode_init)
    log_prior = -0.1 * jnp.sum(z**2)
    return -log_post, {
        "log_posterior": log_post,
        "log_likelihood": log_post - log_prior,
        "log_prior": log_prior,
        "inner": _INNER_DIAGNOSTICS,
        "latent_mode": z[None, :],
    }


@pytest.mark.parametrize("interval_support", [False, True], ids=["point", "interval"])
def test_optimizer_initialization_and_exact_gradient_contract(monkeypatch, interval_support):
    flat_example = jnp.array([0.25, -0.5])
    model = SimpleNamespace(
        observation_support=SimpleNamespace(requires_interval_summary_handling=interval_support)
    )
    candidates = jnp.array([[0.0, 0.0], [4.0, 4.0], [1.0, -2.0]])
    draw_candidates = Mock(return_value=(jnp.array([0, 1], dtype=jnp.uint32), candidates))
    monkeypatch.setattr(map_warmup, "_draw_laplace_init_candidates", draw_candidates)
    expected_start = np.asarray(flat_example if interval_support else candidates[2])
    optimum = np.array([1.0, -2.0]) / 1.1

    def minimize(fun, x0, jac, method, tol, options, callback):
        assert method == "L-BFGS-B"
        assert tol == 1e-3
        assert options["maxiter"] == 9
        np.testing.assert_allclose(x0, expected_start)
        np.testing.assert_allclose(jac(x0), 2.2 * x0 - np.array([2.0, -4.0]), atol=1e-6)
        assert fun(optimum) < fun(x0)
        np.testing.assert_allclose(jac(optimum), 0.0, atol=1e-6)
        callback(optimum)
        return SimpleNamespace(x=optimum, fun=fun(optimum), nit=3, nfev=5, status=0, success=True)

    optimizer = Mock(side_effect=minimize)
    monkeypatch.setattr(map_warmup.spo, "minimize", optimizer)
    result = map_warmup._optimize_laplace_parameter_mode(
        model,
        init_key=jnp.array([0, 0], dtype=jnp.uint32),
        dim=2,
        flat_example=flat_example,
        site_info={},
        runtime_log_posterior_fn=_log_posterior,
        runtime_neg_log_posterior_with_aux_fn=_objective_with_aux,
        observations=jnp.array([[0.0], [1.0]]),
        times=jnp.array([0.0, 1.0]),
        n_init_samples=2,
        maxiter=9,
        tol=1e-3,
    )

    optimizer.assert_called_once()
    assert draw_candidates.call_count == (0 if interval_support else 1)
    np.testing.assert_allclose(result.z_mode, optimum)
    assert result.success
    assert result.n_function_evals == 5
    assert result.final_grad_norm == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("strategy", ["mode_only", "exact_hessian", "optimizer_hess_inv"])
def test_covariance_routes_and_public_draw_extraction(monkeypatch, strategy):
    mode = jnp.array([0.5, -1.0])
    inverse_hessian = np.array([[2.0, 0.5], [0.5, 1.0]])
    mode_result = map_warmup.LaplaceModeOptimizationResult(
        z_mode=mode,
        objective_at_mode=1.0,
        n_iters=3,
        n_function_evals=5,
        status=0,
        success=True,
        optimizer="L-BFGS-B",
        init_log_posterior_best=-2.0,
        optimizer_hess_inv=Mock(todense=Mock(return_value=inverse_hessian)),
        final_grad_norm=0.0,
        final_eval_diagnostics={
            "log_posterior": -1.0,
            "log_likelihood": -0.8,
            "log_prior": -0.2,
            "inner": _INNER_DIAGNOSTICS,
        },
    )
    parameters = Parameterization(
        initial_position=mode,
        unravel=lambda z: {"theta": z},
        constrain=lambda z: {"theta": 7.0 + 2.0 * z, "theta_decentered": z},
        log_prior=lambda z: -jnp.sum(z**2),
    )
    bundle = {
        "dim": 2,
        "flat_example": mode,
        "site_info": {"theta": {}},
        "parameters": parameters,
        "public_sites": {"theta"},
        "log_posterior_fn": _log_posterior,
        "neg_log_posterior_fn": _negative_log_posterior,
        "neg_log_posterior_with_aux_fn": _objective_with_aux,
    }
    monkeypatch.setattr(map_warmup, "get_laplace_backend", Mock(return_value=object()))
    monkeypatch.setattr(map_warmup, "_build_map_laplace_bundle", Mock(return_value=bundle))
    monkeypatch.setattr(
        map_warmup, "_optimize_laplace_parameter_mode", Mock(return_value=mode_result)
    )
    hessian = Mock(return_value=jnp.array([[4.0, 1.0], [1.0, 2.0]]))
    monkeypatch.setattr(map_warmup, "_laplace_parameter_hessian_runtime", hessian)
    # Fixed symmetric draws expose covariance orientation and public-coordinate replay.
    noise = jnp.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    def sample(_key, z_mode, chol_cov, *, num_samples):
        assert num_samples == 4
        return z_mode + noise @ chol_cov.T

    sampler = Mock(side_effect=sample)
    monkeypatch.setattr(map_warmup, "_sample_gaussian_parameter_posterior", sampler)
    result = map_warmup.fit_map(
        SimpleNamespace(observation_support=None),
        jnp.zeros((2, 1)),
        jnp.array([0.0, 1.0]),
        num_samples=4,
        compute_parameter_hessian=strategy != "mode_only",
        parameter_covariance_method="exact_hessian"
        if strategy == "exact_hessian"
        else "optimizer_hess_inv",
        hessian_jitter=0.0,
    )

    if strategy == "mode_only":
        covariance = np.zeros((2, 2))
        draws = np.broadcast_to(mode, (4, 2))
    else:
        covariance = (
            np.array([[2.0, -1.0], [-1.0, 4.0]]) / 7
            if strategy == "exact_hessian"
            else inverse_hessian
        )
        draws = np.asarray(mode) + np.asarray(noise) @ np.linalg.cholesky(covariance).T
    assert isinstance(result, WarmupProposal)
    assert result.diagnostics["parameter_covariance_method"] == strategy
    assert result.diagnostics["compute_parameter_hessian"] == (strategy != "mode_only")
    np.testing.assert_allclose(result.diagnostics["parameter_covariance"], covariance, atol=1e-6)
    assert set(result.get_samples()) == {"theta"}
    np.testing.assert_allclose(result.get_samples()["theta"], 7.0 + 2.0 * draws, atol=1e-6)
    assert hessian.call_count == (1 if strategy == "exact_hessian" else 0)
    assert sampler.call_count == (0 if strategy == "mode_only" else 1)
