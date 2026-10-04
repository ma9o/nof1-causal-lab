"""Initialization layouts checked against a full Gaussian residual operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import (
    _build_prior_banded_system,
    _build_prior_tridiagonal_system,
)
from nof1_causal_lab.models.ssm.inference.targets.laplace.solvers import solve_fixed_point_mode

pytestmark = pytest.mark.inference(concern="warmup")


@pytest.mark.parametrize("n_time", [1, 4])
@pytest.mark.parametrize("bandwidth", [0, 1, 3])
def test_prior_layouts_match_joint_gaussian_precision(n_time: int, bandwidth: int) -> None:
    n_latent = 2
    jitter = 1e-3
    transitions = jnp.broadcast_to(
        jnp.array([[0.9, 0.1], [-0.1, 0.8]], dtype=jnp.float16), (n_time, 2, 2)
    )
    covariances = jnp.broadcast_to(
        jnp.array([[0.3, 0.04], [0.04, 0.2]], dtype=jnp.float32), (n_time, 2, 2)
    )
    offsets = jnp.broadcast_to(jnp.array([0.1, -0.2], dtype=jnp.float32), (n_time, 2))
    initial_mean = jnp.array([0.2, -0.1], dtype=jnp.float32)
    initial_cov = jnp.array([[0.6, 0.1], [0.1, 0.4]], dtype=jnp.float32)

    a, q, c = (np.asarray(value, dtype=np.float64) for value in (transitions, covariances, offsets))
    residuals = np.eye(n_time * n_latent)
    covariance = np.zeros_like(residuals)
    means = c.copy()
    means[0] = a[0] @ np.asarray(initial_mean) + c[0]
    for t in range(n_time):
        row = slice(t * n_latent, (t + 1) * n_latent)
        covariance[row, row] = (
            a[0] @ np.asarray(initial_cov) @ a[0].T + q[0] if t == 0 else q[t]
        ) + jitter * np.eye(n_latent)
        if t > 0:
            residuals[row, slice((t - 1) * n_latent, t * n_latent)] = -a[t]
    precision = residuals.T @ np.linalg.solve(covariance, residuals)
    expected_rhs = (residuals.T @ np.linalg.solve(covariance, means.reshape(-1))).reshape(
        n_time, n_latent
    )

    lower, tri_diag, upper, tri_rhs = jax.jit(
        lambda: _build_prior_tridiagonal_system(
            transitions, covariances, offsets, initial_mean, initial_cov, jitter=jitter
        )
    )()
    band_diag, bands, band_rhs = _build_prior_banded_system(
        transitions, covariances, offsets, initial_mean, initial_cov, bandwidth, jitter=jitter
    )
    np.testing.assert_allclose(tri_rhs, expected_rhs, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(band_rhs, expected_rhs, atol=2e-6, rtol=2e-6)
    assert tri_diag.dtype == band_diag.dtype == jnp.float32
    assert bands.shape == (bandwidth, n_time, n_latent, n_latent)
    for t in range(n_time):
        row = slice(t * n_latent, (t + 1) * n_latent)
        np.testing.assert_allclose(band_diag[t], precision[row, row], atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(
            tri_diag[t], precision[row, row] + jitter * np.eye(n_latent), atol=2e-6, rtol=2e-6
        )
        if t > 0:
            np.testing.assert_allclose(
                lower[t],
                precision[row, slice((t - 1) * n_latent, t * n_latent)],
                atol=2e-6,
                rtol=2e-6,
            )
        if t + 1 < n_time:
            expected_upper = precision[row, slice((t + 1) * n_latent, (t + 2) * n_latent)]
            np.testing.assert_allclose(upper[t], expected_upper, atol=2e-6, rtol=2e-6)
            if bandwidth:
                np.testing.assert_allclose(bands[0, t], expected_upper, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(lower[0], 0)
    np.testing.assert_array_equal(upper[-1], 0)
    if bandwidth:
        np.testing.assert_array_equal(bands[0, -1], 0)
        np.testing.assert_array_equal(bands[1:], 0)


@pytest.mark.parametrize("budget", [0, 1, 8])
def test_fixed_point_preserves_iteration_budget_and_last_mode(budget: int) -> None:
    mode, n_steps = solve_fixed_point_mode(
        lambda value, _args: 0.5 * value + 1, jnp.zeros((1,)), max_steps=budget
    )
    steps = max(budget, 1)
    np.testing.assert_allclose(mode, 2 * (1 - 0.5**steps))
    assert int(n_steps) == steps
