"""Independent likelihood and gradient references for Laplace initialization."""

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dynestyx import StochasticContinuousTimeStateEvolution
from numpyro.distributions import MultivariateNormal
from scipy.optimize import brentq
from scipy.stats import norm, t

from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.dynamics.edges import DenseLinear
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField
from nof1_causal_lab.models.ssm.execution.contracts import MeasurementParams
from nof1_causal_lab.models.ssm.execution.dynamical_model import (
    continuous_state_evolution,
)
from nof1_causal_lab.models.ssm.inference.targets.laplace import (
    LaplaceLikelihood,
    _dense_support_laplace_log_lik,
)
from tests.model_fixtures import (
    make_observation_support_runtime,
)
from tests.observation_fixtures import mean_density, observation_kernel, observation_laws

pytestmark = pytest.mark.inference(concern="warmup")


@pytest.mark.parametrize("solver", ["point", "interval", "dense", "interval_blocks"])
def test_student_t_laplace_value_and_gradient_match_scalar_reference(monkeypatch, solver):
    """A single observation of a Gaussian path reduces to a scalar Laplace integral.

    The reference integrates all orthogonal latent directions analytically and
    solves the remaining scalar mode with SciPy. This checks all three
    Optimistix paths against an independent reference.
    """
    from nof1_causal_lab.models.ssm.inference.targets import laplace

    monkeypatch.setattr(laplace, "_should_use_dense_support_laplace", lambda **_kwargs: False)
    n_time, n_latent = (4, 2) if solver == "interval_blocks" else (2, 1)
    state_mean = np.array([0.05, -0.03])[:n_latent]
    state_variance = np.array([0.6, 0.45])[:n_latent]
    process_variance = np.array([0.07, 0.04])[:n_latent]
    loadings = np.array([1.0, 0.4])[:n_latent]
    anchor_times = np.arange(n_time, dtype=float)
    observations = jnp.full((n_time, 1), jnp.nan).at[-1, 0].set(0.25)
    support = None
    weights = np.array([0.0, 1.0])
    if solver != "point":
        weights = np.ones(n_time) / (n_time - 1)
        weights[[0, -1]] /= 2
        start_times, end_times = np.full((2, n_time, 1), np.nan)
        start_times[-1, 0], end_times[-1, 0] = 0.0, n_time - 1
        coefficients = np.full((n_time, 1), 0.5)
        coefficients[0, 0] = 0.0
        support = make_observation_support_runtime(
            anchor_times=anchor_times,
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=[f"{n_time - 1}d"],
            support_start_times=start_times,
            support_end_times=end_times,
            interval_prev_coeffs=coefficients,
            interval_curr_coeffs=coefficients,
            interval_weights=coefficients * 2,
        )
    backend = LaplaceLikelihood(
        n_latent=n_latent,
        n_manifest=1,
        n_ieks_iters=4,
        observation_support=support,
    )
    dynamics = _runtime_dynamics(
        dynamics=-0.09 * jnp.eye(n_latent, dtype=jnp.float32),
        diffusion=jnp.diag(jnp.sqrt(jnp.asarray(process_variance, dtype=jnp.float32))),
    )
    initial = MultivariateNormal(
        loc=jnp.asarray(state_mean, dtype=jnp.float32),
        covariance_matrix=jnp.diag(jnp.asarray(state_variance, dtype=jnp.float32)),
    )

    def _objective(raw):
        measurement = MeasurementParams(
            lambda_mat=jnp.asarray(loadings[None], dtype=jnp.float32),
            manifest_means=jnp.zeros(1, dtype=jnp.float32),
            manifest_cov=(jnp.exp(raw[1]) + 0.1).reshape(1, 1),
        )
        if solver == "dense":
            parameters = {"obs_df": jnp.exp(raw[0]) + 2.5}
            law = observation_laws([DistributionFamily.STUDENT_T], parameters=parameters)[0]
            value, _aux = _dense_support_laplace_log_lik(
                jnp.array([[0.0], [0.25]], dtype=jnp.float32),
                jnp.array([[False], [True]]),
                jnp.full((2, 1, 1), np.exp(-0.09), dtype=jnp.float32),
                jnp.full((2, 1, 1), -0.07 * np.expm1(-0.18) / 0.18, dtype=jnp.float32),
                jnp.zeros((2, 1), dtype=jnp.float32),
                measurement.lambda_mat,
                measurement.manifest_means,
                measurement.manifest_cov,
                initial.mean,
                initial.covariance_matrix,
                observation_kernel(
                    [DistributionFamily.STUDENT_T], [LinkFunction.IDENTITY], parameters
                ),
                mean_density(law),
                support,
                8,
            )
            return value
        return backend.compute_log_likelihood(
            dynamics,
            measurement,
            initial,
            observations,
            jnp.ones(n_time, dtype=jnp.float32),
            observation_laws=observation_laws(
                [DistributionFamily.STUDENT_T], parameters={"obs_df": jnp.exp(raw[0]) + 2.5}
            ),
        )

    anchors = anchor_times + 1
    stationary_var = np.dot(loadings**2, process_variance) / (2 * 0.09)
    initial_variance = np.dot(loadings**2, state_variance)
    covariance = stationary_var * np.exp(-0.09 * np.abs(anchors[:, None] - anchors[None, :])) + (
        initial_variance - stationary_var
    ) * np.exp(-0.09 * (anchors[:, None] + anchors[None, :]))
    prior_mean = weights @ (np.dot(loadings, state_mean) * np.exp(-0.09 * anchors))
    prior_var = weights @ covariance @ weights

    def _reference(raw):
        df, variance = np.exp(raw) + np.array([2.5, 0.1])
        observed = 0.25
        mode = brentq(
            lambda mean: (
                (mean - prior_mean) / prior_var
                - (df + 1) * (observed - mean) / (df * variance + (observed - mean) ** 2)
            ),
            prior_mean,
            observed,
        )
        residual_sq = (observed - mode) ** 2
        curvature = (
            1 / prior_var
            + (df + 1) * (df * variance - residual_sq) / (df * variance + residual_sq) ** 2
        )
        return (
            norm.logpdf(mode, loc=prior_mean, scale=np.sqrt(prior_var))
            + t.logpdf(observed, df, loc=mode, scale=np.sqrt(variance))
            + 0.5 * np.log(2 * np.pi / curvature)
        )

    raw = np.array([0.35, -1.2])
    value, gradient = jax.jit(jax.value_and_grad(_objective))(jnp.asarray(raw, dtype=jnp.float32))
    eps = 1e-4
    reference_gradient = np.array(
        [(_reference(raw + step) - _reference(raw - step)) / (2 * eps) for step in eps * np.eye(2)]
    )
    np.testing.assert_allclose(value, _reference(raw), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(gradient, reference_gradient, rtol=2e-3, atol=2e-4)


def _runtime_dynamics(
    *,
    dynamics: jnp.ndarray,
    diffusion: jnp.ndarray,
    cint: jnp.ndarray | None = None,
) -> StochasticContinuousTimeStateEvolution:
    params = {"drift": dynamics}
    if cint is not None:
        params["cint"] = cint
    return continuous_state_evolution(
        vector_field=VectorField(
            n_latent=int(dynamics.shape[0]),
            components=(DenseLinear(),),
        ),
        vf_params=(params,),
        diffusion=diffusion,
    )


class TestLaplaceSupportAware:
    def test_laplace_banded_matches_dense_reference(self, monkeypatch):
        from nof1_causal_lab.models.ssm.inference.targets import laplace

        # Force the numerical branch on a small path. Float64 support metadata
        # with float32 states also covers the former oversized window smoke.
        banded_solver = Mock(wraps=laplace._support_aware_ieks_laplace)
        monkeypatch.setattr(laplace, "_should_use_dense_support_laplace", lambda **_kwargs: False)
        monkeypatch.setattr(laplace, "_support_aware_ieks_laplace", banded_solver)
        support = make_observation_support_runtime(
            anchor_times=np.array([0.0, 1.0, 2.0]),
            manifest_names=["avg_signal"],
            support_kinds=["interval"],
            observation_windows=["2d"],
            support_start_times=np.array([[np.nan], [np.nan], [0.0]]),
            support_end_times=np.array([[np.nan], [np.nan], [2.0]]),
            interval_prev_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_curr_coeffs=np.array([[0.0], [0.5], [0.5]]),
            interval_weights=np.array([[0.0], [1.0], [1.0]]),
        )
        backend = LaplaceLikelihood(
            n_latent=1,
            n_manifest=1,
            n_ieks_iters=1,
            observation_support=support,
        )
        ct_params = _runtime_dynamics(
            dynamics=jnp.array([[-0.4]], dtype=jnp.float32),
            diffusion=jnp.linalg.cholesky(jnp.array([[0.1]], dtype=jnp.float32)),
            cint=jnp.array([0.0], dtype=jnp.float32),
        )
        meas_params = MeasurementParams(
            lambda_mat=jnp.array([[1.0]], dtype=jnp.float32),
            manifest_means=jnp.array([0.0], dtype=jnp.float32),
            manifest_cov=jnp.array([[0.2]], dtype=jnp.float32),
        )
        init = MultivariateNormal(
            loc=jnp.array([0.0], dtype=jnp.float32),
            covariance_matrix=jnp.array([[1.0]], dtype=jnp.float32),
        )
        observations = jnp.array([[jnp.nan], [jnp.nan], [0.25]], dtype=jnp.float32)
        time_intervals = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)

        # Scalar OU transitions provide an independent input to the dense solver.
        Ad = jnp.full((3, 1, 1), np.exp(-0.4), dtype=jnp.float32)
        Qd = jnp.full((3, 1, 1), -0.1 * np.expm1(-0.8) / 0.8, dtype=jnp.float32)
        cd = jnp.zeros((3, 1), dtype=jnp.float32)
        obs_kernel = observation_kernel(
            [DistributionFamily.GAUSSIAN], [LinkFunction.IDENTITY], None
        )
        mean_log_prob_fn = mean_density(
            observation_laws([DistributionFamily.GAUSSIAN], parameters=None)[0]
        )

        @jax.jit
        def _banded(obs):
            return backend.compute_log_likelihood(
                ct_params,
                meas_params,
                init,
                obs,
                time_intervals,
                observation_laws=observation_laws([DistributionFamily.GAUSSIAN]),
            )

        @jax.jit
        def _dense(obs):
            value, _aux = _dense_support_laplace_log_lik(
                jnp.nan_to_num(obs, nan=0.0),
                ~jnp.isnan(obs),
                Ad,
                Qd,
                cd,
                meas_params.lambda_mat,
                meas_params.manifest_means,
                meas_params.manifest_cov,
                init.mean,
                init.covariance_matrix,
                obs_kernel,
                mean_log_prob_fn,
                support,
                1,
            )
            return value

        banded = _banded(observations)
        dense = _dense(observations)

        banded_solver.assert_called_once()
        # Independent OU covariance for the three latent anchors after the
        # initial prediction. The interval mean has weights (1/4, 1/2, 1/4).
        anchors = np.arange(1.0, 4.0)
        stationary_var = 0.1 / (2 * 0.4)
        covariance = stationary_var * np.exp(-0.4 * np.abs(anchors[:, None] - anchors[None, :])) + (
            1.0 - stationary_var
        ) * np.exp(-0.4 * (anchors[:, None] + anchors[None, :]))
        weights = np.array([0.25, 0.5, 0.25])
        variance = weights @ covariance @ weights + 0.2
        exact = -0.5 * (np.log(2 * np.pi * variance) + 0.25**2 / variance)
        np.testing.assert_allclose([banded, dense], exact, rtol=1e-4, atol=1e-4)
