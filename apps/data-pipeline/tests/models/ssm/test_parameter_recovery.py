"""MAP mode recovery and preconditioning on a known Gaussian process.

Support-specific solver algebra and gradients are checked in the Laplace
reference tests; this fit checks real optimizer convergence, parameter accuracy,
and the optimizer-derived covariance consumed by particle warmup.
"""

import numpy as np
import pytest

from nof1_causal_lab.models.ssm import SSMModel
from nof1_causal_lab.models.ssm.inference.warmup import map as map_warmup
from nof1_causal_lab.models.ssm.inference.warmup.parameter_warmup import (
    _laplace_preconditioner_chol_from_map_result,
)
from tests.model_fixtures import compile_fit_fixture, make_lgss_data

pytestmark = [pytest.mark.inference(concern="warmup"), pytest.mark.inference(concern="recovery")]


@pytest.mark.timeout(300)
def test_map_initialization_recovers_mode_and_builds_preconditioner(monkeypatch):
    # Retain enough observations to distinguish process and observation noise.
    data = make_lgss_data(T=250, decay_diag=-0.3, diff_sd=0.2, obs_sd=0.25)
    model = SSMModel(compile_fit_fixture(data["spec"]))

    def sample_at_mode(_key, z_mode, _chol_cov, *, num_samples):
        assert num_samples == 1
        return z_mode[None, :]

    # Control only the final Gaussian draw so public-coordinate accuracy checks
    # measure the fitted mode. Sampling algebra has its own cheap contract tests;
    # optimization, covariance construction, and preconditioning all remain real.
    monkeypatch.setattr(map_warmup, "_sample_gaussian_parameter_posterior", sample_at_mode)
    result = map_warmup.fit_map(
        model,
        observations=data["observations"],
        times=data["times"],
        num_samples=1,
        # A linear Gaussian model reaches its exact latent mode in one step.
        n_ieks_iters=1,
        maxiter=100,
        tol=1e-5,
        n_init_samples=8,
        parameter_covariance_method="optimizer_hess_inv",
        seed=0,
    )

    diagnostics = result.diagnostics
    assert diagnostics["optimizer"] == "L-BFGS-B"
    assert diagnostics["success"] is True
    assert diagnostics["status"] == 0
    assert diagnostics["mode_log_posterior"] > diagnostics["init_log_posterior_best"]

    mode = result.get_samples()
    assert abs(-abs(float(mode["vf_0_p0"][0])) - data["true_decay_diag"]) < 0.12
    assert abs(float(mode["diffusion_diag_free"][0, 0]) - data["true_diff_diag"]) < 0.08
    assert abs(float(mode["manifest_var_diag_free"][0, 0]) - data["true_obs_sd"]) < 0.05

    assert diagnostics["parameter_covariance_method"] == "optimizer_hess_inv"
    covariance = np.asarray(diagnostics["parameter_covariance"])
    assert np.isfinite(covariance).all()
    assert (np.linalg.eigvalsh(covariance) > 0).all()
    preconditioner = np.asarray(_laplace_preconditioner_chol_from_map_result(result))
    np.testing.assert_allclose(
        preconditioner @ preconditioner.T,
        covariance + 1e-6 * np.eye(covariance.shape[0]),
        rtol=1e-5,
        atol=1e-6,
    )
