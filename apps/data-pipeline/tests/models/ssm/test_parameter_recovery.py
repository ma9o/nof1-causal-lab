"""MAP initialization accuracy and uncertainty on a known Gaussian process.

Support-specific solver algebra and gradients are checked in the Laplace
reference tests; this fit checks optimizer convergence and parameter recovery.
"""

from typing import Any

import jax.numpy as jnp
import pytest

from nof1_causal_lab.models.ssm import (
    SSMModel,
)
from nof1_causal_lab.models.ssm.inference.warmup.map import fit_map
from tests.model_fixtures import (
    make_lgss_data,
)
from tests.ssm_test_utils import assert_recovery_ci

pytestmark = [pytest.mark.warmup, pytest.mark.recovery]


def _assert_lgss_recovery(
    samples: dict[str, jnp.ndarray],
    data: dict[str, Any],
) -> None:
    assert_recovery_ci(
        samples["vf_0_p0"],
        data["true_decay_diag"],
        "Dynamics",
        transform=lambda s: -jnp.abs(s),
    )
    assert_recovery_ci(
        samples["diffusion_diag_free"][:, 0],
        data["true_diff_diag"],
        "Diffusion",
    )
    assert_recovery_ci(
        samples["manifest_var_diag_free"][:, 0],
        data["true_obs_sd"],
        "Obs SD",
    )


def _make_map_recovery_data() -> dict[str, Any]:
    """1D LGSS tuned for MAP: longer series and higher SNR than defaults.

    The longer T and tighter noise make mode-finding and the local Gaussian
    approximation reliable enough for parameter-recovery checks.
    """
    return make_lgss_data(T=250, decay_diag=-0.3, diff_sd=0.2, obs_sd=0.25)


# =============================================================================
# MAP
# =============================================================================


class TestMapLaplaceRecovery:
    """Canonical MAP + Laplace recovery tests."""

    @pytest.mark.timeout(300)
    def test_map_recovery(self):
        """MAP recovers a well-identified 1D LGSS through the Laplace backend.

        This checks more than execution:
        1. L-BFGS-B converges on a genuinely informative dataset.
        2. The Gaussian parameter-space approximation contains the truth in its
           90% intervals.
        3. Posterior means stay close to the generating parameters, so the
           approximation is not passing only because the intervals are overly
           wide.
        """
        data = _make_map_recovery_data()
        model = SSMModel(data["spec"])

        result = fit_map(
            model,
            observations=data["observations"],
            times=data["times"],
            num_samples=1000,
            # A linear Gaussian model reaches its exact latent mode in one step.
            n_ieks_iters=1,
            maxiter=100,
            tol=1e-5,
            n_init_samples=8,
            parameter_covariance_method="exact_hessian",
            seed=0,
        )

        assert result.diagnostics["optimizer"] == "L-BFGS-B"
        assert result.diagnostics["success"] is True
        assert result.diagnostics["status"] == 0

        samples = result.get_samples()
        _assert_lgss_recovery(samples, data)

        dynamics_mean = float(jnp.mean(-jnp.abs(samples["vf_0_p0"])))
        diff_mean = float(jnp.mean(samples["diffusion_diag_free"][:, 0]))
        obs_mean = float(jnp.mean(samples["manifest_var_diag_free"][:, 0]))

        assert abs(dynamics_mean - data["true_decay_diag"]) < 0.12
        assert abs(diff_mean - data["true_diff_diag"]) < 0.08
        assert abs(obs_mean - data["true_obs_sd"]) < 0.05
