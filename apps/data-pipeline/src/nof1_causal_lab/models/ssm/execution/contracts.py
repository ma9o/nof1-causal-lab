"""Measurement parameters and initialization-backend diagnostics.

Defines the interface that likelihood backends must implement:
compute_log_likelihood(params, observations, times) -> jnp.ndarray

Returns the (T,) cumulative log-normalizing-constant array from the filter.
The total log-likelihood is lnc[-1]; per-timestep one-step-ahead predictive
log-likelihoods are jnp.diff(lnc, prepend=0.0).

Used by Laplace likelihood backends to inject marginalized state likelihoods
into NumPyro models via numpyro.factor().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol

import jax.numpy as jnp

if TYPE_CHECKING:
    from dynestyx import StochasticContinuousTimeStateEvolution
    from jax.typing import ArrayLike, DTypeLike
    from numpyro.distributions import MultivariateNormal

    from nof1_causal_lab.models.ssm.shapes import Array, Float

MISSING_DATA_LARGE_VAR = 1e10
CHOL_JITTER = 1e-8
NUMERICAL_EPSILON = 1e-10
PROB_CLIP_MIN = 1e-7

type LikelihoodParameterValue = jnp.ndarray | int | float
type LikelihoodExtraParams = dict[str, LikelihoodParameterValue]

LIKELIHOOD_SOLVER_KIND_POINT_IEKS = 1
LIKELIHOOD_SOLVER_KIND_SUPPORT_IEKS = 2
LIKELIHOOD_SOLVER_KIND_DENSE_SUPPORT = 3


class MeasurementParams(NamedTuple):
    """Measurement mapping before the family-specific link and observation law.

    ``lambda_mat @ state + manifest_means`` is the linear predictor.
    ``manifest_cov`` supplies Gaussian covariance or the configured family's
    dispersion inputs; the observation model determines the actual density.
    """

    lambda_mat: Float[Array, "M D"]
    manifest_means: Float[Array, " M"]
    manifest_cov: Float[Array, "M M"]  # Σ_R


class InitializationLikelihoodBackend(Protocol):
    """Likelihood operation required by the NumPyro initialization model."""

    def compute_log_likelihood(
        self,
        dynamics: StochasticContinuousTimeStateEvolution,
        measurement_params: MeasurementParams,
        initial_state: MultivariateNormal,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        /,
        *,
        extra_params: LikelihoodExtraParams | None = None,
    ) -> jnp.ndarray: ...


def build_likelihood_eval_aux(
    dtype: DTypeLike,
    *,
    solver_kind: int,
    n_iterations: ArrayLike = 0,
    n_accepted_steps: ArrayLike = 0,
    init_log_joint: ArrayLike = float("nan"),
    final_log_joint: ArrayLike = float("nan"),
    final_rel_change: ArrayLike = float("nan"),
    final_damping: ArrayLike = float("nan"),
    final_step_alpha: ArrayLike = float("nan"),
    final_step_norm: ArrayLike = float("nan"),
    laplace_logdet: ArrayLike = float("nan"),
    min_chol_diag: ArrayLike = float("nan"),
) -> dict[str, jnp.ndarray]:
    """Build a fixed-shape backend diagnostic payload for host-side progress logs."""
    return {
        "solver_kind": jnp.asarray(solver_kind, dtype=jnp.int32),
        "n_iterations": jnp.asarray(n_iterations, dtype=jnp.int32),
        "n_accepted_steps": jnp.asarray(n_accepted_steps, dtype=jnp.int32),
        "init_log_joint": jnp.asarray(init_log_joint, dtype=dtype),
        "final_log_joint": jnp.asarray(final_log_joint, dtype=dtype),
        "final_rel_change": jnp.asarray(final_rel_change, dtype=dtype),
        "final_damping": jnp.asarray(final_damping, dtype=dtype),
        "final_step_alpha": jnp.asarray(final_step_alpha, dtype=dtype),
        "final_step_norm": jnp.asarray(final_step_norm, dtype=dtype),
        "laplace_logdet": jnp.asarray(laplace_logdet, dtype=dtype),
        "min_chol_diag": jnp.asarray(min_chol_diag, dtype=dtype),
    }
