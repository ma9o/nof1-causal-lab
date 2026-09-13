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

from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp

if TYPE_CHECKING:
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


def build_likelihood_eval_aux(
    dtype,
    *,
    solver_kind: int,
    **overrides,
) -> dict[str, jnp.ndarray]:
    """Build a fixed-shape backend diagnostic payload for host-side progress logs."""
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    aux = {
        "solver_kind": jnp.asarray(solver_kind, dtype=jnp.int32),
        "n_iterations": jnp.asarray(0, dtype=jnp.int32),
        "n_accepted_steps": jnp.asarray(0, dtype=jnp.int32),
        "init_log_joint": nan,
        "final_log_joint": nan,
        "final_rel_change": nan,
        "final_damping": nan,
        "final_step_alpha": nan,
        "final_step_norm": nan,
        "laplace_logdet": nan,
        "min_chol_diag": nan,
    }
    for key, value in overrides.items():
        if key not in aux:
            raise KeyError(f"Unknown likelihood-eval aux field: {key}")
        aux[key] = jnp.asarray(value, dtype=aux[key].dtype)
    return aux
