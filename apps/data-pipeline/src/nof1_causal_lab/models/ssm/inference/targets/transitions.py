"""Initialization-only Gaussian views of the declared Dynestyx state evolution."""

from typing import TYPE_CHECKING, cast

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
from dynestyx.inference.configs.discretizer import ExactAffineConfig

from nof1_causal_lab.models.ssm.covariance_utils import CHOL_JITTER
from nof1_causal_lab.models.ssm.dynamics.linearisation import infer_linearisation
from nof1_causal_lab.models.ssm.shapes import Array, Float

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.dynamics.vector_field import StructuralDrift


def build_discrete_transitions(
    dynamics: dsx.StochasticContinuousTimeStateEvolution,
    time_intervals: Float[Array, " T"],
    *,
    linearization_states: Array | None = None,
) -> dsx.LinearGaussianParams:
    """Ask Dynestyx for the local Gaussian model used to initialize particles.

    Affine fields use their state-independent Jacobian at zero. Nonlinear fields
    require an explicit reference trajectory. Neither view defines the production
    particle target, which uses Euler-Maruyama over the true nonlinear drift.
    """
    drift = cast("StructuralDrift", dynamics.drift)
    n_latent = drift.vector_field.n_latent
    time_intervals = jnp.asarray(time_intervals)
    shape = (time_intervals.shape[0], n_latent)
    if infer_linearisation(drift.vector_field) == "constant":
        states = jnp.zeros(shape, dtype=time_intervals.dtype)
    else:
        if linearization_states is None:
            raise ValueError(
                "Trajectory-dependent vector-field discretization requires "
                "linearization_states with one state per interval."
            )
        states = jnp.asarray(linearization_states)
        if states.shape != shape:
            raise ValueError(f"linearization_states must have shape {shape}, got {states.shape}")

    # The covariance extracted from the Van Loan exponential is linear in L Lᵀ.
    # Keep its noise block near
    # unit scale so large Pathfinder draws do not exhaust expm's squaring
    # budget solely because of diffusion magnitude. Restore the exact scale
    # afterward. The library's explicit covariance regularization is confined
    # to this Gaussian initialization view, including lawless input coordinates.
    diffusion = dynamics.diffusion.as_matrix(x=None, u=None, t=0, state_dim=n_latent)
    scale = jax.lax.stop_gradient(jnp.maximum(1.0, jnp.max(jnp.abs(diffusion))))
    normalized = eqx.tree_at(
        lambda value: value.diffusion, dynamics, dsx.FullDiffusion(diffusion / scale)
    )

    starts = jnp.cumsum(time_intervals) - time_intervals - time_intervals[0]

    def at_interval(state: jax.Array, start: jax.Array, dt: jax.Array) -> dsx.LinearGaussianParams:
        affine_drift = dsx.linearize_drift(normalized.total_drift, x=state, u=None, t=start)
        affine_model = dsx.StochasticContinuousTimeStateEvolution(
            drift=affine_drift, diffusion=normalized.diffusion
        )
        evolution = cast(
            "dsx.LinearGaussianStateEvolution",
            dsx.discretize_state_evolution(
                affine_model, ExactAffineConfig(covariance_jitter=CHOL_JITTER)
            ),
        )
        params = evolution.params_at(start, start + dt)
        return params._replace(cov=(params.cov * scale) * scale)

    return jax.vmap(at_interval)(states, starts, time_intervals)
