"""Initialization-only Gaussian views of the declared Dynestyx state evolution."""

from typing import TYPE_CHECKING, cast

import dynestyx as dsx
import jax
import jax.numpy as jnp
from dynestyx.inference.configs.discretizer import LocalLinearizationConfig

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

    def at_interval(state, dt):
        return dsx.linearized_transition_parameters(
            dynamics,
            LocalLinearizationConfig(covariance_jitter=0.0),
            linearization_state=state,
            previous_control=None,
            previous_time=0.0,
            time=dt,
        )

    return jax.vmap(at_interval)(states, time_intervals)
