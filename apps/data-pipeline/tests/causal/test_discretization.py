"""Scientific drift derivatives and Dynestyx views used to initialize particles."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.models.ssm.dynamics import (
    Intervention,
    VectorField,
    VectorFieldArgs,
)
from nof1_causal_lab.models.ssm.dynamics.edges import DenseLinear
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec, compile_dynamics
from nof1_causal_lab.models.ssm.execution.dynamical_model import continuous_state_evolution
from nof1_causal_lab.models.ssm.inference.targets.transitions import build_discrete_transitions
from tests.dynamics_fixtures import hill_term, interaction_term, potential_term


def _drift_derivatives(field, state, args):
    derivative = jax.jacfwd(lambda x: field(jnp.array(0.0), x, args))(state)
    return derivative, field(jnp.array(0.0), state, args) - derivative @ state


def _dense_matrix_vector_field(n_latent: int) -> VectorField:
    return VectorField(n_latent=n_latent, components=(DenseLinear(),))


# =============================================================================
# Per-primitive linearization checks
# =============================================================================


class TestLinearizePrimitives:
    def test_dense_linear_recovers_matrix(self):
        """``linearize`` on a single ``DenseLinear`` component must return
        ``(A, c)`` exactly — autodiff through ``A @ x + c`` reproduces ``A``."""
        A = jnp.array([[-1.0, 0.5], [0.3, -2.0]])
        c = jnp.array([0.1, -0.2])
        vf = _dense_matrix_vector_field(n_latent=2)
        args = VectorFieldArgs(params=({"drift": A, "cint": c},), intervention=Intervention.none())
        x_lin = jnp.array([0.7, -0.4])
        A_loc, b_loc = _drift_derivatives(vf, x_lin, args)
        assert jnp.allclose(A_loc, A, atol=1e-6)
        assert jnp.allclose(b_loc, c, atol=1e-6)

    def test_hill_jacobian_matches_analytic(self):
        """At ``x = EC50``, ``dHill/dx = Emax · n / (4 · EC50)``. The
        Jacobian entry for the source must match this."""
        Emax, EC50, n = 2.0, 1.0, 2.0
        vf = VectorField(
            n_latent=2,
            components=(hill_term(source=0, target=1).build(),),
        )
        params: tuple[dict[str, jax.Array], ...] = (
            {
                scientific_id("parameter", "emax"): jnp.asarray(Emax),
                scientific_id("parameter", "ec50"): jnp.asarray(EC50),
                scientific_id("parameter", "exponent"): jnp.asarray(n),
            },
        )
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        x_lin = jnp.array([EC50, 0.0])  # source at EC50
        A_loc, _ = _drift_derivatives(vf, x_lin, args)
        expected_slope = Emax * n / (4.0 * EC50)
        assert float(A_loc[1, 0]) == pytest.approx(expected_slope, abs=1e-4)
        # Other entries: target's effect on itself is 0 (no decay/feedback);
        # source has no self-influence either.
        assert float(A_loc[0, 0]) == pytest.approx(0.0, abs=1e-6)
        assert float(A_loc[1, 1]) == pytest.approx(0.0, abs=1e-6)

    def test_multiplicative_jacobian_off_diagonals(self):
        """For ``f(η) = w · η_a · η_b`` at ``(a₀, b₀)``: ``∂f/∂η_a = w · b₀``,
        ``∂f/∂η_b = w · a₀``."""
        w = 0.5
        a0, b0 = 3.0, 4.0
        vf = VectorField(
            n_latent=3,
            components=(interaction_term(source_a=0, source_b=1, target=2).build(),),
        )
        params: tuple[dict[str, jax.Array], ...] = (
            {scientific_id("parameter", "weight"): jnp.asarray(w)},
        )
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        x_lin = jnp.array([a0, b0, 0.0])
        A_loc, b_loc = _drift_derivatives(vf, x_lin, args)
        assert float(A_loc[2, 0]) == pytest.approx(w * b0, abs=1e-6)
        assert float(A_loc[2, 1]) == pytest.approx(w * a0, abs=1e-6)
        # Intercept: f(x_lin) - A · x_lin = w·a·b - (w·b·a + w·a·b) = -w·a·b
        f_at_x = w * a0 * b0
        expected_b = jnp.array([0.0, 0.0, f_at_x - (w * b0 * a0 + w * a0 * b0)])
        assert jnp.allclose(b_loc, expected_b, atol=1e-6)


@pytest.mark.warmup
def test_warmup_transitions_match_local_ou_moments():
    """Check local drift, bias, covariance and irregular gaps against scalar OU laws."""
    stiffness = np.array([0.5, 1.5])
    quartic = np.array([0.3, 0.2])
    states = np.array([[0.4, -0.8], [-1.0, 0.3]])
    gaps = np.array([0.2, 1.3])
    variance = np.array([0.05, 0.12])
    field = compile_dynamics(
        DynamicsSpec(
            n_latent=2,
            components=tuple(
                potential_term(
                    target=i, center=0, stiffness=float(stiffness[i]), quartic=float(quartic[i])
                )
                for i in range(2)
            ),
        )
    ).vector_field
    evolution = continuous_state_evolution(field, ({}, {}), jnp.diag(jnp.asarray(variance)))
    transitions = build_discrete_transitions(
        evolution, jnp.asarray(gaps), linearization_states=jnp.asarray(states)
    )

    # f(x) = -k*x - q*x³, so f'(x) = -k - 3*q*x² and f(x) - f'(x)*x = 2*q*x³.
    rate = -stiffness - 3 * quartic * states**2
    intercept = 2 * quartic * states**3
    dt = gaps[:, None]
    expected_A = np.exp(rate * dt)
    expected_bias = intercept * np.expm1(rate * dt) / rate
    expected_cov = variance * np.expm1(2 * rate * dt) / (2 * rate)
    np.testing.assert_allclose(transitions.A, [np.diag(row) for row in expected_A], atol=1e-6)
    assert transitions.bias is not None
    np.testing.assert_allclose(transitions.bias, expected_bias, atol=1e-6)
    np.testing.assert_allclose(transitions.cov, [np.diag(row) for row in expected_cov], atol=1e-6)
