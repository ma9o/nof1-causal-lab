"""Scientific drift derivatives and Dynestyx views used to initialize particles."""

from nof1_causal_lab.artifacts.identity import scientific_id
from tests.dynamics_fixtures import hill_term, interaction_term
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import pytest

from nof1_causal_lab.models.ssm.dynamics import DiagonalDecay, Intercept, Intervention, LinearEdge, VectorField, VectorFieldArgs, simulate
from nof1_causal_lab.models.ssm.dynamics.edges import DenseLinear
from nof1_causal_lab.models.ssm.execution.dynamical_model import continuous_state_evolution
from nof1_causal_lab.models.ssm.inference.targets.transitions import build_discrete_transitions
from tests.model_fixtures import affine_test_evolution


def _drift_derivatives(field, state, args):
    derivative = jax.jacfwd(lambda x: field(jnp.array(0.0), x, args))(state)
    return derivative, field(jnp.array(0.0), state, args) - derivative @ state


def _warmup_transition(field, state, args, covariance, dt):
    evolution = continuous_state_evolution(
        field, args.params, covariance, intervention=args.intervention
    )
    parameters = build_discrete_transitions(
        evolution, jnp.array([dt]), linearization_states=state[None]
    )
    assert parameters.bias is not None
    return parameters.A[0], parameters.cov[0], parameters.bias[0]


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
        params = (
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
        params = ({scientific_id("parameter", "weight"): jnp.asarray(w)},)
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        x_lin = jnp.array([a0, b0, 0.0])
        A_loc, b_loc = _drift_derivatives(vf, x_lin, args)
        assert float(A_loc[2, 0]) == pytest.approx(w * b0, abs=1e-6)
        assert float(A_loc[2, 1]) == pytest.approx(w * a0, abs=1e-6)
        # Intercept: f(x_lin) - A · x_lin = w·a·b - (w·b·a + w·a·b) = -w·a·b
        f_at_x = w * a0 * b0
        expected_b = jnp.array([0.0, 0.0, f_at_x - (w * b0 * a0 + w * a0 * b0)])
        assert jnp.allclose(b_loc, expected_b, atol=1e-6)


# =============================================================================
# discretize_at_state parity for DenseLinear case
# =============================================================================


class TestDiscretizeDenseLinearParity:
    """For a single ``DenseLinear`` component, the warmup transition compiler
    must produce the same matrices as the existing
    Dynestyx exact-affine discretization, regardless of ``x_lin``."""

    def test_matches_linear_path_exactly(self):
        A = jnp.array(
            [
                [-1.5, 0.3, 0.1],
                [0.4, -2.0, 0.2],
                [0.1, 0.5, -1.8],
            ]
        )
        c = jnp.array([1.0, -0.5, 0.2])
        diffusion_cov = jnp.eye(3) * 0.1
        dt = 0.5

        reference = affine_test_evolution(A, diffusion_cov, c).params_at(0.0, dt)
        A_d_ref, Q_d_ref, c_d_ref = reference.A, reference.cov, reference.bias

        vf = _dense_matrix_vector_field(n_latent=3)
        args = VectorFieldArgs(params=({"drift": A, "cint": c},), intervention=Intervention.none())
        # x_lin is irrelevant for a linear field — pick something non-trivial
        x_lin = jnp.array([1.7, -0.3, 0.5])
        A_d, Q_d, c_d = _warmup_transition(vf, x_lin, args, diffusion_cov, dt)

        assert jnp.allclose(A_d, A_d_ref, atol=1e-6)
        assert jnp.allclose(Q_d, Q_d_ref, atol=1e-6)
        assert jnp.allclose(c_d, c_d_ref, atol=1e-6)

    def test_zero_intercept(self):
        A = -jnp.eye(2)
        diffusion_cov = jnp.eye(2) * 0.05
        vf = _dense_matrix_vector_field(n_latent=2)
        args = VectorFieldArgs(
            params=({"drift": A, "cint": jnp.zeros(2)},),
            intervention=Intervention.none(),
        )
        x_lin = jnp.array([0.0, 0.0])
        A_d, _, c_d = _warmup_transition(vf, x_lin, args, diffusion_cov, dt=0.2)
        assert jnp.allclose(A_d, jla.expm(A * 0.2), atol=1e-6)
        assert jnp.allclose(c_d, 0.0, atol=1e-6)


# =============================================================================
# Non-linear SSRI chain: linearization structure + Diffrax cross-check
# =============================================================================


@pytest.mark.cpu_expensive
class TestSSRIChainLinearization:
    """Build the full SSRI chain, compute the Jacobian at baseline steady
    state, and verify the structure matches expectations."""

    DOSE = 0
    ADHERENCE = 1
    C_P = 2
    C_E = 3
    AFFECTIVE = 4

    K_P = 1.0
    K_E0 = 0.1
    DECAY_AFF = 1.0
    EMAX = 2.0
    EC50_VAL = 1.0
    N_HILL = 2.0

    def _build(self):
        vf = VectorField(
            n_latent=5,
            components=(
                DiagonalDecay(),
                Intercept(),
                interaction_term(source_a=self.DOSE, source_b=self.ADHERENCE, target=self.C_P).build(),
                LinearEdge(source=self.C_P, target=self.C_E),
                hill_term(source=self.C_E, target=self.AFFECTIVE).build(),
            ),
        )
        params = (
            {"decay": jnp.array([1.0, 1.0, self.K_P, self.K_E0, self.DECAY_AFF])},
            {"cint": jnp.array([1.0, 1.0, 0.0, 0.0, 0.0])},
            {scientific_id("parameter", "weight"): jnp.asarray(self.K_P)},
            {"weight": jnp.asarray(self.K_E0)},
            {
                scientific_id("parameter", "emax"): jnp.asarray(self.EMAX),
                scientific_id("parameter", "ec50"): jnp.asarray(self.EC50_VAL),
                scientific_id("parameter", "exponent"): jnp.asarray(self.N_HILL),
            },
        )
        return vf, params

    def test_jacobian_at_baseline_has_expected_structure(self):
        vf, params = self._build()
        baseline = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        A_loc, b_loc = _drift_derivatives(vf, baseline, args)

        # Diagonal: each latent's natural decay
        assert float(A_loc[self.DOSE, self.DOSE]) == pytest.approx(-1.0, abs=1e-6)
        assert float(A_loc[self.ADHERENCE, self.ADHERENCE]) == pytest.approx(-1.0, abs=1e-6)
        assert float(A_loc[self.C_P, self.C_P]) == pytest.approx(-self.K_P, abs=1e-6)
        assert float(A_loc[self.C_E, self.C_E]) == pytest.approx(-self.K_E0, abs=1e-6)
        assert float(A_loc[self.AFFECTIVE, self.AFFECTIVE]) == pytest.approx(
            -self.DECAY_AFF, abs=1e-6
        )

        # Multiplicative edge at baseline (dose=1, adherence=1):
        #   ∂(k_p · dose · adherence)/∂dose      = k_p · 1 = k_p
        #   ∂(k_p · dose · adherence)/∂adherence = k_p · 1 = k_p
        assert float(A_loc[self.C_P, self.DOSE]) == pytest.approx(self.K_P, abs=1e-6)
        assert float(A_loc[self.C_P, self.ADHERENCE]) == pytest.approx(self.K_P, abs=1e-6)

        # LinearEdge C_P → C_E with weight k_e0
        assert float(A_loc[self.C_E, self.C_P]) == pytest.approx(self.K_E0, abs=1e-6)

        # HillEdge C_E → AFFECTIVE at C_E=1=EC50: slope = Emax·n/(4·EC50)
        expected_hill_slope = self.EMAX * self.N_HILL / (4.0 * self.EC50_VAL)
        assert float(A_loc[self.AFFECTIVE, self.C_E]) == pytest.approx(
            expected_hill_slope, abs=1e-4
        )

        # b_loc: dynamics at baseline minus A · baseline. At baseline the
        # natural dynamics is zero (steady state), so b_loc = -A · baseline.
        assert jnp.allclose(b_loc, -A_loc @ baseline, atol=1e-5)

    def test_discrete_step_matches_diffrax_for_linearized_system(self):
        """Discretizing at ``x_lin`` over a small ``dt`` and applying once
        must match a Diffrax integration of the linearized vector field
        from the same point. The linearization is the same; this is a
        sanity check that the expm discretization and the Diffrax
        integrator agree on the linear case."""
        vf, params = self._build()
        x_lin = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])
        args = VectorFieldArgs(params=params, intervention=Intervention.none())
        diffusion_cov = jnp.zeros((5, 5))  # deterministic for the check
        dt = 0.1

        A_d, _, b_d = _warmup_transition(vf, x_lin, args, diffusion_cov, dt)
        # Step from the linearization point itself
        next_state_discrete = A_d @ x_lin + b_d

        # Linearized system as a fresh DenseLinear vector field
        A_loc, b_loc = _drift_derivatives(vf, x_lin, args)
        lin_vf = VectorField(n_latent=5, components=(DenseLinear(),))
        lin_args = VectorFieldArgs(
            params=({"drift": A_loc, "cint": b_loc},),
            intervention=Intervention.none(),
        )
        time_grid = jnp.array([0.0, dt])
        traj = simulate(
            lin_vf,
            lin_args.params,
            Intervention.none(),
            x_lin,
            time_grid,
        )
        next_state_diffrax = traj[-1]

        assert jnp.allclose(next_state_discrete, next_state_diffrax, atol=1e-4)
