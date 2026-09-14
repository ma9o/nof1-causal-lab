"""Native node potentials preserve nonlinear drift, metadata, and causal interventions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.ssm.dynamics import (
    Intervention,
    VariableOverride,
    VectorFieldArgs,
    constant_value,
    infer_linearisation,
)
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec, compile_dynamics
from nof1_causal_lab.models.ssm.execution.dynamical_model import continuous_state_evolution
from tests.dynamics_fixtures import linear_term, potential_term
from tests.model_fixtures import model_fixture


@pytest.mark.parametrize("quartic", [0.0, 0.5])
def test_native_potential_matches_restoring_force_and_its_derivatives(quartic):
    compiled = compile_dynamics(
        DynamicsSpec(
            2,
            (
                potential_term(
                    1,
                    center=0.5,
                    stiffness=1.2,
                    quartic=quartic,
                ),
            ),
        )
    )
    args = VectorFieldArgs(({},), Intervention.none())
    evolution = compiled.vector_field.evolution(args)
    assert evolution.potential is not None
    assert evolution.use_negative_gradient
    x = jnp.array([3.0, 2.0])

    def force(value):
        return evolution.total_drift(x=value, u=None, t=jnp.array(0.0))

    displacement = float(x[1]) - 0.5
    np.testing.assert_allclose(force(x), [0.0, -1.2 * displacement - quartic * displacement**3])
    np.testing.assert_allclose(
        jax.jacfwd(force)(x), [[0.0, 0.0], [0.0, -1.2 - 3 * quartic * displacement**2]]
    )
    np.testing.assert_allclose(force(jnp.array([3.0, 0.5])), [0.0, 0.0], atol=1e-7)
    assert infer_linearisation(compiled.vector_field) == "trajectory"


def test_clamp_removes_potential_input_forcing_and_process_noise():
    compiled = compile_dynamics(
        DynamicsSpec(
            2,
            (
                potential_term(
                    0,
                    center=0,
                    stiffness=1,
                    quartic=0.2,
                ),
                linear_term(0, 1, weight=2),
            ),
        )
    )
    intervention = Intervention((VariableOverride(0, constant_value(jnp.array(3.0))),))
    evolution = continuous_state_evolution(
        compiled.vector_field,
        ({}, {}),
        jnp.eye(2),
        intervention=intervention,
    )
    np.testing.assert_allclose(
        evolution.total_drift(jnp.array([3.0, 0.0]), None, jnp.array(0.0)), [0.0, 6.0]
    )
    np.testing.assert_allclose(
        evolution.diffusion.as_matrix(x=jnp.ones(2), u=None, t=0.0, state_dim=2),
        [[0.0, 0.0], [0.0, 1.0]],
    )


def test_potential_coefficients_keep_their_scientific_meanings():
    compiled = compile_dynamics(DynamicsSpec(1, (potential_term(0, quartic=None),)))
    assert {site.site_kind for site in compiled.site_registry} == {
        SiteKind.DYNAMICS_POTENTIAL_CENTER,
        SiteKind.DYNAMICS_DECAY,
        SiteKind.DYNAMICS_POTENTIAL_QUARTIC,
    }
    assert all(site.positions == (0,) for site in compiled.site_registry)


@pytest.mark.parametrize("kwargs", [{"stiffness": 0}, {"quartic": -1}])
def test_invalid_potential_coefficients_are_rejected(kwargs):
    with pytest.raises(ValueError, match=r"(positive|non-negative)"):
        potential_term(0, **kwargs)


def test_directed_edges_cannot_be_reinterpreted_as_potentials():
    model = model_fixture(
        n_latent=2,
        dynamics_spec=DynamicsSpec(2, (potential_term(0), potential_term(1), linear_term(0, 1))),
    )
    edge = model.execution_edges[0]
    changed = edge.model_copy(
        update={
            "mechanisms": tuple(
                term.model_copy(update={"kind": "potential"}) for term in edge.mechanisms
            )
        }
    )
    with pytest.raises(ValueError, match="Potentials belong to nodes"):
        model.revised(edges=tuple(changed if item.id == edge.id else item for item in model.edges))
