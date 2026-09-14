"""Tests for vector-field linearisation and component-native dynamics specs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import numpyro.distributions as ndist
from dynestyx import StochasticContinuousTimeStateEvolution

from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.models.ssm.dynamics import (
    DynamicsSpec,
    Intervention,
    StateDecay,
    VectorField,
    VectorFieldArgs,
    compile_dynamics,
    infer_linearisation,
)
from tests.dynamics_fixtures import decay_term, hill_term, intercept_term, linear_term
from tests.model_fixtures import (
    default_manifest_means_block,
    default_static_state_sd_block,
    model_fixture,
)


class TestInferLinearisation:
    def test_state_decay_only_is_constant(self):
        field = VectorField(
            n_latent=2,
            components=(
                StateDecay(target=0),
                StateDecay(target=1),
            ),
        )
        assert infer_linearisation(field) == "constant"

    def test_expression_terms_use_trajectory_initialization(self):
        spec = DynamicsSpec(
            n_latent=2,
            components=(
                *(decay_term(target=i) for i in range(2)),
                linear_term(source=0, target=1),
            ),
        )
        compiled = compile_dynamics(spec)
        # Generic expressions take the local initialization path; exact execution
        # never substitutes a linear drift based on their scientific role labels.
        assert infer_linearisation(compiled.vector_field) == "trajectory"

    def test_hill_makes_it_trajectory(self):
        spec = DynamicsSpec(
            n_latent=2,
            components=(
                *(decay_term(target=i) for i in range(2)),
                hill_term(
                    source=0,
                    target=1,
                ),
            ),
        )
        compiled = compile_dynamics(spec)
        assert infer_linearisation(compiled.vector_field) == "trajectory"


class TestSSMModelDynamicsDispatch:
    """The NumPyro model samples dynamics and delegates at the backend boundary."""

    def test_nonlinear_dynamics_uses_vector_field_backend_method(self):
        import numpyro
        from numpyro import handlers

        from nof1_causal_lab.models.ssm import SSMModel
        from nof1_causal_lab.models.ssm.structure import (
            DiffusionBlockSpec,
            ManifestCholBlockSpec,
            SparseMatrixBlockSpec,
            SparseVectorBlockSpec,
            T0CholBlockSpec,
        )

        class DynamicsAwareBackend:
            checkpoint_loglik = False

            def compute_log_likelihood(
                self,
                dynamics,
                _measurement_params,
                _initial_state,
                _observations,
                time_intervals,
                **_kwargs,
            ):
                assert isinstance(dynamics, StochasticContinuousTimeStateEvolution)
                numpyro.deterministic(
                    "backend_n_vf_components",
                    jnp.asarray(len(dynamics.drift.args.params)),
                )
                numpyro.deterministic(
                    "backend_drift", dynamics.drift(jnp.ones(2), jnp.empty(0), 0.0)
                )
                return jnp.zeros_like(time_intervals)

        spec = model_fixture(
            n_latent=2,
            n_manifest=2,
            dynamics_spec=DynamicsSpec(
                n_latent=2,
                components=(*(decay_term(target=i) for i in range(2)),),
            ),
            diffusion_block=DiffusionBlockSpec(
                n_latent=2,
                diffusion_chol_support=np.zeros((2, 2), dtype=bool),
                diffusion_chol_template=jnp.eye(2) * 0.1,
            ),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=2,
                n_cols=2,
                free_support=np.zeros((2, 2), dtype=bool),
                template=jnp.eye(2),
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
            manifest_means_block=default_manifest_means_block(2),
            manifest_chol_block=ManifestCholBlockSpec(
                n_manifest=2,
                diag_support=np.zeros(2, dtype=bool),
                template=jnp.eye(2) * 0.2,
            ),
            t0_means_block=SparseVectorBlockSpec(
                n=2,
                free_support=np.zeros(2, dtype=bool),
                template=jnp.zeros(2),
                free_site_name="t0_means_free",
                det_site_name="t0_means",
                support=SupportClass.REAL,
                site_kind=SiteKind.T0_MEANS,
                assembly_group="t0",
                fixed_spec_field="t0_means",
                priors_field="t0_means",
            ),
            t0_chol_block=T0CholBlockSpec(
                n_latent=2,
                diag_support=np.zeros(2, dtype=bool),
                correlation_support=np.zeros((2, 2), dtype=bool),
                template=jnp.eye(2) * 0.3,
            ),
            static_state_sd_block=default_static_state_sd_block(),
        )
        model = SSMModel(spec, priors={"vf_0_p0": dist.Delta(0.3), "vf_1_p0": dist.Delta(0.5)})
        tr = handlers.trace(handlers.seed(model.model, rng_seed=0)).get_trace(
            observations=jnp.zeros((4, 2)),
            times=jnp.arange(4, dtype=jnp.float32),
            likelihood_backend=DynamicsAwareBackend(),
        )

        assert "vf_0_p0" in tr
        assert int(tr["backend_n_vf_components"]["value"]) == 2
        np.testing.assert_allclose(
            tr["backend_drift"]["value"],
            np.array([-0.3, -0.5]),
        )


class TestComponentNativeLinearDynamics:
    """Numerical pins for linear dynamics expressed as first-class components."""

    def test_state_decay_and_edges_match_expected_vector_field(self):
        from numpyro.handlers import condition, seed

        spec = DynamicsSpec(
            n_latent=3,
            components=(
                decay_term(target=0),
                decay_term(target=1),
                decay_term(target=2),
                linear_term(source=1, target=0),
                linear_term(source=2, target=1),
                linear_term(source=0, target=2),
            ),
        )
        compiled = compile_dynamics(spec)
        sample_values = {
            "vf_0_p0": jnp.asarray(0.3),
            "vf_1_p0": jnp.asarray(0.5),
            "vf_2_p0": jnp.asarray(0.7),
            "vf_3_p0": jnp.asarray(0.2),
            "vf_4_p0": jnp.asarray(-0.1),
            "vf_5_p0": jnp.asarray(0.4),
        }
        with (
            seed(rng_seed=0),
            condition(data=sample_values),
        ):
            params = compiled.sample_params(lambda site_name: ndist.Delta(sample_values[site_name]))

        eta = jnp.array([1.0, 2.0, 3.0])
        actual = compiled.vector_field(
            jnp.asarray(0.0),
            eta,
            VectorFieldArgs(params=params, intervention=Intervention.none()),
        )
        expected = jnp.array(
            [
                -0.3 * 1.0 + 0.2 * 2.0,
                -0.5 * 2.0 - 0.1 * 3.0,
                -0.7 * 3.0 + 0.4 * 1.0,
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_delta_state_decay_samples_component_param(self):
        from numpyro.handlers import seed

        spec = DynamicsSpec(
            n_latent=2,
            components=(decay_term(target=1),),
        )
        compiled = compile_dynamics(spec)
        with seed(rng_seed=0):
            params = compiled.sample_params(lambda _: ndist.Delta(jnp.asarray(0.4)))

        decay = params[0][scientific_id("parameter", "decay")]
        assert decay.shape == ()
        np.testing.assert_allclose(decay, 0.4, atol=1e-12)

    def test_state_intercepts_add_to_selected_targets(self):
        from numpyro.handlers import condition, seed

        spec = DynamicsSpec(
            n_latent=3,
            components=(
                intercept_term(target=0),
                intercept_term(target=2),
            ),
        )
        compiled = compile_dynamics(spec)
        sample_values = {
            "vf_0_p0": jnp.asarray(0.1),
            "vf_1_p0": jnp.asarray(-0.2),
        }
        with (
            seed(rng_seed=0),
            condition(data=sample_values),
        ):
            params = compiled.sample_params(lambda site_name: ndist.Delta(sample_values[site_name]))

        actual = compiled.vector_field(
            jnp.asarray(0.0),
            jnp.zeros(3),
            VectorFieldArgs(params=params, intervention=Intervention.none()),
        )
        np.testing.assert_allclose(actual, jnp.array([0.1, 0.0, -0.2]), atol=1e-12)
