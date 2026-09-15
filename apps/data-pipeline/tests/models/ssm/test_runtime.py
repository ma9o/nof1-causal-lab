"""Tests for SSM runtime preparation helpers.

Covers: semantic prior binding and fit-input preparation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
import pytest

from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LinkFunction
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform, SiteKind
from nof1_causal_lab.models.model_distributions import with_parameter_distributions
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.inputs import compile_priors
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.runtime import (
    build_ssm_model,
    prepare_fit_inputs,
    prepare_model_runtime,
    sample_prior_predictive,
)
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    T0CholBlockSpec,
)
from tests.helpers import (
    complete_test_model,
    make_model,
    make_prior_model,
    native_axis_metadata,
)
from tests.model_fixtures import (
    default_diffusion_block,
    default_lambda_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
    full_dense_matrix_dynamics_spec,
    full_diagonal_support,
    model_fixture,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.model import SSMModel
    from nof1_causal_lab.sampler_config import SamplerConfigOverride

# =============================================================================
# normalize_prior_params
# =============================================================================


def _make_spec(
    *,
    n_latent: int = 1,
    n_manifest: int = 1,
    dynamics_spec=None,
    diffusion_block=None,
    lambda_block=None,
    manifest_means_block=None,
    manifest_chol_block=None,
    t0_means_block=None,
    t0_chol_block=None,
    static_state_sd_block=None,
    **kwargs,
) -> ModelSpec:
    """Build an ModelSpec from explicit block specs for tests."""
    if dynamics_spec is None:
        dynamics_spec = full_dense_matrix_dynamics_spec(n_latent)
    return model_fixture(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dynamics_spec,
        diffusion_block=diffusion_block or default_diffusion_block(n_latent),
        lambda_block=lambda_block or default_lambda_block(n_manifest, n_latent),
        manifest_means_block=manifest_means_block or default_manifest_means_block(n_manifest),
        manifest_chol_block=manifest_chol_block or default_manifest_chol_block(n_manifest),
        t0_means_block=t0_means_block or default_t0_means_block(n_latent),
        t0_chol_block=t0_chol_block or default_t0_chol_block(n_latent),
        static_state_sd_block=static_state_sd_block or default_static_state_sd_block(),
        **native_axis_metadata(n_latent, n_manifest, kwargs),
    )


class TestBuilderPriorConversion:
    def test_ar_prior_rejects_negative_support(self):
        model = complete_test_model(make_model(["mood"]))
        _spec, _ = (model, numeric.edge_lag_days(model))
        with pytest.raises(ValueError, match=r"support within \[0, 1\]"):
            compile_priors(
                make_prior_model(
                    model,
                    {
                        "rho_mood": {
                            "distribution": "Uniform",
                            "params": {"lower": -1.0, "upper": 1.0},
                        }
                    },
                )
            )

    def test_initial_state_correlation_priors_are_bounded_to_correlation_scale(self):
        model = _make_spec(n_latent=2, n_manifest=2, dynamics_spec=DynamicsSpec(2, ()))
        correlation = next(
            p
            for p in model.parameters
            if model.parameter_context(p.id).quantity == SiteKind.T0_VAR_LOWER
        )
        model = model.revised(
            parameters=tuple(
                p.model_copy(
                    update={
                        "distribution_transform": PriorAuthoringTransform.INITIAL_STATE_CORRELATION,
                    }
                )
                if p.id == correlation.id
                else p
                for p in model.parameters
            )
        )
        model = with_parameter_distributions(model, {correlation.id: dist.Normal(0.2, 0.8)})
        law = compile_priors(model)[0]["t0_var_lower_free"]
        np.testing.assert_allclose(law.base_dist.loc, [0.2])
        np.testing.assert_allclose(law.base_dist.scale, [0.8])
        np.testing.assert_allclose(law.low, [-1.0])
        np.testing.assert_allclose(law.high, [1.0])

    def test_initial_state_mean_and_sd_priors_bind_to_t0_sites(self):
        model = _make_spec(n_latent=2, n_manifest=2, dynamics_spec=DynamicsSpec(2, ()))
        means = [
            p
            for p in model.parameters
            if model.parameter_context(p.id).quantity == SiteKind.T0_MEANS
        ]
        scales = [
            p
            for p in model.parameters
            if model.parameter_context(p.id).quantity == SiteKind.T0_VAR_DIAG
        ]
        laws = {
            means[0].id: dist.Normal(0.2, 0.3),
            means[1].id: dist.Normal(0.4, 0.5),
            scales[0].id: dist.HalfNormal(0.7),
            scales[1].id: dist.HalfNormal(0.9),
        }
        model = with_parameter_distributions(model, laws)
        priors, bindings, _ = compile_priors(model)
        np.testing.assert_allclose(priors["t0_means_free"].loc, [0.2, 0.4])
        np.testing.assert_allclose(priors["t0_var_diag_free"].scale, [0.7, 0.9])
        assert [bindings.by_parameter[p.id].flat_index for p in means] == [0, 1]

    def test_initial_state_correlation_prior_indices_are_dense_after_mask_filtering(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 1] = True
        model = _make_spec(
            n_latent=3,
            n_manifest=3,
            dynamics_spec=DynamicsSpec(3, ()),
            t0_chol_block=T0CholBlockSpec(
                n_latent=3,
                diag_support=np.ones(3, dtype=bool),
                correlation_support=mask,
                template=jnp.eye(3),
            ),
        )
        _, bindings, _ = compile_priors(model)
        correlation = next(
            p
            for p in model.parameters
            if model.parameter_context(p.id).quantity == SiteKind.T0_VAR_LOWER
        )
        assert bindings.by_parameter[correlation.id].flat_index == 0
        assert numeric.initial_covariance_block(model).correlation_positions == [(2, 1)]

    def test_component_dynamics_parameters_bind_to_their_own_terms(self):
        model = complete_test_model(make_model(["stress", "mood"], [("stress", "mood")]))
        _, bindings, _ = compile_priors(model)
        for parameter in model.parameters:
            if any(
                owner.kind == "mechanism" for owner in model.parameter_context(parameter.id).owners
            ):
                assert bindings.by_parameter[parameter.id].component_index is not None

    def test_cross_lag_prior_requires_the_declared_measurement_clock(self):
        model = complete_test_model(make_model(["stress", "mood"], [("stress", "mood")])).revised(
            measurement_clock=None
        )
        with pytest.raises(ValueError, match="measurement clock"):
            compile_priors(model)


class TestObservationSupportValidation:
    def test_gamma_emission_rejects_zero_observations(self):
        """Gamma likelihoods must fail early when observed data include zeros."""
        X = pl.DataFrame({"time": [0, 1, 2], "screen_gap": [0.0, 1.0, 2.0]})
        spec = _make_spec(
            n_latent=1,
            n_manifest=1,
            latent_names=["screen_gap"],
            manifest_names=["screen_gap"],
            manifest_dists=[DistributionFamily.GAMMA],
            manifest_links=[LinkFunction.LOG],
        )

        with pytest.raises(ValueError, match="Observation support check failed"):
            build_ssm_model(X, model_spec=spec)


class TestPrepareFitInputs:
    def test_sparse_wide_nulls_become_nan_without_fill_forward(self):
        """Sparse wide cells should stay missing and never broadcast across ticks."""
        spec = _make_spec(n_latent=2, n_manifest=2, manifest_names=["x", "y"])
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "x": [10.0, None],
                "y": [None, 30.0],
            }
        )

        observations, times, manifest_names, _wide = prepare_fit_inputs(spec, wide)

        assert manifest_names == ["x", "y"]
        assert jnp.allclose(times, jnp.array([0.0, 1.0], dtype=jnp.float32))
        assert jnp.isclose(observations[0, 0], 10.0)
        assert jnp.isnan(observations[0, 1])
        assert jnp.isnan(observations[1, 0])
        assert jnp.isclose(observations[1, 1], 30.0)

    def test_manifest_standardization_applies_only_to_standardized_channels(self):
        """prepare_fit_inputs should deterministically standardize only marked manifests."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            manifest_names=["x", "y"],
            manifest_standardized=[True, False],
        )
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "x": [10.0, 12.0, 14.0],
                "y": [5.0, 6.0, 7.0],
            }
        )

        observations, times, manifest_names, _wide = prepare_fit_inputs(spec, wide)

        assert manifest_names == ["x", "y"]
        np.testing.assert_allclose(np.asarray(times), np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(
            np.asarray(observations[:, 0]), np.array([-1.0, 0.0, 1.0]), rtol=1e-6
        )
        np.testing.assert_allclose(np.asarray(observations[:, 1]), np.array([5.0, 6.0, 7.0]))

    def test_manifest_standardization_of_constant_column_centers_without_scaling(self):
        """A zero-variance standardized column becomes exactly zero (divisor 1)."""
        spec = _make_spec(
            n_latent=1,
            n_manifest=1,
            manifest_names=["x"],
            manifest_standardized=[True],
        )
        wide = pl.DataFrame({"time": [0.0, 1.0], "x": [4.2, 4.2]})

        observations, _times, _names, _wide = prepare_fit_inputs(spec, wide)

        np.testing.assert_allclose(np.asarray(observations[:, 0]), np.array([0.0, 0.0]))


class TestPrepareModelRuntime:
    def test_preserves_long_observation_metadata_and_augments_support_boundaries(self, caplog):
        data_for_model = pl.DataFrame(
            {
                "indicator_id": ["indicator:3696aef3ff6f446744e5"],
                "value": [1.0],
                "anchor_time": ["2024-02-01T00:00:00"],
                "support_kind": ["interval"],
                "summary_operator": ["mean"],
                "anchor_policy": ["support_end"],
                "observation_window": ["1mo"],
                "support_start": ["2024-01-01T00:00:00"],
                "support_end": ["2024-02-01T00:00:00"],
            }
        )

        class StubModel:
            def __init__(self):
                self.observation_support = None
                self.spec = _make_spec(
                    n_latent=1,
                    n_manifest=1,
                    manifest_names=["stress_score"],
                )
                self.parameter_layout = object()

            def set_observation_support(self, observation_support):
                self.observation_support = observation_support

        with caplog.at_level("INFO"):
            runtime = prepare_model_runtime(
                data_for_model,
                model_spec=(cast("SSMModel", StubModel())).spec,
                model=cast("SSMModel", StubModel()),
                sampler_config=cast(
                    "SamplerConfigOverride",
                    {"method": "marginal_particle_gibbs"},
                ),
            )

        assert runtime.observation_data is not None
        assert (
            runtime.observation_data.columns
            == data_for_model.rename({"indicator_id": "indicator"}).columns
        )
        assert runtime.observation_data["observation_window"][0] == "1mo"
        assert runtime.observation_data["support_end"][0] == "2024-02-01T00:00:00"
        assert runtime.observation_data["anchor_time"][0] == "2024-02-01T00:00:00"
        assert runtime.wide_data["time"].to_list() == [-31.0, 0.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.manifest_names == ["stress_score"]
        assert runtime.observation_support.support_kinds == ["interval"]
        assert runtime.observation_support.summary_operators == ["mean"]
        assert runtime.observation_support.anchor_policies == ["support_end"]
        assert runtime.observation_support.observation_windows == ["1mo"]
        assert runtime.observation_support.requires_interval_summary_handling is True
        assert runtime.observation_support.interval_summary_manifest_names == ["stress_score"]
        assert runtime.observation_support.support_start_times.shape == (2, 1)
        assert runtime.observation_support.support_end_times.shape == (2, 1)
        assert runtime.observation_support.support_start_times[1, 0] == pytest.approx(-31.0)
        assert runtime.observation_support.support_end_times[1, 0] == pytest.approx(0.0)
        assert runtime.observation_support.interval_prev_coeffs.shape == (2, 1, 1)
        assert runtime.observation_support.interval_curr_coeffs.shape == (2, 1, 1)
        assert runtime.observation_support.interval_weights.shape == (2, 1, 1)
        assert runtime.observation_support.emission_slot_indices.tolist() == [[-1], [0]]
        assert runtime.observation_support.interval_prev_coeffs[1, 0, 0] == pytest.approx(15.5)
        assert runtime.observation_support.interval_curr_coeffs[1, 0, 0] == pytest.approx(15.5)
        assert runtime.observation_support.interval_weights[1, 0, 0] == pytest.approx(31.0)
        assert numeric.observation_names(runtime.spec) == ["stress_score"]
        assert runtime.model.observation_support is runtime.observation_support
        assert runtime.inference_structure.structural_backend == "laplace"
        assert runtime.inference_structure.resolved_method == "marginal_particle_gibbs"
        assert runtime.inference_structure.method_override == "marginal_particle_gibbs"
        assert "support-aware observation semantics" in caplog.text

    def test_compiles_overlapping_interval_windows_into_concurrent_slots(self):
        data_for_model = pl.DataFrame(
            {
                "indicator_id": [
                    "indicator:3696aef3ff6f446744e5",
                    "indicator:3696aef3ff6f446744e5",
                ],
                "value": [3.0, 5.0],
                "anchor_time": ["2024-01-03T00:00:00", "2024-01-04T00:00:00"],
                "support_kind": ["interval", "interval"],
                "summary_operator": ["mean", "mean"],
                "anchor_policy": ["support_end", "support_end"],
                "observation_window": ["2d", "2d"],
                "support_start": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
                "support_end": ["2024-01-03T00:00:00", "2024-01-04T00:00:00"],
            }
        )

        class StubModel:
            def __init__(self):
                self.observation_support = None
                self.spec = _make_spec(
                    n_latent=1,
                    n_manifest=1,
                    manifest_names=["stress_score"],
                )
                self.parameter_layout = object()

            def set_observation_support(self, observation_support):
                self.observation_support = observation_support

        runtime = prepare_model_runtime(
            data_for_model,
            model_spec=(cast("SSMModel", StubModel())).spec,
            model=cast("SSMModel", StubModel()),
            sampler_config=cast(
                "SamplerConfigOverride",
                {"method": "marginal_particle_gibbs"},
            ),
        )

        assert runtime.wide_data["time"].to_list() == [-2.0, -1.0, 0.0, 1.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.max_active_windows == 2
        assert runtime.inference_structure.structural_backend == "laplace"
        assert runtime.inference_structure.resolved_method == "marginal_particle_gibbs"
        assert runtime.observation_support.emission_slot_indices.tolist() == [[-1], [-1], [0], [1]]
        assert runtime.observation_support.interval_weights.shape == (4, 1, 2)
        assert runtime.observation_support.interval_weights[1, 0, 0] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[2, 0, 0] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[2, 0, 1] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[3, 0, 1] == pytest.approx(1.0)

    @pytest.mark.predictive
    def test_prior_predictive_reuses_prepared_support_schedule(self):
        data_for_model = pl.DataFrame(
            {
                "indicator_id": ["indicator:3696aef3ff6f446744e5"],
                "value": [1.0],
                "anchor_time": ["2024-02-01T00:00:00"],
                "support_kind": ["interval"],
                "summary_operator": ["mean"],
                "anchor_policy": ["support_end"],
                "observation_window": ["1mo"],
                "support_start": ["2024-01-01T00:00:00"],
                "support_end": ["2024-02-01T00:00:00"],
            }
        )
        model = build_ssm_model(
            pl.DataFrame({"time": [0.0], "stress_score": [1.0]}),
            model_spec=_make_spec(
                n_latent=1,
                n_manifest=1,
                diffusion_block=DiffusionBlockSpec(
                    n_latent=1,
                    diffusion_chol_support=np.diag(full_diagonal_support(1)),
                    diffusion_chol_template=jnp.eye(1, dtype=jnp.float32),
                ),
                manifest_names=["stress_score"],
            ),
        )
        runtime = prepare_model_runtime(
            data_for_model,
            model_spec=(model).spec,
            model=model,
            sampler_config=cast(
                "SamplerConfigOverride",
                {"method": "marginal_particle_gibbs"},
            ),
        )

        samples = sample_prior_predictive(
            runtime.model,
            samples=3,
            times=runtime.times,
            observation_support=runtime.observation_support,
            observation_mask=~jnp.isnan(runtime.observations),
        )

        assert samples["observations"].shape == (3, 2, 1)
        assert samples["observations_mask"].shape == (3, 2, 1)
        assert jnp.isnan(samples["observations"][:, 0, 0]).all()
        assert jnp.isfinite(samples["observations"][:, 1, 0]).all()
        assert (~samples["observations_mask"][:, 0, 0]).all()
        assert samples["observations_mask"][:, 1, 0].all()
