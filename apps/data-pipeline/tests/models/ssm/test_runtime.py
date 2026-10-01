"""Tests for SSM runtime preparation helpers.

Covers: semantic prior binding and fit-input preparation.
"""

from __future__ import annotations

from pathlib import Path

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
import pytest

from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LinkFunction
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform, SiteKind
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.inputs import compile_priors
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.runtime import (
    build_ssm_model,
    prepare_fit_inputs,
    prepare_model_runtime,
    project_observation_data,
)
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    T0CholBlockSpec,
)
from tests.dynamics_fixtures import decay_term
from tests.helpers import make_model, native_axis_metadata
from tests.model_fixtures import compile_fit_fixture, default_diffusion_block, default_lambda_block, default_manifest_chol_block, default_manifest_means_block, default_static_state_sd_block, default_t0_chol_block, default_t0_means_block, full_dense_matrix_dynamics_spec, full_diagonal_support

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.sampler_config import SamplerConfigOverride

# =============================================================================
# normalize_prior_params
# =============================================================================




@pytest.mark.contract
class TestBuilderPriorConversion:
    def test_ar_prior_rejects_negative_support(self):
        with pytest.raises(ValueError, match=r"support within \[0, 1\]"):
            compile_priors(
                ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testbuilderpriorconversion_test_ar_prior_rejects_negative_support_make_prior_model.json').read_text())
            )

    def test_initial_state_correlation_priors_are_bounded_to_correlation_scale(self):
        model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testbuilderpriorconversion_test_initial_state_correlation_priors_are_bounded_to_correlation_scale_with_parameter_distributions.json').read_text())
        law = compile_priors(model)[0]["t0_var_lower_free"]
        np.testing.assert_allclose(law.base_dist.loc, [0.2])
        np.testing.assert_allclose(law.base_dist.scale, [0.8])
        np.testing.assert_allclose(law.low, [-1.0])
        np.testing.assert_allclose(law.high, [1.0])

    def test_initial_state_mean_and_sd_priors_bind_to_t0_sites(self):
        model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testbuilderpriorconversion_test_initial_state_mean_and_sd_priors_bind_to_t0_sites__make_spec.json').read_text())
        means = [
            p
            for p in model.parameters
            if model.parameter_context(p.id).quantity == SiteKind.T0_MEANS
        ]
        model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testbuilderpriorconversion_test_initial_state_mean_and_sd_priors_bind_to_t0_sites_with_parameter_distributions.json').read_text())
        priors, bindings, _ = compile_priors(model)
        np.testing.assert_allclose(priors["t0_means_free"].loc, [0.2, 0.4])
        np.testing.assert_allclose(priors["t0_var_diag_free"].scale, [0.7, 0.9])
        assert [bindings.by_parameter[p.id].flat_index for p in means] == [0, 1]

    def test_initial_state_correlation_prior_indices_are_dense_after_mask_filtering(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 1] = True
        model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testbuilderpriorconversion_test_initial_state_correlation_prior_indices_are_dense_after_mask_filtering__make_spec.json').read_text())
        _, bindings, _ = compile_priors(model)
        correlation = next(
            p
            for p in model.parameters
            if model.parameter_context(p.id).quantity == SiteKind.T0_VAR_LOWER
        )
        assert bindings.by_parameter[correlation.id].flat_index == 0
        assert numeric.initial_covariance_block(model).correlation_positions == [(2, 1)]

    def test_component_dynamics_parameters_bind_to_their_own_terms(self):
        model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testbuilderpriorconversion_test_component_dynamics_parameters_bind_to_their_own_terms_complete_test_model.json').read_text())
        _, bindings, _ = compile_priors(model)
        for parameter in model.parameters:
            if any(
                owner.kind == "mechanism" for owner in model.parameter_context(parameter.id).owners
            ):
                assert bindings.by_parameter[parameter.id].component_index is not None

    def test_cross_lag_prior_requires_the_declared_measurement_clock(self):
        model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testbuilderpriorconversion_test_cross_lag_prior_requires_the_declared_measurement_clock_complete_test_model.json').read_text()).revised(
            measurement_clock=None
        )
        with pytest.raises(ValueError, match="measurement clock"):
            compile_priors(model)


@pytest.mark.contract
class TestObservationSupportValidation:
    def test_gamma_emission_rejects_zero_observations(self):
        """Gamma likelihoods must fail early when observed data include zeros."""
        X = pl.DataFrame({"time": [0, 1, 2], "screen_gap": [0.0, 1.0, 2.0]})
        spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testobservationsupportvalidation_test_gamma_emission_rejects_zero_observations__make_spec.json').read_text())

        with pytest.raises(ValueError, match="Observation support check failed"):
            build_ssm_model(X, inputs=compile_fit_fixture(spec))


@pytest.mark.contract
class TestPrepareFitInputs:
    def test_sparse_wide_nulls_become_nan_without_fill_forward(self):
        """Sparse wide cells should stay missing and never broadcast across ticks."""
        spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testpreparefitinputs_test_sparse_wide_nulls_become_nan_without_fill_forward__make_spec.json').read_text())
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
        spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testpreparefitinputs_test_manifest_standardization_applies_only_to_standardized_channels__make_spec.json').read_text())
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
        spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testpreparefitinputs_test_manifest_standardization_of_constant_column_centers_without_scaling__make_spec.json').read_text())
        wide = pl.DataFrame({"time": [0.0, 1.0], "x": [4.2, 4.2]})

        observations, _times, _names, _wide = prepare_fit_inputs(spec, wide)

        np.testing.assert_allclose(np.asarray(observations[:, 0]), np.array([0.0, 0.0]))


class TestPrepareModelRuntime:
    @pytest.mark.contract
    def test_selected_indicators_keep_the_panel_origin_and_initial_grid_point(self):
        from nof1_causal_lab.models.ssm.observation_support import (
            augment_wide_data_with_support_boundaries,
        )
        from nof1_causal_lab.utils.observation_rows import prepared_time_origin

        early = make_model(["early"])
        late = make_model(["late"])
        rows = pl.DataFrame(
            {
                "indicator_id": [early.indicators[0].id, late.indicators[0].id],
                "value": [1.0, 2.0],
                "anchor_time": [datetime(2024, 1, 2), datetime(2024, 1, 12)],
                "support_start": [datetime(2024, 1, 1), datetime(2024, 1, 11)],
                "support_end": [datetime(2024, 1, 2), datetime(2024, 1, 12)],
                "support_kind": ["point", "point"],
            }
        )
        origin = prepared_time_origin(rows, None)
        assert origin == datetime(2024, 1, 1, tzinfo=UTC)
        for model, expected in ((early, 1.0), (late, 11.0)):
            wide, selected = project_observation_data(rows, model_spec=model, time_origin=origin)
            assert wide["time"].to_list() == [expected]
            augmented = augment_wide_data_with_support_boundaries(
                selected, wide, [model.indicators[0].name], time_origin=origin
            )
            assert augmented["time"].to_list() == [0.0, expected]
            assert augmented[model.indicators[0].name][0] is None

    @pytest.mark.contract
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

        inputs = compile_fit_fixture(
            ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testpreparemodelruntime_test_preserves_long_observation_metadata_and_augments_support_boundaries__make_spec.json').read_text())
        )

        with caplog.at_level("INFO"):
            runtime = prepare_model_runtime(
                data_for_model,
                time_origin=datetime(2024, 1, 1, tzinfo=UTC),
                inputs=inputs,
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
        assert runtime.wide_data["time"].to_list() == [0.0, 31.0]
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
        assert runtime.observation_support.support_start_times[1, 0] == pytest.approx(0.0)
        assert runtime.observation_support.support_end_times[1, 0] == pytest.approx(31.0)
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

    @pytest.mark.contract
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

        inputs = compile_fit_fixture(
            ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testpreparemodelruntime_test_compiles_overlapping_interval_windows_into_concurrent_slots__make_spec.json').read_text())
        )

        runtime = prepare_model_runtime(
            data_for_model,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            inputs=inputs,
            sampler_config=cast(
                "SamplerConfigOverride",
                {"method": "marginal_particle_gibbs"},
            ),
        )

        assert runtime.wide_data["time"].to_list() == [0.0, 1.0, 2.0, 3.0]
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

    @pytest.mark.inference(concern="predictive")
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
            inputs=compile_fit_fixture(
                ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/testpreparemodelruntime_test_prior_predictive_reuses_prepared_support_schedule__make_spec.json').read_text())
            ),
        )
        runtime = prepare_model_runtime(
            data_for_model,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            inputs=model.inputs,
            sampler_config=cast(
                "SamplerConfigOverride",
                {"method": "marginal_particle_gibbs"},
            ),
        )

        from nof1_causal_lab.artifacts.simulation import SimulationSpec
        from nof1_causal_lab.models.ssm.predictive.simulation import generate_simulation_batch

        samples = generate_simulation_batch(
            runtime.model.spec,
            SimulationSpec(start=float(runtime.times[0]), end=float(runtime.times[-1])),
            times=runtime.times,
            draws=3,
            comparison_data=data_for_model,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        ).prediction

        assert samples.trajectory.observations.shape == (3, 2, 1)
        assert samples.trajectory.observations_mask.shape == (3, 2, 1)
        assert jnp.isnan(samples.trajectory.observations[:, 0, 0]).all()
        assert jnp.isfinite(samples.trajectory.observations[:, 1, 0]).all()
        assert (~samples.trajectory.observations_mask[:, 0, 0]).all()
        assert samples.trajectory.observations_mask[:, 1, 0].all()


@pytest.mark.contract
def test_compiled_inputs_own_runtime_derivations(monkeypatch):
    from nof1_causal_lab.models.ssm.compile import prior_compilation
    from nof1_causal_lab.models.ssm.model import SSMModel

    inputs = compile_fit_fixture(ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/compiled_inputs_own_runtime_derivations__make_spec.json').read_text()))

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("runtime recompiled its evidence")

    monkeypatch.setattr(prior_compilation, "compile_priors", unexpected_compile)
    monkeypatch.setattr(prior_compilation, "bind_parameters", unexpected_compile)
    model = SSMModel(inputs)
    assert model.spec is inputs.spec
    assert model.parameter_bindings is inputs.bindings
    assert model.parameter_layout is inputs.parameter_layout
    assert model.get_prior_runtime_bundle() is inputs.prior_runtime_bundle


@pytest.mark.contract
def test_compile_distinguishes_incomplete_unsupported_and_bugs(monkeypatch):
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.compile import inputs as compiler

    incomplete = compiler.compile_ssm_inputs_from_model(ModelSpec())
    assert isinstance(incomplete, compiler.IncompleteModel)
    spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/compile_distinguishes_incomplete_unsupported_and_bugs__make_spec.json').read_text())
    unsupported = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'runtime/compile_distinguishes_incomplete_unsupported_and_bugs_with_parameter_distributions.json').read_text())
    assert isinstance(compiler.compile_ssm_inputs_from_model(unsupported), compiler.UnsupportedFit)

    def broken_compiler(_model):
        raise ValueError("internal compiler bug")

    monkeypatch.setattr(compiler, "compile_priors", broken_compiler)
    with pytest.raises(ValueError, match="internal compiler bug"):
        compiler.compile_ssm_inputs_from_model(spec)


@pytest.mark.contract
def test_fit_resolves_incomplete_model_before_panel_preparation(monkeypatch):
    from nof1_causal_lab.actions.inference import fit as fitting
    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    def unexpected_panel(*_args, **_kwargs):
        raise AssertionError("panel prepared before fit capability was resolved")

    monkeypatch.setattr(fitting, "prepare_model_runtime", unexpected_panel)
    result = fitting.fit_model(ModelSpec(), pl.DataFrame(), time_origin=None)
    assert not result["fitted"]
    assert result["error"]
