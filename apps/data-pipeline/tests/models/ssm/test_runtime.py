"""Tests for SSM runtime preparation helpers.

Covers: semantic prior binding and fit-input preparation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.inputs import compile_priors
from nof1_causal_lab.models.ssm.runtime import (
    BoundPanel,
    bind_panel,
    prepare_fit_inputs,
    project_observation_data,
)
from nof1_causal_lab.sampler_config import SamplerSpec
from tests.model_fixtures import (
    bind_panel_fixture,
    compile_fit_fixture,
    compile_model_fixture,
)

# =============================================================================
# normalize_prior_params
# =============================================================================


@pytest.mark.contract
class TestBuilderPriorConversion:
    def test_ar_prior_rejects_negative_support(self):
        with pytest.raises(ValueError, match=r"support within \[0, 1\]"):
            compile_priors(
                compile_model_fixture(
                    ModelSpec.model_validate_json(
                        (
                            Path(__file__).resolve().parents[2]
                            / "fixtures/models"
                            / "runtime/testbuilderpriorconversion_test_ar_prior_rejects_negative_support_make_prior_model.json"
                        ).read_text()
                    )
                ),
                ModelSpec.model_validate_json(
                    (
                        Path(__file__).resolve().parents[2]
                        / "fixtures/models"
                        / "runtime/testbuilderpriorconversion_test_ar_prior_rejects_negative_support_make_prior_model.json"
                    ).read_text()
                ),
            )

    def test_initial_state_correlation_priors_are_bounded_to_correlation_scale(self):
        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "runtime/testbuilderpriorconversion_test_initial_state_correlation_priors_are_bounded_to_correlation_scale_with_parameter_distributions.json"
            ).read_text()
        )
        law = compile_priors(compile_model_fixture(model), model)[0]["t0_var_lower_free"]
        np.testing.assert_allclose(law.base_dist.loc, [0.2])
        np.testing.assert_allclose(law.base_dist.scale, [0.8])
        np.testing.assert_allclose(law.low, [-1.0])
        np.testing.assert_allclose(law.high, [1.0])

    def test_initial_state_mean_and_sd_priors_bind_to_t0_sites(self):
        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "common/two_state_fixed_drift_model.json"
            ).read_text()
        )
        means = [
            p
            for p in model.parameters
            if model.parameter_context(p.id).quantity == SiteKind.T0_MEANS
        ]
        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "runtime/testbuilderpriorconversion_test_initial_state_mean_and_sd_priors_bind_to_t0_sites_with_parameter_distributions.json"
            ).read_text()
        )
        priors, bindings, _ = compile_priors(compile_model_fixture(model), model)
        np.testing.assert_allclose(priors["t0_means_free"].loc, [0.2, 0.4])
        np.testing.assert_allclose(priors["t0_var_diag_free"].scale, [0.7, 0.9])
        assert [
            {binding.parameter_id: binding for binding in bindings}[p.id].flat_index for p in means
        ] == [0, 1]

    def test_initial_state_correlation_prior_indices_are_dense_after_mask_filtering(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 1] = True
        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "runtime/testbuilderpriorconversion_test_initial_state_correlation_prior_indices_are_dense_after_mask_filtering__make_spec.json"
            ).read_text()
        )
        _, bindings, _ = compile_priors(compile_model_fixture(model), model)
        correlation = next(
            p
            for p in model.parameters
            if model.parameter_context(p.id).quantity == SiteKind.T0_VAR_LOWER
        )
        assert {binding.parameter_id: binding for binding in bindings}[
            correlation.id
        ].flat_index == 0
        assert compile_model_fixture(model).initial_covariance_block.correlation_positions == [
            (2, 1)
        ]

    def test_component_dynamics_parameters_bind_to_their_own_terms(self):
        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "runtime/stress_mood_model.json"
            ).read_text()
        )
        _, bindings, _ = compile_priors(compile_model_fixture(model), model)
        for parameter in model.parameters:
            if any(
                owner.kind == "mechanism" for owner in model.parameter_context(parameter.id).owners
            ):
                assert {binding.parameter_id: binding for binding in bindings}[
                    parameter.id
                ].component_index is not None

    def test_cross_lag_prior_requires_the_declared_measurement_clock(self):
        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "runtime/stress_mood_model.json"
            ).read_text()
        ).revised(measurement_clock=None)
        with pytest.raises(ValueError, match="measurement clock"):
            compile_priors(compile_model_fixture(model), model)


@pytest.mark.contract
class TestObservationSupportValidation:
    def test_gamma_emission_rejects_zero_observations(self):
        """Gamma likelihoods must fail early when observed data include zeros."""
        X = pl.DataFrame({"time": [0, 1, 2], "screen_gap": [0.0, 1.0, 2.0]})
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "runtime/testobservationsupportvalidation_test_gamma_emission_rejects_zero_observations__make_spec.json"
            ).read_text()
        )

        from nof1_causal_lab.models.ssm.observation_support import validate_observation_support

        with pytest.raises(ValueError, match="Observation support check failed"):
            validate_observation_support(compile_model_fixture(spec), X)


@pytest.mark.contract
class TestPrepareFitInputs:
    def test_sparse_wide_nulls_become_nan_without_fill_forward(self):
        """Sparse wide cells should stay missing and never broadcast across ticks."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "runtime/testpreparefitinputs_test_sparse_wide_nulls_become_nan_without_fill_forward__make_spec.json"
            ).read_text()
        )
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "x": [10.0, None],
                "y": [None, 30.0],
            }
        )

        observations, times, manifest_names, _wide = prepare_fit_inputs(
            compile_model_fixture(spec), wide
        )

        assert manifest_names == ("x", "y")
        assert jnp.allclose(times, jnp.array([0.0, 1.0], dtype=jnp.float32))
        assert jnp.isclose(observations[0, 0], 10.0)
        assert jnp.isnan(observations[0, 1])
        assert jnp.isnan(observations[1, 0])
        assert jnp.isclose(observations[1, 1], 30.0)

    def test_manifest_standardization_applies_only_to_standardized_channels(self):
        """prepare_fit_inputs should deterministically standardize only marked manifests."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "runtime/testpreparefitinputs_test_manifest_standardization_applies_only_to_standardized_channels__make_spec.json"
            ).read_text()
        )
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "x": [10.0, 12.0, 14.0],
                "y": [5.0, 6.0, 7.0],
            }
        )

        observations, times, manifest_names, _wide = prepare_fit_inputs(
            compile_model_fixture(spec), wide
        )

        assert manifest_names == ("x", "y")
        np.testing.assert_allclose(np.asarray(times), np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(
            np.asarray(observations[:, 0]), np.array([-1.0, 0.0, 1.0]), rtol=1e-6
        )
        np.testing.assert_allclose(np.asarray(observations[:, 1]), np.array([5.0, 6.0, 7.0]))

    def test_manifest_standardization_of_constant_column_centers_without_scaling(self):
        """A zero-variance standardized column becomes exactly zero (divisor 1)."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "runtime/testpreparefitinputs_test_manifest_standardization_of_constant_column_centers_without_scaling__make_spec.json"
            ).read_text()
        )
        wide = pl.DataFrame({"time": [0.0, 1.0], "x": [4.2, 4.2]})

        observations, _times, _names, _wide = prepare_fit_inputs(compile_model_fixture(spec), wide)

        np.testing.assert_allclose(np.asarray(observations[:, 0]), np.array([0.0, 0.0]))


class TestPrepareModelRuntime:
    @pytest.mark.contract
    def test_selected_indicators_keep_the_panel_origin_and_initial_grid_point(self):
        from nof1_causal_lab.models.ssm.observation_support import (
            augment_wide_data_with_support_boundaries,
        )
        from nof1_causal_lab.utils.observation_rows import prepared_time_origin

        early = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models/common/one_state_gaussian_model.json"
            ).read_text()
        )
        late = ModelSpec.model_validate_json(
            early.model_dump_json()
            .replace(str(early.indicators[0].observation.id), "indicator:late")
            .replace(early.indicators[0].observation.name, "late_obs")
        )
        rows = pl.DataFrame(
            {
                "indicator_id": [
                    early.indicators[0].observation.id,
                    late.indicators[0].observation.id,
                ],
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
            wide, selected = project_observation_data(
                rows, model_spec=compile_model_fixture(model), time_origin=origin
            )
            assert wide["time"].to_list() == [expected]
            augmented = augment_wide_data_with_support_boundaries(
                selected, wide, time_origin=origin
            )
            assert augmented["time"].to_list() == [0.0, expected]
            assert augmented[model.indicators[0].observation.name][0] is None

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
            ModelSpec.model_validate_json(
                (
                    Path(__file__).resolve().parents[2]
                    / "fixtures/models"
                    / "runtime/stress_interval_model.json"
                ).read_text()
            )
        )

        with caplog.at_level("INFO"):
            runtime = bind_panel(
                data_for_model,
                time_origin=datetime(2024, 1, 1, tzinfo=UTC),
                model=inputs.compiled,
            )

        assert isinstance(runtime, BoundPanel)
        assert runtime.rows.column_names == data_for_model.columns
        assert pl.DataFrame(runtime.rows)["observation_window"][0] == "1mo"
        assert pl.DataFrame(runtime.rows)["support_end"][0] == "2024-02-01T00:00:00"
        assert pl.DataFrame(runtime.rows)["anchor_time"][0] == "2024-02-01T00:00:00"
        assert isinstance(runtime, BoundPanel)
        assert runtime.times.tolist() == [0.0, 31.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.manifest_names == ("stress_score",)
        assert runtime.observation_support.support_kinds == ("interval",)
        assert runtime.observation_support.summary_operators == ("mean",)
        assert runtime.observation_support.anchor_policies == ("support_end",)
        assert runtime.observation_support.observation_windows == ("1mo",)
        assert runtime.observation_support.requires_interval_summary_handling is True
        assert runtime.observation_support.interval_summary_manifest_names == ("stress_score",)
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
        assert numeric.observation_names(runtime.model) == ("stress_score",)

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
            ModelSpec.model_validate_json(
                (
                    Path(__file__).resolve().parents[2]
                    / "fixtures/models"
                    / "runtime/stress_interval_model.json"
                ).read_text()
            )
        )

        runtime = bind_panel(
            data_for_model,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            model=inputs.compiled,
        )

        assert isinstance(runtime, BoundPanel)
        assert runtime.times.tolist() == [0.0, 1.0, 2.0, 3.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.max_active_windows == 2
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
        model = compile_fit_fixture(
            ModelSpec.model_validate_json(
                (
                    Path(__file__).resolve().parents[2]
                    / "fixtures/models"
                    / "runtime/stress_interval_model.json"
                ).read_text()
            )
        )
        runtime = bind_panel(
            data_for_model,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            model=model.compiled,
        )

        from nof1_causal_lab.artifacts.simulation import SimulationSpec
        from nof1_causal_lab.models.ssm.predictive.simulation import generate_simulation_batch

        assert isinstance(runtime, BoundPanel)
        samples = generate_simulation_batch(
            runtime,
            SimulationSpec(start=float(runtime.times[0]), end=float(runtime.times[-1])),
            draws=3,
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

    inputs = compile_fit_fixture(
        ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "common/one_state_gaussian_model.json"
            ).read_text()
        )
    )

    def unexpected_compile(*_args, **_kwargs):
        raise AssertionError("runtime recompiled its evidence")

    monkeypatch.setattr(prior_compilation, "compile_priors", unexpected_compile)
    monkeypatch.setattr(prior_compilation, "bind_parameters", unexpected_compile)
    panel = bind_panel_fixture(inputs.compiled, jnp.zeros((2, 1)), jnp.arange(2.0))
    assert panel.model is inputs.compiled
    assert panel.indicator_ids == tuple(
        observation.id for observation in inputs.compiled.observations
    )
    assert panel.rows.column("indicator_id").to_pylist() == [str(panel.indicator_ids[0])] * 2
    with pytest.raises(AttributeError):
        panel.model = inputs.compiled
    with pytest.raises(ValueError, match="read-only"):
        panel.observation_support.anchor_times[0] = 7


@pytest.mark.contract
def test_compile_distinguishes_incomplete_unsupported_and_bugs(monkeypatch):
    from nof1_causal_lab.models.ssm.compile import inputs as compiler

    incomplete = compiler.compile_ssm_inputs_from_model(ModelSpec())
    assert isinstance(incomplete, compiler.IncompleteModel)
    spec = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "common/one_state_gaussian_model.json"
        ).read_text()
    )
    unsupported = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "runtime/compile_distinguishes_incomplete_unsupported_and_bugs_with_parameter_distributions.json"
        ).read_text()
    )
    assert isinstance(compiler.compile_ssm_inputs_from_model(unsupported), compiler.UnsupportedFit)

    def broken_compiler(_compiled, _authored):
        raise ValueError("internal compiler bug")

    monkeypatch.setattr(compiler, "compile_priors", broken_compiler)
    with pytest.raises(ValueError, match="internal compiler bug"):
        compiler.compile_ssm_inputs_from_model(spec)


@pytest.mark.contract
def test_fit_resolves_incomplete_model_before_panel_preparation(monkeypatch):
    from nof1_causal_lab.actions.inference import fit as fitting

    def unexpected_panel(*_args, **_kwargs):
        raise AssertionError("panel prepared before fit capability was resolved")

    monkeypatch.setattr(fitting, "bind_panel", unexpected_panel)
    result = fitting.fit_model(ModelSpec(), pl.DataFrame(), time_origin=None, sampler=SamplerSpec())
    assert not result["fitted"]
    assert result["error"]
