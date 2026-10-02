"""Predictive batch boundaries and exact discrete diagnostic statistics."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.predictive.statistics import observation_signal_and_variance
from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
from tests.model_fixtures import compile_model_fixture


def _trajectory():
    observations = jnp.zeros((2, 3, 2))
    return PredictiveTrajectory(
        latents=jnp.zeros((2, 3, 1)),
        linear_predictors=observations,
        observations=observations,
        observations_mask=jnp.ones(observations.shape, dtype=bool),
        expected_observations=observations,
    )


@pytest.mark.contract
@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("latents", jnp.zeros((2, 1))),
        ("latents", jnp.zeros((2, 4, 1))),
        ("observations", jnp.zeros((1, 3, 2))),
        ("linear_predictors", jnp.zeros((2, 3, 1))),
        ("expected_observations", jnp.zeros((2, 4, 2))),
        ("observations_mask", jnp.ones((2, 3, 2))),
    ],
)
def test_predictive_trajectory_rejects_misaligned_axes(field, values):
    with pytest.raises(ValueError, match=r"axes|axis|boolean"):
        replace(_trajectory(), **{field: values})


@pytest.mark.contract
def test_predictive_batch_preserves_pairing_and_rejects_misaligned_parameters():
    batch = PredictiveDraws(
        parameters={"coefficient": jnp.array([1.0, 2.0])},
        trajectory=_trajectory(),
        reference=_trajectory(),
    )
    assert batch.n_draws == 2
    assert jax.block_until_ready(batch).parameters.keys() == {"coefficient"}
    with pytest.raises(ValueError, match="draw axis"):
        replace(batch, parameters={"coefficient": jnp.ones(3)})
    with pytest.raises(ValueError, match="Reference"):
        replace(batch, reference=replace(_trajectory(), latents=jnp.zeros((2, 3, 2))))


@pytest.mark.inference(concern="predictive")
def test_discrete_diagnostics_share_cutpoints_anchors_and_declared_probabilities():
    spec = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "predictive_batches/discrete_diagnostics_share_cutpoints_anchors_and_padded_probabilities_model_fixture.json"
        ).read_text()
    )
    raw = {
        "obs_ordered_base": jnp.array([[0.0, 0.0], [1.0, 0.0]]),
        "obs_ordered_gaps": jnp.ones((2, 2, 2)),
        "obs_cat_intercepts": jnp.zeros((2, 2, 3)),
        # The categorical anchor must replace 99 by +1, and padded entries vanish.
        "obs_cat_slopes": jnp.full((2, 2, 3), 99.0),
    }
    assert numeric.categorical_anchors(compile_model_fixture(spec))[1]
    predictors = jnp.broadcast_to(jnp.array([-100.0, 0.0, 100.0])[None, :, None], (2, 3, 2))
    batch = PredictiveDraws(
        parameters={**raw, "manifest_cov": jnp.broadcast_to(jnp.eye(2), (2, 2, 2))},
        trajectory=replace(_trajectory(), linear_predictors=predictors),
    )
    indices = np.arange(3)
    ordered, variance = observation_signal_and_variance(
        compile_model_fixture(spec), batch, 0, indices
    )
    assert variance is None
    np.testing.assert_allclose(ordered.sum(axis=2), 1.0, atol=1e-7)
    np.testing.assert_array_equal(ordered[:, 0], [[1.0, 0.0, 0.0, 0.0]] * 2)
    np.testing.assert_array_equal(ordered[:, -1], [[0.0, 0.0, 0.0, 1.0]] * 2)
    np.testing.assert_allclose(ordered[:, 1, 0], jax.nn.sigmoid(jnp.array([0.0, 1.0])))

    categorical, variance = observation_signal_and_variance(
        compile_model_fixture(spec), batch, 1, indices
    )
    assert variance is None
    assert categorical.shape == (2, 3, 2)
    np.testing.assert_allclose(categorical[:, 1, :2], 0.5)
    np.testing.assert_array_equal(categorical[:, 0, :2], [[1.0, 0.0]] * 2)
    np.testing.assert_array_equal(categorical[:, -1, :2], [[0.0, 1.0]] * 2)


@pytest.mark.inference(concern="predictive")
def test_scalar_diagnostics_use_projected_means_at_supported_times():
    means = jnp.array([[[jnp.nan, 0.0], [1e-12, 0.0], [4e-12, 0.0]]] * 2)
    batch = PredictiveDraws(
        parameters={"manifest_cov": jnp.broadcast_to(jnp.eye(2), (2, 2, 2))},
        trajectory=replace(
            _trajectory(),
            linear_predictors=jnp.full((2, 3, 2), 10.0),
            expected_observations=means,
            observations_mask=jnp.isfinite(means),
        ),
    )
    signal, variance = observation_signal_and_variance(
        compile_model_fixture(
            ModelSpec.model_validate_json(
                (
                    Path(__file__).resolve().parents[2]
                    / "fixtures/models"
                    / "predictive_batches/scalar_diagnostics_use_projected_means_at_supported_times_model_fixture.json"
                ).read_text()
            )
        ),
        batch,
        0,
        np.array([1, 2]),
    )
    np.testing.assert_array_equal(signal, np.asarray(means[:, 1:, 0]))
    np.testing.assert_array_equal(variance, signal)


@pytest.mark.inference(concern="predictive")
def test_undefined_student_moments_produce_an_explicit_diagnostic():
    from nof1_causal_lab.models.ssm.reachability import check_transmission

    paths = _trajectory()
    batch = PredictiveDraws(
        parameters={
            "manifest_cov": jnp.broadcast_to(jnp.eye(2), (2, 2, 2)),
            "obs_df": jnp.array([0.5, 5.0]),
        },
        trajectory=paths,
    )
    signal, variance = observation_signal_and_variance(
        compile_model_fixture(
            ModelSpec.model_validate_json(
                (
                    Path(__file__).resolve().parents[2]
                    / "fixtures/models"
                    / "predictive_batches/undefined_student_moments_produce_an_explicit_diagnostic_model_fixture.json"
                ).read_text()
            )
        ),
        batch,
        0,
        np.arange(3),
    )
    result = check_transmission("heavy_tail", signal, variance)
    assert not result.passed
    assert np.isnan(signal[0]).all()
    assert variance is not None
    assert np.isnan(variance[0]).all()
    assert result.assessment.kind == "not_evaluated"
    assert result.assessment.reason == "NONFINITE_SIGNAL"


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize(
    ("families", "interval", "expected_variance", "model_fixture_payload"),
    [
        pytest.param(
            ["gaussian"],
            False,
            1.0001e-08,
            "predictive_batches/diagnostic_noise_matches_point_and_interval_execution_model_fixture_families0-false-1_0001e-08.json",
            id="families0-False-1.0001e-08",
        ),
        pytest.param(
            ["gaussian", "poisson"],
            False,
            1e-10,
            "predictive_batches/gaussian_poisson_model.json",
            id="families1-False-1e-10",
        ),
        pytest.param(
            ["gaussian", "poisson"],
            True,
            1.0001e-08,
            "predictive_batches/gaussian_poisson_model.json",
            id="families2-True-1.0001e-08",
        ),
        pytest.param(
            ["student_t"],
            False,
            1.6666666666666669e-10,
            "predictive_batches/student_t_model.json",
            id="families3-False-1.6666666666666669e-10",
        ),
        pytest.param(
            ["student_t"],
            True,
            1.6666666666666666e-12,
            "predictive_batches/student_t_model.json",
            id="families4-True-1.6666666666666666e-12",
        ),
    ],
)
def test_diagnostic_noise_matches_point_and_interval_execution(
    families, interval, expected_variance, model_fixture_payload
):
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

    channels = len(families)
    spec = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2] / "fixtures/models" / model_fixture_payload
        ).read_text()
    )
    values = jnp.zeros((2, 3, channels))
    prediction = PredictiveDraws(
        parameters={
            "manifest_cov": jnp.broadcast_to(jnp.eye(channels) * 1e-12, (2, channels, channels)),
            "obs_df": jnp.full(2, 5.0),
        },
        trajectory=PredictiveTrajectory(
            values, values, values, jnp.ones_like(values, dtype=bool), values
        ),
    )
    support = None
    if interval:
        support = ObservationSupportRuntime(
            anchor_times=np.arange(3.0),
            manifest_names=numeric.observation_names(compile_model_fixture(spec)),
            support_kinds=("interval",) * channels,
            summary_operators=("mean",) * channels,
            anchor_policies=("support_end",) * channels,
            observation_windows=("2d",) * channels,
            support_start_times=np.tile([[np.nan], [np.nan], [0.0]], (1, channels)),
            support_end_times=np.tile([[np.nan], [np.nan], [2.0]], (1, channels)),
            interval_prev_coeffs=np.tile([[[0.0]], [[0.5]], [[0.5]]], (1, channels, 1)),
            interval_curr_coeffs=np.tile([[[0.0]], [[0.5]], [[0.5]]], (1, channels, 1)),
            interval_weights=np.tile([[[0.0]], [[1.0]], [[1.0]]], (1, channels, 1)),
            emission_slot_indices=np.tile([[-1], [-1], [0]], (1, channels)),
        )
    _, variance = observation_signal_and_variance(
        compile_model_fixture(spec), prediction, 0, np.array([2]), observation_support=support
    )
    assert variance is not None
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-6, atol=0.0)
