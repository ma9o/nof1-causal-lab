"""Tests for the fit-time observation/prior preflight checks."""

import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.ssm.inference import fit
from nof1_causal_lab.models.ssm.preflight import (
    ObservationPreflightError,
    validate_observations_for_fit,
)
from nof1_causal_lab.sampler_config import (
    SamplerSpec,
)
from tests.model_fixtures import bind_panel_fixture, compile_fit_fixture

pytestmark = pytest.mark.contract

RNG = np.random.default_rng(7)


def _observations(mean_a, mean_b, n=200):
    return np.column_stack([RNG.normal(mean_a, 1.0, size=n), RNG.normal(mean_b, 1.0, size=n)])


def _panel(inputs, observations):
    return bind_panel_fixture(
        inputs.compiled, observations, jnp.arange(len(observations), dtype=jnp.float32)
    )


def test_raises_on_unreachable_free_manifest_mean():
    model = compile_fit_fixture(
        ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "fit_preflight/raw_channels_model.json"
            ).read_text()
        )
    )
    with pytest.raises(ObservationPreflightError, match="raw_channel"):
        validate_observations_for_fit(
            model.prior_runtime_bundle, _panel(model, _observations(87.0, 0.1))
        )


def test_passes_when_free_mean_is_within_prior_reach():
    model = compile_fit_fixture(
        ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "fit_preflight/raw_channels_model.json"
            ).read_text()
        )
    )
    validate_observations_for_fit(
        model.prior_runtime_bundle, _panel(model, _observations(0.5, -0.3))
    )


def test_preflight_uses_compiled_authored_location_laws():

    spec = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "fit_preflight/preflight_uses_compiled_authored_location_laws_with_parameter_distributions.json"
        ).read_text()
    )
    model = compile_fit_fixture(spec)
    validate_observations_for_fit(
        model.prior_runtime_bundle, _panel(model, _observations(87.0, 0.1))
    )


def test_fixed_manifest_means_are_not_judged():
    model = compile_fit_fixture(
        ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "fit_preflight/fixed_manifest_means_are_not_judged__model.json"
            ).read_text()
        )
    )
    validate_observations_for_fit(
        model.prior_runtime_bundle, _panel(model, _observations(87.0, 0.1))
    )


def test_binding_standardizes_flagged_channels_before_preflight():
    model = compile_fit_fixture(
        ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "fit_preflight/standardized_channel_model.json"
            ).read_text()
        )
    )
    obs = _observations(87.0, 0.1)
    validate_observations_for_fit(model.prior_runtime_bundle, _panel(model, obs))


def test_non_identity_links_are_not_judged():
    model = compile_fit_fixture(
        ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "fit_preflight/non_identity_links_are_not_judged__model.json"
            ).read_text()
        )
    )
    obs = _observations(0.0, 0.1)
    obs[:, 0] = RNG.poisson(80.0, size=obs.shape[0]).astype(np.float64)
    validate_observations_for_fit(model.prior_runtime_bundle, _panel(model, obs))


def test_nan_only_channels_are_skipped():
    model = compile_fit_fixture(
        ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "fit_preflight/raw_channels_model.json"
            ).read_text()
        )
    )
    obs = _observations(0.2, 0.1)
    obs[:, 0] = np.nan
    validate_observations_for_fit(model.prior_runtime_bundle, _panel(model, obs))


def test_fit_runs_preflight_before_dispatch():
    model = compile_fit_fixture(
        ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "fit_preflight/raw_channels_model.json"
            ).read_text()
        )
    )
    obs = _observations(87.0, 0.1)
    with pytest.raises(ObservationPreflightError, match="raw_channel"):
        fit(
            model.prior_runtime_bundle,
            bind_panel_fixture(
                model.compiled, jnp.asarray(obs), jnp.arange(obs.shape[0], dtype=jnp.float32)
            ),
            sampler=SamplerSpec(),
            clock=time.monotonic,
        )
