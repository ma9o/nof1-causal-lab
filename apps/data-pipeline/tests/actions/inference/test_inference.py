"""Tests for Stage 5 inference task logging and orchestration."""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import override

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.actions.inference import fit as stage5_inference
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm import runtime as runtime_module
from nof1_causal_lab.models.ssm.compile import inputs as compilation
from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs
from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime
from nof1_causal_lab.models.ssm.runtime import BoundPanel
from nof1_causal_lab.sampler_config import SamplerSpec
from tests.model_fixtures import (
    bind_panel_fixture,
    compile_fit_fixture,
)

pytestmark = pytest.mark.contract


class _FakeResult(ParticleMCMCPosterior):
    def __init__(self) -> None:
        from tests.inference_fixtures import particle_posterior

        fixture = particle_posterior(
            JointPosteriorDraws(parameters={"theta": jnp.zeros((4, 1), dtype=jnp.float32)})
        )
        super().__init__(draws=fixture.draws, diagnostics=fixture.diagnostics)

    @override
    def get_inference_diagnostics(self, references):
        from nof1_causal_lab.artifacts.posterior_diagnostics import ChainDiagnostics

        return ChainDiagnostics(num_chains=1, num_samples=4, per_parameter=())

    @override
    def get_chain_detail(self, references):
        return (), ()

    @override
    def get_loo_diagnostics(self, *, observations):
        from nof1_causal_lab.artifacts.posterior_diagnostics import LOODiagnostics

        return LOODiagnostics(elpd_loo=-12.3, p_loo=1, se=0.2, n_data_points=2), ()

    @override
    def get_posterior_marginals(self, references, n_bins: int = 50):
        return ()

    @override
    def get_posterior_pairs(self, references, max_params: int = 6):
        return ()


def _make_observation_support_runtime() -> ObservationSupportRuntime:
    return ObservationSupportRuntime.assembled(
        manifest_names=("sleep_avg", "energy"),
        anchor_times=np.array([0.0, 1.5]),
        support_kinds=("interval", "point"),
        summary_operators=("mean", None),
        anchor_policies=("end", "end"),
        observation_windows=("1d", None),
        support_start_times=np.array([[np.nan, np.nan], [0.0, np.nan]]),
        support_end_times=np.array([[np.nan, np.nan], [1.5, np.nan]]),
        interval_prev_coeffs=np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.5, 0.0], [0.0, 0.0]],
            ]
        ),
        interval_curr_coeffs=np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.5, 0.0], [0.0, 0.0]],
            ]
        ),
        interval_weights=np.array(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ]
        ),
        emission_slot_indices=np.array([[-1, -1], [0, -1]]),
    )


def _make_panel(inputs: CompiledFitInputs) -> BoundPanel:
    return bind_panel_fixture(
        inputs.compiled,
        jnp.array([[0.2, 0.8], [jnp.nan, 0.5]], dtype=jnp.float32),
        jnp.array([0.0, 1.5], dtype=jnp.float32),
        support=_make_observation_support_runtime(),
    )


def test_fit_model_logs_runtime_summary_and_diagnostic_boundaries(monkeypatch, caplog):
    fake_result = _FakeResult()
    spec = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "inference/sleep_state_model.json"
        ).read_text()
    )
    fake_model = compile_fit_fixture(spec)
    runtime = _make_panel(fake_model)

    monkeypatch.setattr(compilation, "compile_ssm_inputs_from_model", lambda _selection: fake_model)
    monkeypatch.setattr(runtime_module, "bind_panel", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(
        stage5_inference, "fit_prepared_model", lambda _prepared, **_kwargs: fake_result
    )

    data_for_model = pl.DataFrame(
        {
            "indicator_id": ["indicator:sleep_avg", "indicator:energy", "indicator:energy"],
            "value": [0.2, 0.8, 0.5],
            "anchor_time": [
                "2024-01-01T00:00:00",
                "2024-01-01T00:00:00",
                "2024-01-02T12:00:00",
            ],
        }
    )

    with caplog.at_level(logging.INFO, logger=stage5_inference.logger.name):
        result = stage5_inference.fit_model(
            StructuralSelection(spec, None),
            data_for_model,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            sampler=SamplerSpec(),
        )

    assert result["fitted"]
    assert result["result"] is fake_result
    assert result["panel"] is runtime
    assert result["inference_diagnostics"].num_samples == 4
    assert result["inference_diagnostics"].per_parameter == ()
    assert "Prepared runtime in" in caplog.text
    assert "support=interval(1: manifest_0) max_active_windows=1" in caplog.text
    assert "Manifest order: manifest_0, manifest_1" in caplog.text
    assert "Starting inference kernel..." in caplog.text
    assert "Collecting sampler diagnostics..." in caplog.text
    assert "Computing leave-one-measurement-row-out diagnostics..." in caplog.text
    assert "Extracting posterior summaries..." in caplog.text
    assert "Posterior summaries ready in" in caplog.text
    assert "n_samples=4" in caplog.text


def test_fit_model_can_skip_loo_diagnostics(monkeypatch, caplog):
    fake_result = _FakeResult()
    spec = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "inference/sleep_state_model.json"
        ).read_text()
    )
    fake_model = compile_fit_fixture(spec)
    runtime = _make_panel(fake_model)

    monkeypatch.setattr(compilation, "compile_ssm_inputs_from_model", lambda _selection: fake_model)
    monkeypatch.setattr(runtime_module, "bind_panel", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(
        stage5_inference, "fit_prepared_model", lambda _prepared, **_kwargs: fake_result
    )

    data_for_model = pl.DataFrame(
        {
            "indicator_id": ["indicator:sleep_avg"],
            "value": [0.2],
            "anchor_time": ["2024-01-01T00:00:00"],
        }
    )

    with caplog.at_level(logging.INFO, logger=stage5_inference.logger.name):
        result = stage5_inference.fit_model(
            StructuralSelection(spec, None),
            data_for_model,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            sampler=SamplerSpec(),
            compute_loo_diagnostics=False,
        )

    assert result["fitted"]
    assert result["loo_diagnostics"] is None
    assert "Skipping LOO diagnostics by configuration." in caplog.text
    assert "Computing leave-one-measurement-row-out diagnostics..." not in caplog.text
