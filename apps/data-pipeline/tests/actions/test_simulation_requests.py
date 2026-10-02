"""One forward-generation contract with dated interventions and durable histories."""

from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.actions.contracts import SimulateRequest
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.scenarios import InterventionSpec
from nof1_causal_lab.artifacts.simulation import SimulationReport, SimulationSpec
from tests.inference_fixtures import particle_posterior
from tests.model_fixtures import compile_model_fixture

pytestmark = pytest.mark.contract


def test_simulation_defaults_to_an_unintervened_model_continuation():
    request = SimulateRequest(model_revision="a" * 40, end=12)
    assert request.start is None
    assert request.interventions == ()
    assert set(request.model_dump()) == {
        "action",
        "model_revision",
        "start",
        "end",
        "interventions",
    }
    with pytest.raises(ValidationError, match="Extra inputs"):
        request.revised(comparison_panel_revision="b" * 40)


def test_interventions_round_trip_with_absolute_times():
    spec = SimulationSpec(
        start=0,
        end=3,
        interventions=(
            InterventionSpec(target="construct:x", time=1, value=2),
            InterventionSpec(target="construct:x", time=2, value=4),
        ),
    )
    assert SimulationSpec.model_validate_json(spec.model_dump_json()) == spec


@pytest.mark.parametrize(
    "events",
    [
        [{"time": -1, "value": 2}],
        [{"time": 4, "value": 2}],
        [{"time": 1, "value": float("nan")}],
        [{"time": float("inf"), "value": 2}],
        [{"time": 1, "value": [[1, 2], [2, 4]]}],
        [{"kind": "hold", "start": 0, "end": 2, "value": 1}],
        [{"time": 1, "value": 1}, {"time": 1, "value": 2}],
    ],
)
def test_intervention_times_values_and_conflicts_are_explicit(events):
    with pytest.raises(ValidationError):
        SimulationSpec.model_validate(
            {
                "start": 0,
                "end": 3,
                "interventions": [{"target": "construct:x", **event} for event in events],
            }
        )


def _response():
    def series(reference, action):
        def numeric(values):
            return {
                "kind": "numeric",
                "mean": values,
                "lower": values,
                "upper": values,
                "n_draws": [100] * 3,
            }

        return {"label": "Saved series", "reference": numeric(reference), "action": numeric(action)}

    result = {
        "outcome": "construct:y",
        "labels": {"construct:x": "Treatment", "construct:y": "Outcome"},
        "effect_trajectory": [
            {"day": day, "effect": 0.1 * day, "lower_95": 0.0, "upper_95": 0.2 * day}
            for day in (0, 1, 2)
        ],
        "summary": {
            "mean": 0.2,
            "median": 0.2,
            "lower_95": 0.1,
            "upper_95": 0.3,
            "prob_positive": 1.0,
        },
        "reference_mean": 1.0,
    }

    return {
        "design": {
            "end": 2,
            "interventions": [{"target": "construct:x", "time": 0, "value": 1}],
        },
        "times": [0, 1, 2],
        "draws": 100,
        "seed": 0,
        "time_origin": None,
        "predictive": {
            "states": {
                "construct:x": series([1.0, 1.0, 1.0], [0.5, 0.5, 0.5]),
                "construct:y": series([1.0, 1.0, 1.0], [1.0, 1.1, 1.2]),
            },
            "indicators": {"indicator:y": series([1.0, 1.0, 1.0], [1.0, 1.1, 1.2])},
            "fit_reliability": "converged",
        },
        "model": {"workspace_id": "QUERY", "revision": "a" * 40, "path": "model.json"},
        "state_ids": ["construct:x", "construct:y"],
        "observation_layout": {
            "variables": [
                {
                    "id": "indicator:y",
                    "name": "y",
                    "measurement_dtype": "continuous",
                    "aggregation": "last",
                    "observation_window": "1d",
                }
            ],
            "support_start_times": "starts",
            "support_end_times": "ends",
            "mask": "mask",
        },
        "parameter_draws": {},
        "latent_paths": "paths",
        "observations": "observations",
        "reference_latent_paths": "reference-paths",
        "reference_observations": "reference-observations",
        "causal_result": result,
    }


def test_one_request_can_produce_independently_pinned_responses():
    value = _response()
    first = SimulationReport.model_validate(value)
    value["model"]["revision"] = "b" * 40
    second = SimulationReport.model_validate(value)
    assert first.design == second.design
    assert first.model.revision == "a" * 40
    assert second.model.revision == "b" * 40
    assert SimulationReport.model_validate_json(first.model_dump_json()) == first


@pytest.mark.parametrize(
    "violation",
    ["reference_length", "action_length", "missing_series", "missing_target", "unknown_construct"],
)
def test_simulation_trajectories_share_a_grid_and_resolve_constructs(violation):
    value = _response()
    trajectories = value["predictive"]["states"]
    if violation in {"reference_length", "action_length"}:
        field = "reference" if violation == "reference_length" else "action"
        trajectories["construct:x"][field]["mean"].pop()
        message = "align with simulation times"
    elif violation == "missing_series":
        del trajectories["construct:x"]["action"]
        message = "Field required"
    elif violation == "missing_target":
        del trajectories["construct:x"]
        message = "include the outcome and interventions"
    else:
        trajectories["construct:unknown"] = trajectories["construct:x"]
        message = "cover the recorded simulation layout"
    with pytest.raises(ValidationError, match=message):
        SimulationReport.model_validate(value)


@pytest.mark.parametrize("missing", ["reference_latent_paths", "reference_observations"])
def test_causal_reports_require_their_effects_and_paired_draws(missing):
    payload = _response()
    del payload[missing]
    with pytest.raises(ValidationError, match="paired reference"):
        TypeAdapter(SimulationReport).validate_python(payload)


@pytest.mark.parametrize(
    ("kind", "current_panel", "reliable", "scientific_model_payload"),
    [
        pytest.param(
            "authored",
            True,
            "not_fitted",
            "common/stress_sleep_model.json",
            id="authored-True-not_fitted",
        ),
        pytest.param(
            "authored",
            False,
            "not_fitted",
            "common/stress_sleep_model.json",
            id="authored-False-not_fitted",
        ),
        pytest.param(
            "fitted",
            True,
            "converged",
            "common/stress_sleep_model.json",
            id="fitted-True-converged",
        ),
        pytest.param(
            "fitted",
            True,
            "unconverged",
            "common/stress_sleep_model.json",
            id="fitted-True-unconverged",
        ),
        pytest.param(
            "unknown",
            True,
            "unknown",
            "common/stress_sleep_model.json",
            id="unknown-True-unknown",
        ),
    ],
)
def test_runner_derives_origin_and_reliability_from_current_panel_or_fit(
    tmp_path, monkeypatch, kind, current_panel, reliable, scientific_model_payload
):
    from datetime import UTC, datetime
    from importlib import import_module

    import jax.numpy as jnp

    from nof1_causal_lab.actions import scenarios
    from nof1_causal_lab.actions.runners import run_action
    from nof1_causal_lab.models.ssm.inference.persistence import condition_model
    from nof1_causal_lab.models.ssm.inference.types import (
        JointPosteriorDraws,
    )
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.state import StudyState
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import data
    from tests.helpers import run_async
    from tests.inference_fixtures import inference_log
    from tests.integration.runner_fixtures import panel_metadata
    from tests.model_fixtures import parameter_draws

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("ORIGIN")
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / scientific_model_payload
        ).read_text()
    )
    prior = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    fit_panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={"metadata.json": panel_metadata().model_dump(mode="json")},
    )
    record = prior
    fit_origin = datetime(2024, 1, 2, tzinfo=UTC)
    if kind in {"fitted", "unknown"}:
        model = condition_model(
            model,
            compile_model_fixture(model),
            particle_posterior(
                JointPosteriorDraws(parameter_draws(model, 2), jnp.zeros((2, 2, 2)))
            ),
            times=jnp.array([0.0, 1.0]),
        )
        record = store.write_artifact(
            "model",
            derived_from={"model": prior.revision, "panel": fit_panel.revision}
            if kind == "fitted"
            else {},
            produced_by="fit" if kind == "fitted" else "edit_model",
            json_files={"model.json": model.model_dump(mode="json")},
        )
        if kind == "fitted":
            from nof1_causal_lab.artifacts.identity import GitRef
            from nof1_causal_lab.models.ssm.inference.convergence import parameter_convergence
            from nof1_causal_lab.study.records import ModelFitResult
            from tests.action_fixtures import applied_record

            fit = inference_log(model).record.attempt.outcome.result.report.revised(
                **{"time_origin": fit_origin}
            )
            if reliable == "unconverged":
                diagnostics = fit.inference_diagnostics
                assert diagnostics is not None
                poor = diagnostics.revised(
                    **{
                        "per_parameter": tuple(
                            row.revised(**{"r_hat": 1.2}) for row in diagnostics.per_parameter
                        )
                    }
                )
                fit = fit.revised(
                    **{
                        "inference_diagnostics": poor,
                        "convergence": parameter_convergence(poor),
                    }
                )
            StudyRepository("ORIGIN").append(
                applied_record(
                    ModelFitResult(
                        model=GitRef(
                            workspace_id="ORIGIN", revision=prior.revision, path="model.json"
                        ),
                        panel=GitRef(
                            workspace_id="ORIGIN", revision=fit_panel.revision, path="panel.parquet"
                        ),
                        produced=(prior, fit_panel, record),
                        report=fit,
                    ),
                    seq=1,
                )
            )
    current_origin = datetime(2026, 1, 1, tzinfo=UTC)
    metadata = panel_metadata()
    panel = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="prepare_data",
        json_files={
            "metadata.json": metadata.revised(time_origin=current_origin).model_dump(mode="json")
        },
    )
    state = StudyState().with_artifacts([record, panel] if current_panel else [record])
    expected_origin = fit_origin if kind == "fitted" else current_origin if current_panel else None
    expected_panel = (
        fit_panel.revision if kind == "fitted" else panel.revision if current_panel else None
    )
    response = SimulationReport.model_validate(_response())

    def generate(_model, _design, *, revision, time_origin, fit_reliability, **_kwargs):
        assert time_origin == expected_origin
        assert fit_reliability == reliable
        return response.revised(
            model=revision,
            time_origin=time_origin,
            predictive=response.predictive.revised(fit_reliability=fit_reliability),
        )

    monkeypatch.setattr(import_module("nof1_causal_lab.actions.simulate"), "simulate", generate)
    monkeypatch.setattr(
        scenarios, "summarize_causal_simulation", lambda _model, report, **_kwargs: report
    )
    effects = run_async(
        run_action(
            "ORIGIN",
            SimulateRequest(
                model_revision=record.revision,
                start=response.design.start,
                end=response.design.end,
                interventions=response.design.interventions,
            ),
            state,
        )
    )
    saved = effects.report
    assert saved.time_origin == expected_origin
    assert saved.origin_panel_revision == expected_panel
    assert saved.predictive.fit_reliability == reliable
    assert saved.law is not None
    assert saved.law.kind == kind
    assert (effects.panel.revision if effects.panel is not None else None) == (
        panel.revision if current_panel else None
    )
