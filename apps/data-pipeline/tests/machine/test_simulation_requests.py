"""One forward-generation contract with dated interventions and durable histories."""

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.actions.contracts import SimulateRequest
from nof1_causal_lab.artifacts.scenarios import InterventionSpec
from nof1_causal_lab.artifacts.simulation import SimulationReport, SimulationSpec

pytestmark = pytest.mark.contract


def test_simulation_defaults_to_an_unintervened_model_continuation():
    request = SimulateRequest(model_revision="a" * 40, end=12)
    assert request.start is None
    assert request.interventions == ()
    assert set(request.model_dump()) == {
        "action",
        "model_revision",
        "comparison_panel_revision",
        "start",
        "end",
        "interventions",
    }


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
    result = {
        "outcome": "construct:y",
        "time_grid_days": [0.0, 1.0, 2.0],
        "labels": {"construct:x": "Treatment", "construct:y": "Outcome"},
        "trajectories": {
            "construct:x": {
                "reference_mean": [1.0, 1.0, 1.0],
                "action_mean": [0.5, 0.5, 0.5],
            },
            "construct:y": {
                "reference_mean": [1.0, 1.0, 1.0],
                "action_mean": [1.0, 1.1, 1.2],
            },
        },
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
        "model": {"workspace_id": "QUERY", "revision": "a" * 40, "path": "model.json"},
        "state_ids": ["construct:x", "construct:y"],
        "indicator_ids": ["indicator:y"],
        "observation_layout": {
            "variables": [{"id": "indicator:y", "name": "y", "measurement_dtype": "continuous",
                "aggregation": "last", "observation_window": "1d"}],
            "support_start_times": "starts", "support_end_times": "ends", "mask": "mask",
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
    trajectories = value["causal_result"]["trajectories"]
    if violation in {"reference_length", "action_length"}:
        field = "reference_mean" if violation == "reference_length" else "action_mean"
        trajectories["construct:x"][field].pop()
        message = "align with time_grid_days"
    elif violation == "missing_series":
        del trajectories["construct:x"]["action_mean"]
        message = "Field required"
    elif violation == "missing_target":
        del trajectories["construct:x"]
        message = "include the outcome and interventions"
    else:
        trajectories["construct:unknown"] = trajectories["construct:x"]
        message = "must have construct labels"
    with pytest.raises(ValidationError, match=message):
        SimulationReport.model_validate(value)


@pytest.mark.parametrize("missing", ["reference_latent_paths", "reference_observations"])
def test_causal_reports_require_their_effects_and_paired_draws(missing):
    payload = _response()
    del payload[missing]
    with pytest.raises(ValidationError, match="paired reference"):
        TypeAdapter(SimulationReport).validate_python(payload)
