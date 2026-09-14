"""Runtime responses include rerunnable requests and their execution basis."""

import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts.scenarios import ScenarioRequest, SimulationResult


def request():
    return ScenarioRequest.model_validate(
        {
            "clamps": [{"target": "construct:x", "mode": "shift", "amount": -0.5}],
            "outcome": "construct:y",
        }
    )


def _response():
    return {
        "request": request().model_dump(mode="json"),
        "model": {"workspace_id": "QUERY", "version": 1},
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


def test_one_request_can_produce_independently_pinned_responses():
    value = _response()
    first = SimulationResult.model_validate(value)
    value["model"]["version"] = 2
    second = SimulationResult.model_validate(value)
    assert first.request == second.request
    assert first.model.version == 1
    assert second.model.version == 2
    assert SimulationResult.model_validate_json(first.model_dump_json()) == first


@pytest.mark.parametrize(
    "violation",
    ["reference_length", "action_length", "missing_series", "missing_target", "unknown_construct"],
)
def test_simulation_trajectories_share_a_grid_and_resolve_constructs(violation):
    value = _response()
    trajectories = value["trajectories"]
    if violation in {"reference_length", "action_length"}:
        field = "reference_mean" if violation == "reference_length" else "action_mean"
        trajectories["construct:x"][field].pop()
        message = "align with time_grid_days"
    elif violation == "missing_series":
        del trajectories["construct:x"]["action_mean"]
        message = "Field required"
    elif violation == "missing_target":
        del trajectories["construct:x"]
        message = "include the requested constructs"
    else:
        trajectories["construct:unknown"] = trajectories["construct:x"]
        message = "must have construct labels"
    with pytest.raises(ValidationError, match=message):
        SimulationResult.model_validate(value)
