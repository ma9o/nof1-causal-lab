"""Report responses retain rerunnable requests and their original execution basis."""

import pytest

from nof1_causal_lab.artifacts.scenarios import ScenarioRequest, SimulationResult
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.writes import execute_write
from tests.helpers import make_model


def request():
    return ScenarioRequest.model_validate(
        {
            "clamps": [{"target": {"id": "construct:x"}, "mode": "shift", "amount": -0.5}],
            "outcome": {"id": "construct:y"},
        }
    )


def test_one_request_can_produce_independently_pinned_responses(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    value = {
        "request": request().model_dump(mode="json"),
        "provenance": {
            "model": {"workspace_id": "QUERY", "version": 1},
            "rtol": 1e-4,
            "atol": 1e-6,
            "max_steps": 4096,
            "draw_count": 2,
            "time_grid_days": [0.0, 1.0, 2.0],
        },
        "labels": {"construct:x": "Treatment", "construct:y": "Outcome"},
        "summary": {
            "mean": 0.2,
            "median": 0.2,
            "lower_95": 0.1,
            "upper_95": 0.3,
            "prob_positive": 1.0,
        },
        "reference_mean": 1.0,
    }
    first = SimulationResult.model_validate(value)
    value["provenance"]["model"]["version"] = 2
    second = SimulationResult.model_validate(value)
    assert first.request == second.request
    assert first.provenance.model.version == 1
    assert second.provenance.model.version == 2
    assert SimulationResult.model_validate_json(first.model_dump_json()) == first
    from nof1_causal_lab.machine.store import EpisodeJournal
    from tests.inference_fixtures import inference_log

    store = ArtifactStore("QUERY")
    model = make_model(["X", "Y"], [("X", "Y")])
    info = store.write_version(
        "model",
        provenance="computed",
        derived_from={},
        produced_by="run:posterior",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    record = inference_log(model, version=1).model_copy(update={"produced": [info]})
    EpisodeJournal("QUERY").append(record)
    payload = {"intervention_results": [], "simulation_results": [first.model_dump(mode="json")]}
    committed = execute_write(
        "QUERY", "baseline_report", payload, "human", EpisodeState()
    ).produced[0]
    retained = store.read_json_file("baseline_report", committed.version, "baseline_report.json")[
        "simulation_results"
    ][0]
    assert SimulationResult.model_validate(retained) == first
    payload["simulation_results"][0]["provenance"]["model"]["version"] = 2
    with pytest.raises(ValueError, match="absent model revision"):
        execute_write("QUERY", "baseline_report", payload, "human", EpisodeState())
    assert store.list_versions("baseline_report") == [1]
