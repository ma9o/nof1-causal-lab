"""Scientific questions survive refits while each evaluation preserves its exact basis."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts.baseline_report import SavedScenario, SavedScenariosArtifact
from nof1_causal_lab.artifacts.effects import EffectSummary
from nof1_causal_lab.artifacts.identity import ArtifactRef, ModelRef
from nof1_causal_lab.artifacts.scenarios import (
    ScenarioDefinition,
    ScenarioEvaluation,
    ScenarioEvaluationResult,
    ScenarioQuery,
    ScenarioResult,
    ScenarioStartResult,
    SimulateScenarioResult,
)
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.writes import execute_write


def definition():
    return {
        "start": {"kind": "baseline"},
        "clamps": [
            {
                "variable": "Treatment",
                "target": {"kind": "construct", "id": "construct:x"},
                "mode": "shift",
                "amount": -0.5,
            }
        ],
        "outcome": {"kind": "construct", "id": "construct:y"},
        "readout": {"horizon_days": 30},
    }


def query_from(payload=None):
    return ScenarioQuery.from_definition(ScenarioDefinition.model_validate(payload or definition()))


def evaluation_for(query, *, model="QUERY", version=1):
    return ScenarioEvaluation.for_query(
        query,
        model=ModelRef(id=model),
        posterior=ArtifactRef(artifact_id="posterior", version=version),
    )


def evaluated(query, *, model="QUERY", version=1):
    evaluation = evaluation_for(query, model=model, version=version)
    return ScenarioEvaluationResult(
        evaluation=evaluation,
        result=ScenarioResult(
            evaluation_id=evaluation.id,
            start=ScenarioStartResult(kind="baseline", state_source="baseline_steady_state"),
            outcome_label="Outcome at execution",
            summary=EffectSummary(
                mean=0.2, median=0.2, lower_95=0.1, upper_95=0.3, prob_positive=0.99
            ),
            reference_mean=1.0,
        ),
    )


def test_query_identity_ignores_labels_but_preserves_scientific_meaning():
    payload = definition()
    original = query_from(payload)
    payload["clamps"][0]["variable"] = "Renamed treatment"
    assert query_from(payload).id == original.id
    payload["readout"]["horizon_days"] = 60
    assert query_from(payload).id != original.id
    tampered = deepcopy(original.model_dump(mode="json"))
    tampered["readout"]["horizon_days"] = 60
    with pytest.raises(ValidationError, match="identity does not match"):
        ScenarioQuery.model_validate(tampered)


def test_query_identity_survives_model_revisions_and_refits():
    query = query_from()
    evaluations = [
        evaluation_for(query),
        evaluation_for(query, version=2),
        evaluation_for(query, model="REVISED"),
    ]
    assert {item.query_id for item in evaluations} == {query.id}
    assert len({item.id for item in evaluations}) == 3
    tampered = evaluations[0].model_dump(mode="json")
    tampered["posterior"]["version"] = 2
    with pytest.raises(ValidationError, match="execution basis"):
        ScenarioEvaluation.model_validate(tampered)


def test_evaluation_must_pin_a_posterior():
    with pytest.raises(ValidationError, match="must pin a posterior"):
        ScenarioEvaluation.for_query(
            query_from(),
            model=ModelRef(id="QUERY"),
            posterior=ArtifactRef(artifact_id="panel", version=1),
        )


@pytest.mark.parametrize("with_evaluations", [False, True])
def test_saved_query_preserves_all_historical_evaluations(monkeypatch, tmp_path, with_evaluations):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("QUERY")
    for _ in range(2):
        latest = store.write_version(
            "posterior", provenance="computed", derived_from={}, produced_by="run:posterior"
        )
    ArtifactStore("REVISED").write_version(
        "posterior", provenance="computed", derived_from={}, produced_by="run:posterior"
    )
    query = query_from()
    evaluations = (
        [evaluated(query), evaluated(query, version=2), evaluated(query, model="REVISED")]
        if with_evaluations
        else []
    )
    scenario = SavedScenario(label="Keep", query=query, evaluations=evaluations)
    effects = execute_write(
        "QUERY",
        "saved_scenarios",
        {"scenarios": [scenario.model_dump(mode="json")]},
        "human",
        EpisodeState(current={"posterior": latest}),
    )
    saved = effects.produced[0]
    assert saved.derived_from == {}
    payload = store.read_json_file("saved_scenarios", saved.version, "saved_scenarios.json")
    assert SavedScenario.model_validate(payload["scenarios"][0]) == scenario


def test_saved_evaluation_rejects_an_unavailable_posterior(monkeypatch, tmp_path):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    query = query_from()
    scenario = SavedScenario(label="Keep", query=query, evaluations=[evaluated(query)])
    with pytest.raises(ValueError, match="posterior"):
        execute_write(
            "QUERY",
            "saved_scenarios",
            {"scenarios": [scenario.model_dump(mode="json")]},
            "human",
            EpisodeState(),
        )


def test_containers_reject_mismatched_query_or_evaluation():
    query = query_from()
    item = evaluated(query)
    response = SimulateScenarioResult(query=query, **item.model_dump())
    assert SimulateScenarioResult.model_validate_json(response.model_dump_json()) == response
    other = definition()
    other["readout"]["horizon_days"] = 60
    with pytest.raises(ValidationError, match="different query"):
        SimulateScenarioResult(query=query_from(other), **item.model_dump())
    with pytest.raises(ValidationError, match="different evaluation"):
        ScenarioEvaluationResult(evaluation=evaluation_for(query, version=2), result=item.result)
    with pytest.raises(ValidationError, match="different query"):
        SavedScenario(label="Keep", query=query_from(other), evaluations=[item])


def test_save_one_scientific_query_with_unique_evaluations():
    query = query_from()
    item = evaluated(query)
    with pytest.raises(ValidationError, match="duplicate evaluations"):
        SavedScenario(label="Keep", query=query, evaluations=[item, item])
    scenario = SavedScenario(label="Keep", query=query, evaluations=[item])
    with pytest.raises(ValidationError, match="Save each scientific query once"):
        SavedScenariosArtifact(scenarios=[scenario, scenario])
