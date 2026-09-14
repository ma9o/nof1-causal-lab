"""Measurement authoring enriches the same scientific entities before compilation."""

import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
    measurement_structure_grounding,
)
from nof1_causal_lab.models.identification import identify_model
from nof1_causal_lab.models.model_structure import StructuralCompilationError
from nof1_causal_lab.utils.model_structure import get_edges, get_state_names
from tests.helpers import fixture_entity_id, graph_constructs, make_model


@pytest.fixture
def measured_model():
    model = make_model(["Treatment", "Outcome"], [("Treatment", "Outcome")])
    return model.revised(default_outcome=model.constructs[1].id)


def test_valid_measurement_preserves_whole_model(measured_model):
    payload = measured_model.model_dump(mode="json")
    output, feedback = measurement_structure_grounding(payload)
    assert feedback == "VALID"
    assert output == payload
    restored = ModelSpec.model_validate(output)
    assert restored == measured_model
    assert get_state_names(restored) == ["Treatment", "Outcome"]


def test_missing_outcome_indicator_returns_error(measured_model):
    payload = measured_model.model_dump(mode="json")
    graph_constructs(payload)[1]["indicators"] = []
    output, feedback = measurement_structure_grounding(payload)
    assert output is None
    assert "Outcome construct 'Outcome'" in feedback


def test_duplicate_operationalization_returns_error(measured_model):
    payload = measured_model.model_dump(mode="json")
    indicators = graph_constructs(payload)[0]["indicators"]
    indicators.append({**indicators[0], "id": "indicator:copy", "name": "treatment_copy"})
    output, feedback = measurement_structure_grounding(payload)
    assert output is None
    assert "duplicate indicator operationalizations" in feedback


def test_semantic_collision_returns_error(measured_model):
    payload = measured_model.model_dump(mode="json")
    graph_constructs(payload)[0]["indicators"][0]["how_to_measure"] = (
        "Count the number of treatments administered"
    )
    output, feedback = measurement_structure_grounding(payload)
    assert output is None
    assert "Semantic collision" in feedback


def test_measurements_retain_both_constructs(measured_model):
    output, feedback = measurement_structure_grounding(measured_model.model_dump(mode="json"))
    assert feedback == "VALID"
    model = ModelSpec.model_validate(output)
    assert get_state_names(model) == ["Treatment", "Outcome"]
    assert [(edge["cause"], edge["effect"]) for edge in get_edges(model)] == [
        ("Treatment", "Outcome")
    ]


def test_removed_usage_is_rejected(measured_model):
    payload = measured_model.model_dump(mode="json")
    graph_constructs(payload)[0]["usage"] = {
        "kind": "known_input",
        "source_indicator_id": graph_constructs(payload)[1]["indicators"][0]["id"],
    }
    output, feedback = measurement_structure_grounding(payload)
    assert output is None
    assert "usage" in feedback.lower()


def test_unknown_usage_is_rejected(measured_model):
    payload = measured_model.model_dump(mode="json")
    graph_constructs(payload)[0]["usage"] = {"kind": "unregistered"}
    output, feedback = measurement_structure_grounding(payload)
    assert output is None
    assert "usage" in feedback


def test_identification_is_a_separate_derived_finding(measured_model):
    output, feedback = measurement_structure_grounding(measured_model.model_dump(mode="json"))
    assert feedback == "VALID"
    model = ModelSpec.model_validate(output)
    report = identify_model(model)
    assert model.default_outcome is not None
    assert output is not None
    assert report.outcome == model.default_outcome
    assert model.constructs[0].id in report.estimable_treatments
    assert "identifiability" not in output


def test_unobserved_static_confounder_survives_measurement_authoring():
    model = make_model(
        ["Sleep", "Mood", "Chronotype"],
        [("Sleep", "Mood"), ("Chronotype", "Sleep"), ("Chronotype", "Mood")],
    )
    payload = model.model_dump(mode="json")
    payload["default_outcome"] = model.constructs[1].id
    graph_constructs(payload)[2].update(
        role="exogenous", temporal_status="time_invariant", indicators=[]
    )
    for edge in payload["edges"][1:]:
        edge["lagged"] = False
    output, feedback = measurement_structure_grounding(payload)
    assert feedback == "VALID"
    scientific_model = ModelSpec.model_validate(output)
    assert len(scientific_model.constructs) == 3
    with pytest.raises(StructuralCompilationError, match="Required unmeasured constructs"):
        scientific_model.require_execution_structure()
    report = identify_model(scientific_model)
    sleep = report.non_identifiable[model.constructs[0].id]
    assert sleep.confounders == [model.constructs[2].id]


def test_invalid_schema_is_reported():
    output, feedback = measurement_structure_grounding({"indicators": [{"bad": "data"}]})
    assert output is None
    assert "VALIDATION ERRORS" in feedback


def test_unmeasured_mediator_remains_scientific_but_not_an_executable_state():
    model = make_model(
        ["Treatment", "Mediator", "Outcome"],
        [("Treatment", "Mediator"), ("Mediator", "Outcome")],
    )
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                item.model_copy(update={"indicators": ()}) if item.name == "Mediator" else item
                for item in model.constructs
            ),
        ),
        default_outcome=fixture_entity_id("construct", "Outcome"),
    )
    plan = model
    assert get_state_names(plan) == ["Treatment", "Outcome"]
    assert get_edges(plan) == []
    assert model.get_construct(fixture_entity_id("construct", "Mediator")).name == "Mediator"
