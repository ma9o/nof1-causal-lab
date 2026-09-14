"""Whole-model authoring returns intrinsic errors and checks requested readiness."""

import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.flows.model_authoring import validate_model_submission
from tests.helpers import graph_constructs, invalid_dict_payload, make_model


def test_valid_partial_model_can_be_enriched_for_measurement():
    model = make_model(["stress", "sleep"], [("stress", "sleep")])
    candidate, message = validate_model_submission(model.model_dump(mode="json"), measurements=True)
    assert message == "VALID"
    assert ModelSpec.model_validate(candidate) == model
    partial = model.revised(
        edges=replace_constructs(
            model.edges, tuple(c.model_copy(update={"indicators": ()}) for c in model.constructs)
        ),
        measurement_clock=None,
    )
    assert validate_model_submission(partial.model_dump(mode="json"))[1] == "VALID"
    result, error = validate_model_submission(partial.model_dump(mode="json"), measurements=True)
    assert result is None
    assert "clock and indicators" in error


@pytest.mark.parametrize(
    "payload",
    [
        "not a dict",
        {"edges": []},
        {"constructs": "bad"},
        {"constructs": [42]},
        {"constructs": [{"name": "bad"}]},
    ],
)
def test_invalid_authored_structure_returns_errors(payload):
    candidate, error = validate_model_submission(invalid_dict_payload(payload))
    assert candidate is None
    assert error.startswith("VALIDATION ERRORS:")


@pytest.mark.parametrize("bad", ["bad", [42], [{"name": "bad"}]])
def test_invalid_owned_indicator_returns_errors(bad):
    data = make_model(["stress"]).model_dump(mode="json")
    graph_constructs(data)[0]["indicators"] = bad
    result, error = validate_model_submission(data, measurements=True)
    assert result is None
    assert "indicators" in error


@pytest.mark.parametrize("kind", ["indicator", "edge"])
def test_duplicate_entity_identity_is_rejected(kind):
    data = make_model(["stress", "sleep"], [("stress", "sleep")]).model_dump(mode="json")
    collection = (
        graph_constructs(data)[0]["indicators"] if kind == "indicator" else data[kind + "s"]
    )
    collection.append(collection[0].copy())
    result, error = validate_model_submission(data)
    assert result is None
    assert f"Duplicate {kind} IDs" in error


def test_duplicate_names_with_distinct_identities_are_rejected():
    data = make_model(["stress", "sleep"], [("stress", "sleep")]).model_dump(mode="json")
    graph_constructs(data)[1]["name"] = "stress"
    assert "Duplicate construct names" in validate_model_submission(data)[1]
    data = make_model(["stress", "sleep"], [("stress", "sleep")]).model_dump(mode="json")
    graph_constructs(data)[1]["indicators"][0]["name"] = "stress_obs"
    assert "Duplicate indicator names" in validate_model_submission(data)[1]


def test_validation_collects_errors_from_multiple_entities():
    data = {
        "edges": [
            {
                "id": "edge:invalid",
                "description": "Invalid endpoints",
                "cause": {"name": "bad1"},
                "effect": {"name": "bad2"},
            }
        ]
    }
    with pytest.raises(ValidationError) as exc:
        ModelSpec.model_validate(data)
    assert {error["loc"][2] for error in exc.value.errors() if len(error["loc"]) > 2} == {
        "cause",
        "effect",
    }
    assert validate_model_submission(data)[0] is None


def test_edge_payload_must_be_a_valid_entity():
    data = make_model(["stress"]).model_dump(mode="json")
    data["edges"] = ["not a dict"]
    result, error = validate_model_submission(data)
    assert result is None
    assert "edges" in error
