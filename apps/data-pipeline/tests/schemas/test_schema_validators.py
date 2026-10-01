"""Whole-model validation reports intrinsic errors; measurement readiness is a separate requirement."""

import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.compilation_errors import IncompleteModelError
from tests.helpers import graph_constructs, invalid_dict_payload, make_model

pytestmark = pytest.mark.contract


def test_valid_partial_model_can_be_enriched_for_measurement():
    model = make_model(["stress", "sleep"], [("stress", "sleep")])
    candidate = ModelSpec.model_validate(model.model_dump(mode="json"))
    candidate.require_measurements()
    assert candidate == model
    partial = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                type(c).model_validate({**c.model_dump(), "indicators": ()})
                for c in model.constructs
            ),
        ),
        measurement_clock=None,
    )
    reloaded = ModelSpec.model_validate(partial.model_dump(mode="json"))
    with pytest.raises(IncompleteModelError, match="clock and indicators"):
        reloaded.require_measurements()


@pytest.mark.parametrize("payload", ["not a dict", {"constructs": "bad"}])
def test_invalid_authored_structure_returns_errors(payload):
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(invalid_dict_payload(payload))


@pytest.mark.parametrize("bad", ["bad", [42], [{"name": "bad"}]])
def test_invalid_owned_indicator_returns_errors(bad):
    data = make_model(["stress"]).model_dump(mode="json")
    graph_constructs(data)[0]["indicators"] = bad
    with pytest.raises(ValidationError, match="indicators"):
        ModelSpec.model_validate(data)


@pytest.mark.parametrize("kind", ["indicator", "edge"])
def test_duplicate_entity_identity_is_rejected(kind):
    data = make_model(["stress", "sleep"], [("stress", "sleep")]).model_dump(mode="json")
    collection = (
        graph_constructs(data)[0]["indicators"] if kind == "indicator" else data[kind + "s"]
    )
    collection.append(collection[0].copy())
    with pytest.raises(ValidationError, match=f"Duplicate {kind} IDs"):
        ModelSpec.model_validate(data)


def test_duplicate_names_with_distinct_identities_are_rejected():
    data = make_model(["stress", "sleep"], [("stress", "sleep")]).model_dump(mode="json")
    graph_constructs(data)[1]["name"] = "stress"
    with pytest.raises(ValidationError, match="Duplicate construct names"):
        ModelSpec.model_validate(data)
    data = make_model(["stress", "sleep"], [("stress", "sleep")]).model_dump(mode="json")
    graph_constructs(data)[1]["indicators"][0]["name"] = "stress_obs"
    with pytest.raises(ValidationError, match="Duplicate indicator names"):
        ModelSpec.model_validate(data)


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


def test_edge_payload_must_be_a_valid_entity():
    data = make_model(["stress"]).model_dump(mode="json")
    data["edges"] = ["not a dict"]
    with pytest.raises(ValidationError, match="edges"):
        ModelSpec.model_validate(data)
