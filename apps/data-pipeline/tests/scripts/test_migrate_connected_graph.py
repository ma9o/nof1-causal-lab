"""Offline graph conversion preserves identities without inventing scientific edges."""

from copy import deepcopy

import pytest
from scripts.migrate_connected_graph import connect_endpoints

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from tests.helpers import make_model


def _catalogue():
    model = make_model(["A", "B", "Y"], [("A", "Y"), ("B", "Y")])
    return {
        **model.model_dump(mode="json", exclude={"edges"}),
        "constructs": [construct.model_dump(mode="json") for construct in model.constructs],
        "edges": [
            {
                **edge.model_dump(mode="json", exclude={"cause", "effect"}),
                "cause_id": edge.cause.id,
                "effect_id": edge.effect.id,
            }
            for edge in model.edges
        ],
    }


def test_offline_conversion_preserves_the_source_and_shared_endpoint_identity():
    source = _catalogue()
    original = deepcopy(source)
    converted = connect_endpoints(source)
    model = ModelSpec.model_validate(converted)
    assert source == original
    assert "constructs" not in converted
    assert [construct.id for construct in model.constructs] == [
        construct["id"] for construct in source["constructs"]
    ]
    assert model.edges[0].effect is model.edges[1].effect
    edges = converted["edges"]
    assert isinstance(edges, list)
    shared = edges[1]
    assert isinstance(shared, dict)
    assert shared["effect"] == {
        "kind": "construct",
        "id": model.edges[0].effect.id,
    }


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("isolated", "Isolated constructs"),
        ("undefined", "Undefined construct endpoint"),
        ("duplicate", "unique identities"),
    ],
)
def test_offline_conversion_rejects_invalid_membership(problem, message):
    source = _catalogue()
    if problem == "isolated":
        source["edges"].pop()
    elif problem == "undefined":
        source["edges"][0]["cause_id"] = "construct:unknown"
    else:
        source["constructs"].append(source["constructs"][0])
    with pytest.raises(ValueError, match=message):
        connect_endpoints(source)
