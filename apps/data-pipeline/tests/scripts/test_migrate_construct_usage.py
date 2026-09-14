"""Offline migration preserves scientific identity and requires reviewed active declarations."""

import pytest
from scripts.migrate_construct_usage import convert_construct_usage

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from tests.helpers import graph_constructs, make_model


def test_usage_migration_requires_scientific_revisions_without_mutating_source():
    model = make_model(["X", "Y"], [("X", "Y")])
    payload = model.model_dump(mode="json")
    owner = graph_constructs(payload)[0]
    owner["usage"] = None
    assert ModelSpec.model_validate(convert_construct_usage(payload)) == model
    owner["usage"] = {"kind": "known_input", "source_indicator_id": owner["indicators"][0]["id"]}
    with pytest.raises(ValueError, match="reviewed scientific revisions"):
        convert_construct_usage(payload)
    replacement = model.constructs[0].model_dump(mode="json")
    restored = ModelSpec.model_validate(
        convert_construct_usage(payload, revisions={owner["id"]: replacement})
    )
    assert restored == model
    assert owner["usage"]["kind"] == "known_input"
    with pytest.raises(ValueError, match="preserve construct IDs"):
        convert_construct_usage(
            payload, revisions={owner["id"]: {**replacement, "id": "construct:wrong"}}
        )
