"""Offline identity changes preserve retained laws, coordinates, and scalar findings."""

import json
from pathlib import Path

from scripts.migrate_mechanism_identity import (
    identify_mechanisms,
    remap_references,
    retired_identity_map,
)

from nof1_causal_lab.artifacts.model_spec import ModelSpec


def test_anonymous_fixture_conversion_preserves_retained_numerical_inputs_and_outputs():
    root = Path(__file__).resolve().parents[4] / "data/DEMO/fixture/artifacts"
    original = json.loads((root / "model.json").read_text())
    model = ModelSpec.model_validate(original)
    mapping = retired_identity_map(model)
    reverse = {new: old for old, new in mapping.items()}
    from scripts.migrate_component_slots import convert_component_slots

    from tests.slot_fixtures import flat_catalogue_payload

    anonymous = remap_references(flat_catalogue_payload(model), reverse)
    for field, terms in (("constructs", "dynamics"), ("edges", "mechanisms")):
        for owner in anonymous[field]:
            for term in owner[terms]:
                term.pop("id")
    for parameter in anonymous["parameters"]:
        parameter["owners"] = [
            owner for owner in parameter["owners"] if owner["kind"] != "mechanism"
        ]
    migrated, actual_mapping = identify_mechanisms(anonymous)
    assert ModelSpec.model_validate(convert_component_slots(migrated)) == model
    assert actual_mapping == mapping
    for filename in ("baseline_report.json",):
        retained = json.loads((root / filename).read_text())
        old = remap_references(retained, reverse)
        converted = remap_references(old, mapping)
        assert converted == retained
