"""Offline identity changes preserve retained laws, coordinates, and scalar findings."""

from copy import deepcopy

from scripts.migrate_mechanism_identity import identify_mechanisms, remap_references

from nof1_causal_lab.artifacts.identity import scientific_id


def test_anonymous_conversion_preserves_retained_numerical_inputs_and_outputs():
    # Author the historical schema directly. Reversing today's DEMO model through
    # several unrelated schema migrations is neither lossless nor this contract.
    construct_id = scientific_id("construct", "signal")
    edge_id = scientific_id("edge", "signal-to-outcome")
    payload = {"constructs": [], "edges": [], "parameters": []}
    expected = deepcopy(payload)
    mapping: dict[str, str] = {}
    for field, terms, owner_kind, owner_id, kind, slot, quantity in (
        (
            "constructs",
            "dynamics",
            "construct",
            construct_id,
            "node_potential",
            "stiffness",
            "drift_diag",
        ),
        ("edges", "mechanisms", "edge", edge_id, "linear", "weight", "drift_offdiag"),
    ):
        mechanism_id = scientific_id("mechanism", [owner_id, kind])
        old = scientific_id("parameter", [quantity, [owner_id]])
        new = scientific_id("parameter", [quantity, sorted([owner_id, mechanism_id])])
        old_element = scientific_id("element", [old, "scalar"])
        new_element = scientific_id("element", [new, "scalar"])
        mapping.update({old: new, old_element: new_element})
        term = {
            "kind": kind,
            slot: {"kind": "estimated", "parameter_id": old},
            "offset": {"kind": "fixed", "value": 0.25},
        }
        parameter = {
            "id": old,
            "quantity": quantity,
            "owners": [{"kind": owner_kind, "id": owner_id}],
            "prior": {"distribution": "Normal", "params": {"loc": -0.2, "scale": 0.1}},
        }
        payload[field].append({"id": owner_id, terms: [term]})
        payload["parameters"].append(parameter)
        expected[field].append(
            {
                "id": owner_id,
                terms: [
                    {**term, "id": mechanism_id, slot: {"kind": "estimated", "parameter_id": new}}
                ],
            }
        )
        expected["parameters"].append(
            {
                **parameter,
                "id": new,
                "owners": [*parameter["owners"], {"kind": "mechanism", "id": mechanism_id}],
            }
        )

    original = deepcopy(payload)
    migrated, actual_mapping = identify_mechanisms(payload)
    assert payload == original
    assert actual_mapping == mapping
    assert migrated == expected

    # Retained coordinates occur in both keys and values. Arbitrary prose and
    # numerical results must survive without substring replacement or refitting.
    retained = {
        "draws": {old: [0.12, -0.05]},
        "coordinates": [{"parameter_id": old, "element_id": old_element}],
        "finding": {"mean": 0.035, "interval": [-0.05, 0.12]},
        "label": f"literal {old}",
    }
    assert remap_references(retained, mapping) == {
        **retained,
        "draws": {new: [0.12, -0.05]},
        "coordinates": [{"parameter_id": new, "element_id": new_element}],
    }
