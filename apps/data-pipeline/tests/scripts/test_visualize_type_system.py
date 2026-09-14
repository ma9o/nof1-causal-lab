"""The visualization reflects schema dependencies without losing union branches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from scripts.visualize_type_system import build_type_graph, compact_graph, graph_dot, select_view

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject


def test_graph_traverses_union_and_container_references_and_filters_roots():
    schema: JsonObject = {
        "$defs": {
            "Root": {
                "x-layer": "read_models",
                "x-concern": "read_models",
                "description": "The root combines A & B into one view.\n\nAdditional implementation details.",
                "properties": {
                    "entities": {"items": {"oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]}}
                },
            },
            "A": {
                "x-layer": "read_models",
                "x-concern": "read_models",
                "description": "A represents an owned entity.",
                "properties": {"owner": {"$ref": "#/$defs/Owner"}},
            },
            "B": {
                "description": "B represents an independent entity.",
                "x-layer": "findings",
                "x-concern": "read_models",
            },
            "Owner": {
                "description": "An owner identifies the containing model.",
                "x-layer": "identity",
                "x-concern": "identity",
            },
            "Unrelated": {
                "description": "This type belongs to another part of the API.",
                "x-layer": "read_models",
                "x-concern": "read_models",
            },
        }
    }
    graph = build_type_graph(schema, "Root")
    assert set(graph) == {"Root", "A", "B", "Owner"}
    assert set(graph.edges) == {("Root", "A"), ("Root", "B"), ("A", "Owner")}
    assert graph.edges["Root", "A"]["fields"] == {"entities"}
    dot = graph_dot(graph, "Types")
    assert '"A" -> "Owner" [xlabel="owner", constraint=false]' in dot
    label = dot.split('"Root" [label=<', 1)[1].split(">, fillcolor", 1)[0]
    assert "<B>Root</B>" in label
    assert "The root combines A &amp; B into one view." in label
    assert "Additional implementation details." not in label
    assert "Additional implementation details." in dot


def test_recursive_types_are_not_misrepresented_as_a_dag():
    schema: JsonObject = {
        "$defs": {
            "Node": {
                "x-layer": "transport",
                "x-concern": "api_tools",
                "description": "A node points to the next item in a recursive chain.",
                "properties": {"next": {"$ref": "#/$defs/Node"}},
            }
        }
    }
    graph = build_type_graph(schema)
    assert graph.has_edge("Node", "Node")


@pytest.mark.parametrize("description", [None, "", "A label without a sentence"])
def test_types_require_an_opening_role_sentence(description):
    schema: JsonObject = {"$defs": {"Node": {"description": description}}}
    with pytest.raises(ValueError, match="Type 'Node' needs"):
        build_type_graph(schema)


@pytest.fixture
def projection_schema() -> JsonObject:
    def _record(
        name: str,
        properties: JsonObject,
        *,
        concern: str = "read_models",
        layer: str = "read_models",
    ) -> JsonObject:
        return {
            "description": f"{name} describes a typed contract.",
            "type": "object",
            "properties": properties,
            "x-layer": layer,
            "x-concern": concern,
        }

    def _ref(name: str) -> JsonObject:
        return {"$ref": f"#/$defs/{name}"}

    def _sourced(value: JsonObject) -> JsonObject:
        return {
            **_record("Sourced", {"value": value, "source": _ref("FactSource")}),
            "title": "Sourced[TestValue]",
            "x-python-module": "nof1_causal_lab.machine.snapshot_models",
        }

    return {
        "properties": {"report": _ref("IdentificationReport")},
        "$defs": {
            "ModelSnapshot": _record(
                "ModelSnapshot",
                {
                    "raw": _ref("Record"),
                    "estimate": {"anyOf": [_ref("Sourced_Record_"), {"type": "null"}]},
                    "alternatives": {"type": "array", "items": _ref("Sourced_Record_")},
                    "counts": {"type": "array", "items": _ref("Sourced_int_")},
                    "nullable_id": {"anyOf": [_ref("ConstructId"), {"type": "null"}]},
                    "lookup": {
                        "type": "object",
                        "propertyNames": _ref("ConstructId"),
                        "additionalProperties": _ref("Mode"),
                    },
                },
            ),
            "Record": _record(
                "Record",
                {
                    "id": _ref("ConstructId"),
                    "mode": _ref("Mode"),
                    "expression": _ref("WindowExpression"),
                    "coordinate": _ref("ParameterCoordinate"),
                    "source": _ref("FactSource"),
                    "next": _ref("Record"),
                },
            ),
            "ParameterCoordinate": _record(
                "ParameterCoordinate",
                {"site": {"type": "string"}},
                concern="identity",
                layer="identity",
            ),
            "ConstructId": {
                "type": "string",
                "description": "A construct identity names a persistent entity.",
                "x-concern": "identity",
                "x-layer": "identity",
            },
            "WindowExpression": {
                "type": "string",
                "description": "A validated support-window expression.",
                "x-concern": "scientific_model",
                "x-layer": "authored",
            },
            "Mode": {
                "type": "string",
                "enum": ["retained", "excluded"],
                "description": "A mode records an allowed disposition.",
                "x-concern": "scientific_model",
                "x-layer": "authored",
            },
            "FactSource": _record("FactSource", {"pointer": {"type": "string"}}),
            "Sourced_Record_": _sourced(_ref("Record")),
            "Sourced_int_": _sourced({"type": "integer"}),
            "SimilarShape": _record(
                "SimilarShape", {"value": _ref("Record"), "source": _ref("FactSource")}
            ),
            "IdentificationReport": _record(
                "IdentificationReport",
                {"finding": _ref("Record")},
                concern="scientific_model",
                layer="findings",
            ),
            "RunOperation": _record(
                "RunOperation", {}, concern="execution_provenance", layer="machine"
            ),
            "Response": _record(
                "Response",
                {"move": _ref("RunOperation"), "result": _ref("Record")},
                concern="api_tools",
                layer="transport",
            ),
        },
    }


def test_compact_retains_field_types_provenance_and_distinct_reference_paths(projection_schema):
    full = build_type_graph(projection_schema)
    compact = compact_graph(full)
    assert (
        not {"ConstructId", "Mode", "WindowExpression", "Sourced_Record_", "Sourced_int_"}
        & compact.nodes
    )
    assert {"Record", "ParameterCoordinate", "FactSource", "SimilarShape"} <= compact.nodes
    assert compact.edges["ModelSnapshot", "Record"] == {
        "fields": {"raw"},
        "sourced_fields": {"estimate.value", "alternatives[].value"},
    }
    assert compact.has_edge("Record", "Record")
    assert not compact.has_edge("Record", "FactSource")
    assert "source: FactSource" in compact.nodes["Record"]["annotations"]
    assert "expression: WindowExpression" in compact.nodes["Record"]["annotations"]
    assert "validated support-window expression" in compact.nodes["Record"]["description"]
    assert set(compact.nodes["ModelSnapshot"]["annotations"]) == {
        "counts: list<Sourced<integer>>",
        "lookup: map<ConstructId, Mode>",
        "nullable_id: ConstructId | null",
    }
    assert (
        "estimate: Sourced<Record> | null; source: FactSource"
        in compact.nodes["ModelSnapshot"]["description"]
    )
    assert 'Mode values: "retained" | "excluded"' in compact.nodes["Record"]["description"]
    assert full.has_edge("Record", "FactSource")
    assert full.has_edge("ModelSnapshot", "Sourced_Record_")
    assert full.edges["ModelSnapshot", "Record"] == {"fields": {"raw"}}
    assert "annotations" not in full.nodes["ModelSnapshot"]


@pytest.mark.parametrize("root", ["ConstructId", "Mode", "WindowExpression", "Sourced_Record_"])
def test_compact_retains_explicitly_requested_roots(projection_schema, root):
    assert root in compact_graph(build_type_graph(projection_schema, root))


def test_sourced_union_arrays_preserve_every_target_and_container_path(projection_schema):
    projection_schema["$defs"]["Sourced_Record_"]["properties"]["value"] = {
        "type": "array",
        "items": {"oneOf": [{"$ref": "#/$defs/Record"}, {"$ref": "#/$defs/SimilarShape"}]},
    }
    compact = compact_graph(build_type_graph(projection_schema))
    for target in ("Record", "SimilarShape"):
        assert compact.edges["ModelSnapshot", target]["sourced_fields"] == {
            "estimate.value[]",
            "alternatives[].value[]",
        }


def test_artifact_view_uses_registered_payloads_and_machine_view_marks_its_boundary(
    projection_schema,
):
    graph = build_type_graph(projection_schema)
    stored = select_view(graph, "artifacts")
    assert stored.graph["roots"] == {"IdentificationReport"}
    assert "Record" in stored
    assert not {"RunOperation", "Response", "ModelSnapshot"} & stored.nodes
    machine = select_view(graph, "machine")
    assert set(machine) == {"RunOperation", "Response", "Record"}
    assert set(machine.edges) == {("Response", "RunOperation"), ("Response", "Record")}
    assert machine.nodes["Record"]["external"] is True
    assert "external" not in graph.nodes["Record"]
    semantic = select_view(graph, "semantic")
    assert semantic.graph["roots"] == {"ModelSnapshot"}
    assert not {"Response", "RunOperation", "IdentificationReport"} & semantic.nodes
