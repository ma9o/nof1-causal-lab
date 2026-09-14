"""Broad contracts for advisory findings, source attribution, and CLI output."""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pytest
from scripts.type_system_analysis import analyze_type_graph
from scripts.type_system_usage import collect_source_evidence
from scripts.visualize_type_system import build_type_graph, compact_graph

from scripts import visualize_type_system

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject


def _ref(name: str) -> JsonObject:
    return {"$ref": f"#/$defs/{name}"}


def _record(name: str, properties: JsonObject) -> JsonObject:
    return {
        "title": name,
        "description": f"{name} represents an example domain concept.",
        "type": "object",
        "properties": properties,
        "required": sorted(properties),
        "x-layer": "authored",
        "x-concern": "scientific_model",
        "x-python-module": "domain.models",
    }


@pytest.fixture
def analysis_workspace(tmp_path: Path) -> tuple[JsonObject, Path]:
    source_root = tmp_path / "apps/data-pipeline/src"
    package = source_root / "domain"
    package.mkdir(parents=True)
    package.joinpath("models.py").write_text(
        dedent("""
        class Wrapper:
            def validate(self):
                return self.payload

        class Split:
            def validate(self):
                return self.alpha + self.beta + self.gamma + self.delta

        class Choice:
            @field_validator("target", mode="before")
            def resolve_target(cls, value):
                return value
    """)
    )
    package.joinpath("consumers.py").write_text(
        dedent("""
        from typing import TYPE_CHECKING, Annotated
        if TYPE_CHECKING:
            from .models import Split as Parts

        def first(value: Parts):
            return value.alpha + value.beta

        def second(value: "Parts"):
            return value.alpha - value.beta

        def third(value: Annotated[Parts, "note"]):
            return value.gamma + value.delta

        def fourth(value: Parts | None):
            return value.gamma - value.delta

        def untyped(value):
            return value.alpha + value.gamma

        def container(value: list[Parts]):
            return value.alpha + value.gamma

        def rebound(value: Parts):
            value = unknown()
            return value.alpha + value.gamma

        def nested(value: Parts):
            def inner(value):
                return value.alpha + value.gamma
            return inner(value)
    """)
    )
    common: JsonObject = {
        "x": {"type": "number", "minimum": 0},
        "y": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
    }
    definitions: JsonObject = {
        "Wrapper": _record("Wrapper", {"payload": _ref("Payload")}),
        "Payload": _record("Payload", {"value": {"type": "number"}}),
        "TwinA": _record("TwinA", {**common, "kind": {"const": "a"}}),
        "TwinB": _record("TwinB", {**common, "kind": {"const": "b"}}),
        "Different": _record(
            "Different",
            {
                "x": {"type": "number", "minimum": -1},
                "y": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
            },
        ),
        "OptionalFields": {**_record("OptionalFields", common), "required": []},
        "OtherDefaults": _record(
            "OtherDefaults",
            {
                "x": {"type": "number", "minimum": 0, "default": 1},
                "y": {"type": "array", "items": {"type": "integer"}, "minItems": 1, "default": [1]},
            },
        ),
        "Split": _record(
            "Split", {name: {"type": "number"} for name in ("alpha", "beta", "gamma", "delta")}
        ),
        "Choice": _record(
            "Choice",
            {
                "target": {"anyOf": [_ref("ItemId"), _ref("Entity"), {"type": "null"}]},
                "count": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "objects": {"oneOf": [_ref("TwinA"), _ref("TwinB")]},
                "by_id": {
                    "type": "object",
                    "additionalProperties": {"oneOf": [_ref("MapArm"), _ref("TwinA")]},
                },
                "items": {
                    "type": "array",
                    "items": {"oneOf": [_ref("ArrayArm"), _ref("TwinB")]},
                },
                "optional": {"anyOf": [_ref("NullableWrapper"), {"type": "null"}]},
            },
        ),
        "MapArm": _record("MapArm", {"reason": {"type": "string"}}),
        "ArrayArm": _record("ArrayArm", {"amount": {"type": "number"}}),
        "NullableWrapper": _record("NullableWrapper", {"label": {"type": "string"}}),
        "ItemId": {
            "type": "string",
            "description": "An identity names an entity.",
            "x-layer": "identity",
            "x-concern": "identity",
        },
        "Entity": _record("Entity", {"id": _ref("ItemId"), "label": {"type": "string"}}),
        "Expression": {
            "description": "An expression recursively combines alternatives.",
            "oneOf": [_ref("Leaf"), _ref("Branch")],
            "x-layer": "authored",
            "x-concern": "scientific_model",
        },
        "Leaf": _record("Leaf", {"kind": {"const": "leaf"}, "value": {"type": "number"}}),
        "Branch": _record("Branch", {"child": _ref("Expression")}),
    }
    definitions["Root"] = _record(
        "Root",
        {
            name.lower(): _ref(name)
            for name in definitions
            if name not in {"Leaf", "Branch", "ItemId", "MapArm", "ArrayArm", "NullableWrapper"}
        },
    )
    return {"$defs": definitions, "properties": {"root": _ref("Root")}}, source_root


def test_four_reports_use_schema_constraints_and_attributed_source_evidence(analysis_workspace):
    schema, source_root = analysis_workspace
    graph = build_type_graph(schema)
    source = collect_source_evidence(graph, source_root)
    wrappers, repeated, clusters, alternatives = analyze_type_graph(graph, source, set(graph))

    wrapper_names = {name for candidate in wrappers.candidates for name in candidate.types}
    assert "Wrapper" in wrapper_names
    assert "NullableWrapper" in wrapper_names
    assert not {"Payload", "Leaf", "Branch", "ItemId", "Root", "MapArm", "ArrayArm"} & wrapper_names
    wrapper = next(
        candidate for candidate in wrappers.candidates if candidate.types == ("Wrapper",)
    )
    assert any("validate" in note for note in wrapper.evidence)
    assert [candidate.types for candidate in repeated.candidates] == [("TwinA", "TwinB")]
    assert [candidate.types for candidate in clusters.candidates] == [("Split",)]
    assert any("{alpha, beta}" in note for note in clusters.candidates[0].evidence)
    assert any("{delta, gamma}" in note for note in clusters.candidates[0].evidence)
    assert len(alternatives.candidates) == 1
    assert "Choice.target" in alternatives.candidates[0].summary
    assert any("resolve_target" in note for note in alternatives.candidates[0].evidence)

    external = [use for use in source.uses["Split"] if not use.internal]
    assert len(external) == 4
    assert {use.fields for use in external} == {
        frozenset({"alpha", "beta"}),
        frozenset({"gamma", "delta"}),
    }


def test_analysis_preserves_global_ownership_and_ignores_visual_compaction(analysis_workspace):
    schema, source_root = analysis_workspace
    graph = build_type_graph(schema)
    original_edges = list(graph.edges(data=True))
    source = collect_source_evidence(graph, source_root)
    compact_graph(graph)
    reports = analyze_type_graph(graph, source, {"Payload"})
    assert not any(report.candidates for report in reports)
    assert list(graph.edges(data=True)) == original_edges

    source_root.joinpath("domain/bridge.py").write_text(
        "from .models import Split\ndef bridge(value: Split):\n    return value.alpha + value.gamma\n"
    )
    source = collect_source_evidence(graph, source_root)
    assert not analyze_type_graph(graph, source, {"Split"})[2].candidates


def test_cli_refreshes_diagram_and_saves_complete_reports_with_bounded_stdout(
    analysis_workspace, tmp_path, monkeypatch, capsys
):
    schema, _ = analysis_workspace
    exporter = ModuleType("scripts.export_schemas")
    exporter.__dict__["export_schemas"] = lambda: schema
    monkeypatch.setitem(sys.modules, "scripts.export_schemas", exporter)
    monkeypatch.setattr(visualize_type_system, "REPO_ROOT", tmp_path)

    def _render_dot(command, *, check):
        assert check is True
        assert command[:2] == ["dot", "-Tsvg"]
        assert "digraph" in Path(command[2]).read_text()
        Path(command[4]).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')

    monkeypatch.setattr(visualize_type_system.subprocess, "run", _render_dot)
    stem = tmp_path / "custom-map"
    stem.with_suffix(".svg").write_text("stale")
    visualize_type_system.main(["--output", str(stem), "--limit", "1"])
    output = capsys.readouterr().out
    full_report = stem.with_suffix(".analysis.md").read_text()
    assert "stale" not in stem.with_suffix(".svg").read_text()
    assert full_report.count("\n## ") == 4
    assert "Showing 1 of" in output
    assert "Showing 1 of" not in full_report
    assert "SVG refreshed:" in output
    assert "Complete analysis:" in output

    visualize_type_system.main(["--output", str(stem), "--detail", "full", "--limit", "0"])
    assert stem.with_suffix(".analysis.md").read_text() == full_report
    assert "Showing 1 of" not in capsys.readouterr().out

    visualize_type_system.main(["--root", "Wrapper", "--output", str(stem)])
    focused = stem.with_suffix(".analysis.md").read_text()
    assert "# Type-system analysis: Wrapper" in focused
    assert "Wrapper: review" in focused
    assert "Payload: review" not in focused


def test_field_signature_normalization_preserves_nested_property_names_and_nominal_ids(
    analysis_workspace,
):
    schema, source_root = analysis_workspace
    definitions = cast("dict[str, JsonObject]", schema["$defs"])
    for name, text_type in (("TwinA", "string"), ("TwinB", "number")):
        definitions[name] = _record(
            name,
            {
                "nested": {"type": "object", "properties": {"description": {"type": text_type}}},
                "id": _ref("ItemId") if name == "TwinA" else {"type": "string"},
            },
        )
    graph = build_type_graph(schema)
    source = collect_source_evidence(graph, source_root)
    assert not analyze_type_graph(graph, source, {"TwinA", "TwinB"})[1].candidates
