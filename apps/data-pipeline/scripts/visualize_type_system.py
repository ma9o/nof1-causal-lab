"""Render the exported backend type dependency graph as Graphviz DOT and SVG.

Run from the repository root with ``bun run types:graph`` for the compact map.
Use ``--view semantic``, ``--view artifacts``, or ``--view machine`` to focus on
one interface. ``--detail full`` keeps every type and its opening role sentence;
``--root TypeName`` follows any individual type's dependencies.
The scientific model groups data, model choices, checks, inference, and analysis.
Colors describe conceptual roles. Cross-group references do not set layout ranks.
Red borders and titles highlight central domain objects; hover for their role.
Edges mean "has a field referencing this type", not causal relationships.
Requires the Graphviz ``dot`` executable.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from html import escape
from pathlib import Path
from textwrap import wrap
from typing import TYPE_CHECKING

import networkx as nx

from scripts.type_system_catalog import (
    CONCERNS,
    CORE_DOMAIN_OBJECTS,
    LAYERS,
    SCIENTIFIC_MODEL_GROUPS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.json_types import JsonObject, JsonValue

REPO_ROOT = Path(__file__).resolve().parents[3]
VIEWS = {
    "all": ("Exported backend contracts", "backend-types"),
    "semantic": ("Semantic model", "semantic-model"),
    "artifacts": ("Stored artifact payloads", "stored-artifacts"),
    "machine": ("Machine & transport", "machine-transport"),
}
CORE_DOMAIN_STYLE = 'color="#dc2626", fontcolor="#b91c1c", penwidth=2.8'


def _reference_paths(value: JsonValue, path: str = "") -> dict[str, set[str]]:
    """Locate local references, retaining the containers traversed by each path."""
    paths: dict[str, set[str]] = {}

    def visit(value: JsonValue, path: str) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item, path)
        elif isinstance(value, dict):
            for key, item in value.items():
                if key == "$ref":
                    if not isinstance(item, str) or not item.startswith("#/$defs/"):
                        raise ValueError(f"Expected a local type reference, received {item!r}")
                    paths.setdefault(item.removeprefix("#/$defs/"), set()).add(path)
                elif key == "properties" and isinstance(item, dict):
                    for field, definition in item.items():
                        visit(definition, f"{path}.{field}" if path else field)
                elif key == "prefixItems" and isinstance(item, list):
                    for index, definition in enumerate(item):
                        visit(definition, f"{path}[{index}]")
                else:
                    suffix = {
                        "items": "[]",
                        "additionalProperties": "{value}",
                        "propertyNames": "{key}",
                    }.get(key, "")
                    visit(item, path + suffix)

    visit(value, path)
    return paths


def _references(value: JsonValue) -> set[str]:
    return set(_reference_paths(value))


def build_type_graph(schema: Mapping[str, JsonValue], root: str | None = None) -> nx.DiGraph:
    """Discover actual schema dependencies, including nested containers and unions."""
    exported = schema["$defs"]
    if not isinstance(exported, dict):
        raise ValueError("The exported schema must contain an object-valued $defs")
    definitions: dict[str, JsonObject] = {}
    for name, definition in exported.items():
        if not isinstance(definition, dict):
            raise ValueError(f"Expected an object schema for {name}")
        definitions[name] = definition
    graph = nx.DiGraph()
    for name, definition in definitions.items():
        description = definition.get("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Type {name!r} needs a description explaining its role")
        summary = " ".join(description.strip().split("\n\n", 1)[0].split())
        if not re.search(r"[.!?](?:\s|$)", summary):
            raise ValueError(f"Type {name!r} needs a complete opening role sentence")
        graph.add_node(
            name,
            description=description,
            summary=summary,
            layer=definition["x-layer"],
            concern=definition["x-concern"],
            schema=definition,
        )
    for name, definition in definitions.items():
        properties = definition.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(f"Expected object properties for {name}")
        for field, value in properties.items():
            for target in _references(value):
                if target not in definitions:
                    raise ValueError(f"{name}.{field} references missing type {target}")
                labels = graph.get_edge_data(name, target, {}).get("fields", set())
                graph.add_edge(name, target, fields=labels | {field})
        for target in _references(
            {key: value for key, value in definition.items() if key != "properties"}
        ):
            if target not in definitions:
                raise ValueError(f"{name} references missing type {target}")
            labels = graph.get_edge_data(name, target, {}).get("fields", set())
            graph.add_edge(name, target, fields=labels | {"variant"})
    # These top-level schema references are exported from ARTIFACT_CONTRACTS.
    graph.graph["artifact_roots"] = _references(schema.get("properties", {}))
    graph.graph["roots"] = {name for name, degree in graph.in_degree if degree == 0}
    if root is not None:
        graph = _dependency_closure(graph, {root})
    return graph


def _dependency_closure(graph: nx.DiGraph, roots: set[str]) -> nx.DiGraph:
    missing = roots - graph.nodes
    if missing:
        raise ValueError(f"Unknown root types: {', '.join(sorted(missing))}")
    included = roots | set().union(*(nx.descendants(graph, root) for root in roots))
    selected = graph.subgraph(included).copy()
    selected.graph["roots"] = roots
    return selected


def select_view(graph: nx.DiGraph, view: str) -> nx.DiGraph:
    """Select complete dependency closures or the machine's explicit type boundary."""
    if view == "all":
        return graph.copy()
    if view == "semantic":
        return _dependency_closure(graph, {"ModelSnapshot"})
    if view == "artifacts":
        return _dependency_closure(graph, graph.graph["artifact_roots"])
    if view != "machine":
        raise ValueError(f"Unknown type-system view {view!r}")
    owned = {
        name
        for name, data in graph.nodes(data=True)
        if data["concern"] in {"execution_provenance", "api_tools"}
    }
    boundary = {target for source in owned for target in graph.successors(source)} - owned
    selected = graph.subgraph(owned | boundary).copy()
    selected.remove_edges_from(list(selected.out_edges(boundary)))
    nx.set_node_attributes(selected, dict.fromkeys(boundary, True), "external")
    selected.graph["roots"] = {
        name for name in owned if not any(parent in owned for parent in graph.predecessors(name))
    }
    return selected


def _is_sourced(definition: JsonObject) -> bool:
    title = definition.get("title")
    return (
        definition.get("x-python-module") == "nof1_causal_lab.machine.snapshot_models"
        and isinstance(title, str)
        and title.startswith("Sourced[")
    )


def _type_label(value: JsonValue, graph: nx.DiGraph) -> str:
    """Describe folded fields without losing nullability, containers, or union arms."""
    if not isinstance(value, dict):
        raise ValueError(f"Expected a field schema, received {value!r}")
    if "$ref" in value:
        name = next(iter(_references({"$ref": value["$ref"]})))
        definition = graph.nodes[name]["schema"]
        if _is_sourced(definition):
            return f"Sourced<{_type_label(definition['properties']['value'], graph)}>"
        return name
    for keyword in ("oneOf", "anyOf"):
        branches = value.get(keyword)
        if isinstance(branches, list):
            return " | ".join(_type_label(branch, graph) for branch in branches)
    values = value.get("enum")
    if isinstance(values, list):
        return " | ".join(json.dumps(item, ensure_ascii=False) for item in values)
    if "const" in value:
        return json.dumps(value["const"], ensure_ascii=False)
    if value.get("type") == "array":
        prefix = value.get("prefixItems")
        if isinstance(prefix, list):
            return "tuple<" + ", ".join(_type_label(item, graph) for item in prefix) + ">"
        return f"list<{_type_label(value['items'], graph)}>"
    items = value.get("additionalProperties")
    if value.get("type") == "object" and isinstance(items, dict):
        keys = _type_label(value["propertyNames"], graph) if "propertyNames" in value else "string"
        return f"map<{keys}, {_type_label(items, graph)}>"
    return str(value["type"])


def compact_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Fold vocabulary and Sourced[T] visually, retaining the underlying field paths."""
    roots = graph.graph["roots"]
    vocabulary = {
        name
        for name, data in graph.nodes(data=True)
        if name not in roots
        and (
            "enum" in data["schema"]
            or data["schema"].get("type") in {"string", "integer", "number", "boolean"}
        )
    }
    wrappers = {
        name
        for name, data in graph.nodes(data=True)
        if name not in roots and not data.get("external") and _is_sourced(data["schema"])
    }
    folded = vocabulary | wrappers
    compact = graph.subgraph(graph.nodes - folded).copy()
    compact.remove_edges_from(list(compact.edges))
    compact.graph.update(compact=True, folded_types=folded, source_type_count=len(graph))
    for source in compact:
        annotations: set[str] = set()
        notes: set[str] = set()
        definition = graph.nodes[source]["schema"]
        for _, target, data in graph.out_edges(source, data=True):
            fields = data["fields"]
            if target in vocabulary or target == "FactSource":
                for field in fields:
                    value = definition if field == "variant" else definition["properties"][field]
                    annotations.add(f"{field}: {_type_label(value, graph)}")
                target_schema = graph.nodes[target]["schema"]
                notes.add(f"{target}: {target_schema['description']}")
                if "enum" in target_schema:
                    notes.add(f"{target} values: {_type_label(target_schema, graph)}")
                continue
            if target in wrappers:
                value_paths = _reference_paths(
                    graph.nodes[target]["schema"]["properties"]["value"], "value"
                )
                value_targets = set(value_paths)
                visible_targets = value_targets - vocabulary
                parent_paths: set[str] = set()
                for field in fields:
                    value = definition if field == "variant" else definition["properties"][field]
                    parent_paths.update(_reference_paths(value, field)[target])
                    notes.add(f"{field}: {_type_label(value, graph)}; source: FactSource")
                    if not visible_targets or value_targets & vocabulary:
                        annotations.add(f"{field}: {_type_label(value, graph)}")
                for value_target in visible_targets:
                    labels = compact.get_edge_data(source, value_target, {}).get(
                        "sourced_fields", set()
                    )
                    compact.add_edge(
                        source,
                        value_target,
                        sourced_fields=labels
                        | {
                            f"{parent}.{child}"
                            for parent in parent_paths
                            for child in value_paths[value_target]
                        },
                    )
                continue
            labels = compact.get_edge_data(source, target, {}).get("fields", set())
            compact.add_edge(source, target, fields=labels | fields)
        compact.nodes[source]["annotations"] = sorted(annotations)
        compact.nodes[source]["description"] = "\n\n".join(
            [graph.nodes[source]["description"], *sorted(annotations | notes)]
        )
    return compact


def _text_rows(text: str) -> str:
    return "".join(
        f'{escape(line)}<BR ALIGN="LEFT"/>'
        for line in wrap(text, width=48, break_long_words=False, break_on_hyphens=False)
    )


def graph_dot(graph: nx.DiGraph, title: str) -> str:
    """Group by concern and emphasize core domain objects over the role colors."""

    def quote(value: object) -> str:
        return json.dumps(value, ensure_ascii=False)

    # Rank dependencies within each group; references to other subjects must not
    # pull otherwise independent groups across the full width of the diagram.
    layout_groups = {name: data["concern"] for name, data in graph.nodes(data=True)}
    lines = [
        "digraph BackendTypes {",
        f'  graph [rankdir=LR, newrank=true, bgcolor="#f8fafc", pad=0.4, nodesep=0.25, ranksep=0.55, label={quote(title)}, labelloc=t, fontname=Helvetica, fontsize=24];',
        '  node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#cbd5e1", fontname=Helvetica, fontsize=12, margin="0.16,0.10"];',
        '  edge [color="#94a3b8", fontcolor="#475569", fontname=Helvetica, fontsize=9, arrowsize=0.65];',
    ]
    lines.append(
        '  subgraph cluster_legend { label="Type roles & emphasis"; color="#cbd5e1"; fontsize=16;'
    )
    for layer, (label, color) in LAYERS.items():
        lines.append(f'    "legend:{layer}" [label={quote(label)}, fillcolor={quote(color)}];')
    lines.append(
        f'    "legend:domain" [label="Core domain object", {CORE_DOMAIN_STYLE}, '
        'tooltip="Red border and title: central scientific objects and their UI projections. Hover over a highlighted type for its semantic role."];'
    )
    if graph.graph.get("compact"):
        legend = (
            "Solid: field reference\nDashed: field through Sourced<T>.value\n"
            "Inline: scalar IDs, enums, FactSource\nSourced<T> carries value + FactSource\n"
            "Outlined external types are not expanded"
        )
        lines.append(f'    "legend:notation" [label={quote(legend)}, fontsize=10];')
    lines.append("  }")
    for concern, (label, _) in CONCERNS.items():
        members = sorted(
            name for name, data in graph.nodes(data=True) if data["concern"] == concern
        )
        if not members:
            continue
        unit = "box" if graph.graph.get("compact") else "type"
        count = (
            f"{len(members)} {unit}{'' if len(members) == 1 else ('es' if unit == 'box' else 's')}"
        )
        lines.append(f"  subgraph cluster_{concern} {{")
        lines.append(
            f'    graph [label={quote(f"{label} · {count}")}, labeljust=l, fontsize=20, fontcolor="#0f172a", color="#94a3b8", style="rounded,filled", fillcolor="#ffffff", margin=24];'
        )
        grouped: set[str] = set()
        if concern == "scientific_model":
            for group, (group_label, modules) in SCIENTIFIC_MODEL_GROUPS.items():
                owners = {f"nof1_causal_lab.{module}" for module in modules}
                group_members = [
                    name
                    for name in members
                    if graph.nodes[name]["schema"].get("x-python-module") in owners
                ]
                if not group_members:
                    continue
                grouped.update(group_members)
                layout_groups.update(dict.fromkeys(group_members, f"{concern}/{group}"))
                lines.append(f"    subgraph cluster_{concern}_{group} {{")
                lines.append(
                    f"      graph [label={quote(group_label)}, fontsize=16, "
                    'color="#e2e8f0", fillcolor="#f8fafc", margin=16];'
                )
                lines.extend(f"      {quote(name)};" for name in group_members)
                lines.append("    }")
        lines.extend(f"    {quote(name)};" for name in members if name not in grouped)
        lines.append("  }")
    for name in sorted(graph):
        color = LAYERS[graph.nodes[name]["layer"]][1]
        tooltip = graph.nodes[name]["description"]
        node = graph.nodes[name]
        details = node["annotations"] if graph.graph.get("compact") else [node["summary"]]
        if node.get("external"):
            details = ["external type · fields not expanded", *details]
        summary = "".join(_text_rows(detail) for detail in details)
        body = (
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="#475569">{summary}</FONT></TD></TR>'
            if summary
            else ""
        )
        label = (
            '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">'
            f'<TR><TD ALIGN="LEFT"><B>{escape(name)}</B></TD></TR>'
            f"{body}</TABLE>"
        )
        style = ', style="rounded,dashed,filled"' if node.get("external") else ""
        if name in CORE_DOMAIN_OBJECTS:
            style += f", {CORE_DOMAIN_STYLE}"
            tooltip = f"Core domain object: {CORE_DOMAIN_OBJECTS[name]}\n\n{tooltip}"
        lines.append(
            f"  {quote(name)} [label=<{label}>, fillcolor={quote(color)}, tooltip={quote(tooltip)}{style}];"
        )
    for source, target, data in sorted(graph.edges(data=True)):
        for field_kind, style in (
            ("fields", ""),
            ("sourced_fields", ', style=dashed, color="#7c3aed", fontcolor="#6d28d9"'),
        ):
            if data.get(field_kind):
                label = "\n".join(
                    wrap(", ".join(sorted(data[field_kind])), width=36, break_long_words=False)
                )
                label_attribute = "label"
                if layout_groups[source] != layout_groups[target]:
                    style += ", constraint=false"
                    # Place these labels after layout, without introducing rank nodes.
                    label_attribute = "xlabel"
                lines.append(
                    f"  {quote(source)} -> {quote(target)} [{label_attribute}={quote(label)}{style}];"
                )
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    from scripts.export_schemas import export_schemas

    parser = argparse.ArgumentParser(description=__doc__)
    focus = parser.add_mutually_exclusive_group()
    focus.add_argument("--root", help="Only include this type and its dependencies")
    focus.add_argument("--view", choices=VIEWS, default="all", help="Select a concern-focused view")
    parser.add_argument(
        "--detail",
        choices=("compact", "full"),
        default="compact",
        help="Compact annotations or every exported type (default: compact)",
    )
    parser.add_argument("--output", type=Path, help="Output path stem (without extension)")
    args = parser.parse_args()
    graph = build_type_graph(export_schemas(), args.root)
    if args.root is None:
        graph = select_view(graph, args.view)
    if args.detail == "compact":
        graph = compact_graph(graph)
    name = args.root or VIEWS[args.view][1]
    if args.detail == "full":
        name += "-full"
    stem = args.output or REPO_ROOT / ".local" / "type-system" / name
    stem.parent.mkdir(parents=True, exist_ok=True)
    dot_path = stem.with_suffix(".dot")
    svg_path = stem.with_suffix(".svg")
    scope = args.root or VIEWS[args.view][0]
    title = f"{scope} · {len(graph)} boxes · {args.detail} field dependencies"
    dot_path.write_text(graph_dot(graph, title))
    subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
    print(f"{len(graph)} boxes, {graph.number_of_edges()} connections; schema references verified")
    if args.detail == "compact":
        print(
            f"{len(graph.graph['folded_types'])} of {graph.graph['source_type_count']} types folded into field annotations or sourced edges"
        )
    print(svg_path)
    print(dot_path)


if __name__ == "__main__":
    main()
