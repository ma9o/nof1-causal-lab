"""Evidence-backed, advisory simplification reports for the full exported schema.

These heuristics retrieve review candidates; they do not establish semantic
equivalence. In particular, JSON Schema omits many Python validation rules.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, cast

import networkx as nx

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nof1_causal_lab.json_types import JsonObject, JsonValue
    from scripts.type_system_usage import SourceEvidence


@dataclass(frozen=True)
class Candidate:
    types: tuple[str, ...]
    summary: str
    evidence: tuple[str, ...]
    review: str


@dataclass(frozen=True)
class AnalysisReport:
    title: str
    method: str
    candidates: tuple[Candidate, ...]


def _properties(graph: nx.DiGraph, name: str) -> dict[str, JsonObject]:
    return cast("dict[str, JsonObject]", graph.nodes[name]["schema"].get("properties", {}))


def _payload_fields(graph: nx.DiGraph, name: str) -> dict[str, JsonObject]:
    return {
        field: schema for field, schema in _properties(graph, name).items() if "const" not in schema
    }


def _normalized_schema(value: JsonValue) -> JsonValue:
    """Ignore prose, while retaining defaults, references, and validation constraints."""
    if not isinstance(value, dict):
        return value
    normalized: JsonObject = {}
    for key, item in value.items():
        if key in {"title", "description", "examples", "$comment"} or key.startswith("x-"):
            continue
        if key in {"properties", "patternProperties", "$defs", "dependentSchemas"}:
            normalized[key] = {
                name: _normalized_schema(child) for name, child in cast("JsonObject", item).items()
            }
        elif key in {"anyOf", "oneOf", "allOf", "prefixItems"}:
            children = [_normalized_schema(child) for child in cast("list[JsonValue]", item)]
            normalized[key] = (
                children
                if key == "prefixItems"
                else sorted(children, key=lambda child: json.dumps(child, sort_keys=True))
            )
        elif key in {
            "items",
            "additionalProperties",
            "propertyNames",
            "contains",
            "not",
            "if",
            "then",
            "else",
        }:
            normalized[key] = _normalized_schema(item)
        elif key in {"required", "enum"}:
            normalized[key] = sorted(
                cast("list[JsonValue]", item), key=lambda child: json.dumps(child, sort_keys=True)
            )
        else:
            normalized[key] = item
    return normalized


def _field_signatures(graph: nx.DiGraph, name: str) -> dict[str, str]:
    required = graph.nodes[name]["schema"].get("required", [])
    return {
        field: json.dumps([field in required, _normalized_schema(schema)], sort_keys=True)
        for field, schema in _payload_fields(graph, name).items()
    }


def _declarations(names: tuple[str, ...], source: SourceEvidence) -> tuple[str, ...]:
    return tuple(
        f"Declared at {source.declarations[name]}" for name in names if name in source.declarations
    )


def _single_owner_wrappers(graph: nx.DiGraph, source: SourceEvidence) -> AnalysisReport:
    union_arms: set[str] = set()
    for name, data in graph.nodes(data=True):
        for _, branches in _choice_sites(data["schema"], name):
            if sum(branch.get("type") != "null" for branch in branches) > 1:
                union_arms.update(
                    str(branch["$ref"]).removeprefix("#/$defs/")
                    for branch in branches
                    if "$ref" in branch
                )
    candidates: list[Candidate] = []
    for name, data in sorted(graph.nodes(data=True)):
        fields = _payload_fields(graph, name)
        if (
            not 1 <= len(fields) <= 2
            or data["layer"] == "identity"
            or "[" in data["schema"].get("title", "")
            or name in graph.graph["artifact_roots"]
            or name in union_arms
            or graph.in_degree(name) != 1
        ):
            continue
        owner = next(graph.predecessors(name))
        owner_fields = graph.edges[owner, name]["fields"]
        if len(owner_fields) != 1 or "variant" in owner_fields or nx.has_path(graph, name, owner):
            continue
        notes = [
            f"Only exported owner: {owner}.{next(iter(owner_fields))}; payload fields: {', '.join(sorted(fields))}.",
            f"Roles: {owner}={graph.nodes[owner]['layer']}; {name}={data['layer']} ({data['concern']}).",
            *_declarations((name,), source),
        ]
        if source.methods.get(name):
            notes.append("Declared behavior: " + "; ".join(source.methods[name]))
        elif name not in source.declarations:
            notes.append("No direct Python class declaration indexed; behavior is unknown.")
        candidates.append(
            Candidate(
                (name,),
                f"{name}: review this small, singly owned record",
                tuple(notes),
                "Check independent identity, provenance, lifecycle, and inherited validators before inlining; exported ownership is not code usage.",
            )
        )
    return AnalysisReport(
        "Single-owner wrappers",
        "One exported owner field and at most two non-discriminator fields. Excludes identity types, artifact roots, generic instantiations, union arms, and recursion.",
        tuple(candidates),
    )


def _repeated_field_structures(graph: nx.DiGraph, source: SourceEvidence) -> AnalysisReport:
    signatures = {
        name: _field_signatures(graph, name)
        for name, data in sorted(graph.nodes(data=True))
        if "[" not in data["schema"].get("title", "")
    }
    ranked: list[tuple[float, Candidate]] = []
    for first, second in combinations(signatures, 2):
        left, right = signatures[first], signatures[second]
        if (graph.nodes[first]["layer"], graph.nodes[first]["concern"]) != (
            graph.nodes[second]["layer"],
            graph.nodes[second]["concern"],
        ):
            continue
        shared = {field for field in left.keys() & right.keys() if left[field] == right[field]}
        if len(shared) < 2 or not shared - {"id", "name", "description", "label", "title"}:
            continue
        overlap = len(shared) / len(left.keys() | right.keys())
        if overlap < 0.75:
            continue
        differences = sorted((left.keys() | right.keys()) - shared)
        tags = sorted(
            field
            for field in _properties(graph, first).keys() | _properties(graph, second).keys()
            if "const" in _properties(graph, first).get(field, {})
            or "const" in _properties(graph, second).get(field, {})
        )
        notes = [
            f"Exact field signatures overlap {len(shared)}/{len(left.keys() | right.keys())} ({overlap:.0%}): {', '.join(sorted(shared))}.",
            f"Same role/concern: {graph.nodes[first]['layer']}/{graph.nodes[first]['concern']}.",
            f"Other payload fields: {', '.join(differences) or 'none'}; discriminator fields to compare: {', '.join(tags) or 'none'}.",
            *_declarations((first, second), source),
        ]
        ranked.append(
            (
                overlap,
                Candidate(
                    (first, second),
                    f"{first} + {second}: review shared field structure",
                    tuple(notes),
                    "Compare scientific meanings, discriminator values, and Python validators before extracting a shared concept.",
                ),
            )
        )
    return AnalysisReport(
        "Repeated field structures",
        "At least two matching non-discriminator fields and 75% Jaccard overlap within the same role/concern. Signatures retain requiredness, defaults, cardinality, nominal references, and schema constraints; prose is ignored.",
        tuple(
            candidate for _, candidate in sorted(ranked, key=lambda item: (-item[0], item[1].types))
        ),
    )


def _field_usage_clusters(graph: nx.DiGraph, source: SourceEvidence) -> AnalysisReport:
    candidates: list[Candidate] = []
    for name in sorted(graph):
        fields = set(_payload_fields(graph, name))
        uses = [
            use for use in source.uses.get(name, ()) if not use.internal and use.fields & fields
        ]
        co_use = nx.Graph()
        for use in uses:
            observed = use.fields & fields
            co_use.add_nodes_from(observed)
            co_use.add_edges_from(combinations(sorted(observed), 2))
        groups = sorted((sorted(group) for group in nx.connected_components(co_use)), key=tuple)
        supported = [
            group
            for group in groups
            if len(group) >= 2 and sum(bool(use.fields & set(group)) for use in uses) >= 2
        ]
        if len(supported) < 2:
            continue
        notes = [
            f"Observed {len(co_use)}/{len(fields)} payload fields in {len(uses)} external functions; no observed function joins these groups."
        ]
        for group in supported:
            consumers = [use.location for use in uses if use.fields & set(group)]
            notes.append(f"{{{', '.join(group)}}}: " + "; ".join(consumers))
        unobserved = fields - set(co_use)
        if unobserved:
            notes.append(
                "Fields without attributed accesses (not unused): " + ", ".join(sorted(unobserved))
            )
        notes.extend(_declarations((name,), source))
        candidates.append(
            Candidate(
                (name,),
                f"{name}: review {len(supported)} separate field-use groups",
                tuple(notes),
                "Check whole-object consumers and invariants before splitting. Class methods are excluded from clustering; incomplete static coverage can create apparent separation.",
            )
        )
    return AnalysisReport(
        "Field-usage clusters",
        "Connected components of fields co-accessed by external Python functions. Requires two groups with at least two fields and two consumers each; only attributed accesses count.",
        tuple(candidates),
    )


def _choice_sites(value: JsonObject, path: str) -> Iterator[tuple[str, list[JsonObject]]]:
    for keyword in ("oneOf", "anyOf"):
        if keyword in value:
            branches = cast("list[JsonObject]", value[keyword])
            yield path, branches
            for index, branch in enumerate(branches):
                yield from _choice_sites(branch, f"{path}.{keyword}[{index}]")
    for field, child in cast("dict[str, JsonObject]", value.get("properties", {})).items():
        yield from _choice_sites(child, f"{path}.{field}")
    for keyword, suffix in (
        ("items", "[]"),
        ("additionalProperties", "{value}"),
        ("propertyNames", "{key}"),
    ):
        child = value.get(keyword)
        if isinstance(child, dict):
            yield from _choice_sites(child, path + suffix)
    for index, child in enumerate(cast("list[JsonObject]", value.get("prefixItems", []))):
        yield from _choice_sites(child, f"{path}[{index}]")


def _alternative_representations(graph: nx.DiGraph, source: SourceEvidence) -> AnalysisReport:
    candidates: list[Candidate] = []
    for name in sorted(graph):
        for field, schema in _properties(graph, name).items():
            for path, branches in _choice_sites(schema, field):
                forms: dict[str, list[str]] = defaultdict(list)
                for branch in branches:
                    if "$ref" in branch:
                        target = str(branch["$ref"]).removeprefix("#/$defs/")
                        definition = graph.nodes[target]["schema"]
                        if graph.nodes[target]["layer"] == "identity":
                            forms["reference"].append(target)
                        elif definition.get("type") == "object":
                            forms["object"].append(target)
                    elif (
                        branch.get("type") in {"number", "integer", "string", "boolean"}
                        and "const" not in branch
                        and "enum" not in branch
                    ):
                        forms["literal"].append(str(branch["type"]))
                if not forms["reference"] or not (forms["object"] or forms["literal"]):
                    continue
                notes = [
                    "Alternatives in one field: "
                    + "; ".join(
                        f"{kind}={', '.join(values)}"
                        for kind, values in sorted(forms.items())
                        if values
                    )
                    + "."
                ]
                behavior = [
                    use for use in source.uses.get(name, ()) if field in use.fields and use.behavior
                ]
                notes.extend(f"Validation/conversion evidence: {use.location}" for use in behavior)
                if not behavior:
                    notes.append(
                        "No field-specific validation/conversion function was attributed by this source scan."
                    )
                notes.extend(_declarations((name,), source))
                candidates.append(
                    Candidate(
                        (name,),
                        f"{name}.{path}: review literal/inline versus reference forms",
                        tuple(notes),
                        "Confirm these alternatives denote the same concept. Check whether one canonical internal representation could simplify branches while preserving authoring and identity semantics.",
                    )
                )
    return AnalysisReport(
        "Alternative representations",
        "Union sites accepting an identity/reference type alongside an inline object or primitive literal. Nullability alone and unrelated object-only unions are excluded. This is a canonicalization hypothesis, not equivalence proof.",
        tuple(candidates),
    )


def analyze_type_graph(
    graph: nx.DiGraph, source: SourceEvidence, selected: set[str]
) -> tuple[AnalysisReport, ...]:
    """Measure the full graph, then select findings touching the requested view."""
    reports = (
        _single_owner_wrappers(graph, source),
        _repeated_field_structures(graph, source),
        _field_usage_clusters(graph, source),
        _alternative_representations(graph, source),
    )
    return tuple(
        AnalysisReport(
            report.title,
            report.method,
            tuple(
                candidate
                for candidate in report.candidates
                if selected.intersection(candidate.types)
            ),
        )
        for report in reports
    )


def render_analysis(
    reports: tuple[AnalysisReport, ...],
    *,
    source: SourceEvidence,
    type_count: int,
    selected_count: int,
    scope: str,
    limit: int = 0,
) -> str:
    """Render a console-readable Markdown report; zero limit includes every finding."""
    lines = [
        f"# Type-system analysis: {scope}",
        "",
        f"Advisory review candidates; no semantic equivalence or automatic refactoring is implied. Analyzed {type_count} full-schema types; {selected_count} selected types; {source.files} Python source files.",
        "",
        "Source coverage: direct accesses on annotated parameters and instance-method self, plus field-validator/serializer declarations. Aliases, container elements, dynamic accesses, inherited behavior, and non-Python consumers are not resolved. Missing evidence never means unused.",
        "",
    ]
    for report in reports:
        lines.extend(
            [f"## {report.title} ({len(report.candidates)} candidates)", "", report.method, ""]
        )
        shown = report.candidates[:limit] if limit else report.candidates
        if not shown:
            lines.extend(["No candidates met this heuristic in the selected scope.", ""])
        for candidate in shown:
            lines.append(f"- **{candidate.summary}**")
            lines.extend(f"  - {item}" for item in candidate.evidence)
            lines.extend([f"  - Review: {candidate.review}", ""])
        if len(shown) < len(report.candidates):
            lines.extend(
                [
                    f"Showing {len(shown)} of {len(report.candidates)}; the saved report contains every candidate. Use --limit 0 to print all.",
                    "",
                ]
            )
    return "\n".join(lines)
