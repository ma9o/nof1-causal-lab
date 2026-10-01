"""Export API contracts, tool schemas, metadata, OpenAPI, and the curl skill.

Run ``bun run codegen`` or ``bun run codegen:check`` from the repository root.
All Python exports are generated together before their TypeScript consumers.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.actions.data_diff import DataDiffReport, DataDiffRequest
from nof1_causal_lab.actions.results import ActionPoll, ActionReceipt

# Import all artifact contracts — this pulls in every nested domain model
from nof1_causal_lab.actions.revisions import ModelDiffReport, RevisionCatalog
from nof1_causal_lab.actions.status import StudyStatus
from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS
from nof1_causal_lab.artifacts.effects import EffectSummary
from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.scenarios import (
    CausalEffectResult,
    EffectTrajectoryPoint,
)
from nof1_causal_lab.distributions import OBSERVATION_FAMILY_SPECS
from nof1_causal_lab.study.snapshot_models import ModelSnapshot
from nof1_causal_lab.study.view_models import ArtifactViewResponse
from nof1_causal_lab.study.visual_models import (
    MechanismCurves,
    MechanismViewRequest,
    ObservationHistory,
    ParameterDraws,
    PredictiveHistory,
    SimulationPaths,
)
from nof1_causal_lab.study_api import (
    ArtifactEnvelope,
    AttemptTraceIndex,
    CapabilitiesResponse,
    EventsResponse,
    TimelineResponse,
    UploadResponse,
    WorkspaceEntry,
    WorkspaceList,
)
from nof1_causal_lab.utils.llm import LLMTrace
from scripts.codegen.type_system_catalog import annotate_definitions

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic.json_schema import JsonSchemaValue

REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = REPO_ROOT / "packages" / "api-types" / "schemas"
SKILL_PATH = REPO_ROOT / ".agents" / "skills" / "nof1-study-api" / "SKILL.md"

EXPORTED_API_MODELS: tuple[type[BaseModel], ...] = (
    CapabilitiesResponse,
    WorkspaceEntry,
    WorkspaceList,
    UploadResponse,
    ArtifactEnvelope,
    TimelineResponse,
    AttemptTraceIndex,
    EventsResponse,
    RevisionCatalog,
    ModelDiffReport,
    DataDiffReport,
    DataDiffRequest,
    LLMTrace,
    ModelSnapshot,
    ArtifactViewResponse,
    MechanismCurves,
    MechanismViewRequest,
    ObservationHistory,
    ParameterDraws,
    PredictiveHistory,
    SimulationPaths,
    StudyStatus,
    ActionReceipt,
    ActionPoll,
)

EXPORTED_TOOL_MODELS: tuple[type[BaseModel], ...] = (
    EffectSummary,
    EffectTrajectoryPoint,
    CausalEffectResult,
)


def _make_defaults_required(schema: JsonSchemaValue) -> JsonSchemaValue:
    """Make all properties with defaults required in serialization schema.

    Pydantic marks fields with defaults as optional in JSON Schema, but in
    serialization mode they're always present. This post-processes the schema
    to make them required so TypeScript types aren't overly permissive.

    Only applies to object schemas that have 'properties'.
    Does NOT touch fields where the default is None and the type includes null
    (those are genuinely optional/nullable).
    """
    # Recurse into $defs
    if "$defs" in schema:
        for name, defn in schema["$defs"].items():
            schema["$defs"][name] = _make_defaults_required(defn)

    # Recurse into properties
    if "properties" in schema:
        props = schema["properties"]
        current_required = set(schema.get("required", []))

        for prop_name, prop_schema in props.items():
            # Skip already-required fields
            if prop_name in current_required:
                continue

            # Skip fields that are nullable (anyOf with null) — these are
            # genuinely optional fields that default to None
            if _is_nullable(prop_schema):
                continue

            # This field has a default but is not nullable — make it required
            current_required.add(prop_name)

        if current_required:
            schema["required"] = sorted(current_required)

        # Recurse into nested properties
        for prop_schema in props.values():
            _make_defaults_required(prop_schema)

    # Recurse into items (arrays)
    if "items" in schema:
        _make_defaults_required(schema["items"])

    # Recurse into anyOf/oneOf
    for key in ("anyOf", "oneOf"):
        if key in schema:
            for i, item in enumerate(schema[key]):
                schema[key][i] = _make_defaults_required(item)

    return schema


def _is_nullable(prop_schema: JsonSchemaValue) -> bool:
    """Check if a property schema allows null (e.g., anyOf with null type)."""
    # Direct null type
    if prop_schema.get("type") == "null":
        return True

    # Default is None
    if prop_schema.get("default") is None and "default" in prop_schema:
        return True

    # anyOf contains null
    any_of: list[dict[str, object]] = prop_schema.get("anyOf", [])
    return any(item.get("type") == "null" for item in any_of)


def _collect_model_schema(model_cls: type[BaseModel], all_defs: JsonSchemaValue) -> dict[str, str]:
    schema = model_cls.model_json_schema(mode="serialization")
    defs = schema.pop("$defs", {})
    all_defs.update(defs)
    model_name = model_cls.__name__
    all_defs[model_name] = {k: v for k, v in schema.items() if k not in ("$defs",)}
    return {"$ref": f"#/$defs/{model_name}"}


def export_schemas() -> JsonSchemaValue:
    """Build a combined JSON Schema with exported Python models in $defs."""
    all_defs: JsonSchemaValue = {}
    artifact_refs: dict[str, dict[str, str]] = {}

    for artifact_id, model_cls in ARTIFACT_CONTRACTS.items():
        artifact_refs[artifact_id] = _collect_model_schema(model_cls, all_defs)

    for model_cls in (*EXPORTED_API_MODELS, *EXPORTED_TOOL_MODELS):
        _collect_model_schema(model_cls, all_defs)

    annotate_definitions(all_defs)
    combined = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CausalSSMContracts",
        "description": "Combined JSON Schema for exported artifact contracts and facade API models. Generated from Python Pydantic models.",
        "type": "object",
        "properties": artifact_refs,
        "$defs": dict(sorted(all_defs.items())),
    }

    # Post-process: make non-nullable defaults required
    return _make_defaults_required(combined)


def export_metadata() -> JsonSchemaValue:
    """Export distribution catalog metadata for TypeScript type-safe rendering maps."""
    site_kind_values = {sk.value for sk in SiteKind if sk.name.startswith("OBS_")}
    catalog_hypers = {h for spec in OBSERVATION_FAMILY_SPECS for h in spec.hyperparameters}
    if catalog_hypers != site_kind_values:
        diff = catalog_hypers.symmetric_difference(site_kind_values)
        raise ValueError(
            f"ObservationFamilyCatalogEntry.hyperparameters out of sync with SiteKind: {diff}"
        )
    return {
        "artifactIds": list(ARTIFACT_IDS),
        "observationHyperparametersByDistribution": {
            spec.family.value: list(spec.hyperparameters)
            for spec in OBSERVATION_FAMILY_SPECS
            if spec.hyperparameters
        },
    }


_BASE_URL = "${TOOL_SERVER_URL:-http://localhost:8100}"
_MAX_EXAMPLE_DEPTH = 8


def _example_from_schema(
    schema: JsonSchemaValue,
    components: JsonSchemaValue,
    *,
    depth: int = 0,
    seen: frozenset[str] = frozenset(),
) -> Any:
    """A minimal JSON skeleton satisfying a schema, for a copy-pasteable body.

    Required object properties only, first branch of a union, first enum value.
    The authoritative, hand-written body examples live in the app description;
    this is just a shape hint next to each endpoint.
    """
    if depth > _MAX_EXAMPLE_DEPTH:
        return {}

    ref = schema.get("$ref")
    if ref:
        name = ref.rsplit("/", 1)[-1]
        if name in seen:
            return {}
        return _example_from_schema(
            components.get(name, {}), components, depth=depth, seen=seen | {name}
        )

    for combinator in ("oneOf", "anyOf", "allOf"):
        branches = schema.get(combinator)
        if branches:
            non_null = [b for b in branches if b.get("type") != "null"] or branches
            return _example_from_schema(non_null[0], components, depth=depth + 1, seen=seen)

    if "const" in schema:
        return schema["const"]
    enum = schema.get("enum")
    if enum:
        return enum[0]

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        props: JsonSchemaValue = schema.get("properties", {})
        required = set(schema.get("required", []))
        # Include required fields plus any discriminator/fixed-value field (a
        # `const`, e.g. move `kind`) even when a default makes it non-required,
        # so the skeleton is a valid body. Property order is preserved.
        return {
            name: _example_from_schema(prop, components, depth=depth + 1, seen=seen)
            for name, prop in props.items()
            if name in required or "const" in prop
        }
    if schema_type == "array":
        items = schema.get("items")
        return (
            [_example_from_schema(items, components, depth=depth + 1, seen=seen)] if items else []
        )
    if schema_type == "integer" or schema_type == "number":
        return 0
    if schema_type == "boolean":
        return False
    if schema_type == "null":
        return None
    return "string"


def _path_with_placeholders(path: str, parameters: list[JsonSchemaValue]) -> str:
    """Substitute path params with uppercase placeholders for the curl example."""
    result = path
    for param in parameters:
        if param.get("in") == "path":
            name = param["name"]
            result = result.replace("{" + name + "}", name.upper())
    return result


def _curl_block(
    method: str,
    path: str,
    operation: JsonSchemaValue,
    components: JsonSchemaValue,
) -> str:
    parameters = operation.get("parameters", [])
    url = f"{_BASE_URL}{_path_with_placeholders(path, parameters)}"
    lines = [f'curl -s "{url}"']
    if method != "get":
        lines[0] += " \\"
        lines.append(f"  -X {method.upper()} \\")
        lines.append("  -H 'Content-Type: application/json' \\")
        body_schema = (
            operation.get("requestBody", {})
            .get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        example = _example_from_schema(body_schema, components) if body_schema else {}
        lines.append(f"  -d '{json.dumps(example)}'")
    return "```bash\n" + "\n".join(lines) + "\n```"


def _parameters_block(operation: JsonSchemaValue) -> str | None:
    parameters = operation.get("parameters", [])
    if not parameters:
        return None
    rows = ["**Parameters**", ""]
    for param in parameters:
        location = param.get("in", "query")
        required = "required" if param.get("required") else "optional"
        description = param.get("description", "").strip().replace("\n", " ")
        suffix = f" — {description}" if description else ""
        rows.append(f"- `{param['name']}` ({location}, {required}){suffix}")
    return "\n".join(rows)


def _skill_frontmatter() -> str:
    """YAML frontmatter for the Agent Skill.

    `name` and `description` satisfy both the Codex requirement and Claude Code's
    trigger heuristic; the description says what the skill does and when to reach
    for it. Kept as a controlled single-line double-quoted scalar for maximal
    cross-parser compatibility.
    """
    description = (
        "Drive or inspect a nof1-causal-lab study over HTTP with "
        "curl: edit models, prepare data, fit and simulate; inspect revisions, "
        "read study state/timeline/artifacts, and invoke "
        "scientific tools with dispatch and polling against the tool server. Use when working on a study "
        "as an external agent instead of the web viewer."
    )
    return f'---\nname: nof1-study-api\ndescription: "{description}"\n---'


def render_skill(openapi: JsonSchemaValue) -> str:
    info = openapi.get("info", {})
    components = openapi.get("components", {}).get("schemas", {})
    title = info.get("title", "Agent API")

    blocks: list[str] = [
        _skill_frontmatter(),
        f"# {title} — curl skill",
        (
            "> Auto-generated from `packages/api-types/schemas/openapi.json` (the FastAPI "
            "OpenAPI spec) by `apps/data-pipeline/scripts/codegen/export_api.py`. "
            "Edit the route docstrings, not this file."
        ),
        inspect.cleandoc(info.get("description", "")),
        "## Endpoints",
    ]

    for path in sorted(openapi.get("paths", {})):
        methods = openapi["paths"][path]
        for method in sorted(methods):
            operation = methods[method]
            if not isinstance(operation, dict):
                continue
            blocks.append(f"### {method.upper()} `{path}`")
            description = inspect.cleandoc(
                operation.get("description") or operation.get("summary") or ""
            )
            if description:
                blocks.append(description)
            params = _parameters_block(operation)
            if params:
                blocks.append(params)
            blocks.append(_curl_block(method, path, operation, components))

    text = "\n\n".join(block for block in blocks if block)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.rstrip("\n") + "\n"


def main(*, check: bool = False) -> bool:
    from nof1_causal_lab.tool_server import app

    openapi = app.openapi()
    outputs = {
        OUTPUT_DIR / "contracts.json": json.dumps(export_schemas(), indent=2) + "\n",
        OUTPUT_DIR / "metadata.json": json.dumps(export_metadata(), indent=2) + "\n",
        OUTPUT_DIR / "openapi.json": json.dumps(openapi, indent=2) + "\n",
        SKILL_PATH: render_skill(openapi),
    }
    changed = []
    for path, rendered in outputs.items():
        if check:
            if not path.exists() or path.read_text() != rendered:
                changed.append(path.relative_to(REPO_ROOT))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
    if changed:
        print("API exports are out of date. Run `bun run codegen`.", file=sys.stderr)
        for path in changed:
            print(f"  {path}", file=sys.stderr)
        return True
    print(f"API exports {'checked' if check else 'written'}: {len(outputs)} files")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify generated files without writing."
    )
    raise SystemExit(main(check=parser.parse_args().check))
