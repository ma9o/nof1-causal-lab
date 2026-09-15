"""Shared contract primitives for persisted artifact payloads and tool schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel  # noqa: TC002

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001


def _inline_refs(schema: UncheckedJsonObject) -> UncheckedJsonObject:
    """Inline finite definitions and retain recursive references within the schema."""
    defs = schema.get("$defs", {})
    if not defs:
        return schema

    recursive = False

    def _resolve(node: Any, ancestors: frozenset[str] = frozenset()) -> Any:
        nonlocal recursive
        if isinstance(node, list):
            return [_resolve(item, ancestors) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            ref_path = node["$ref"]
            ref_name = ref_path.rsplit("/", 1)[-1]
            if ref_name in ancestors:
                recursive = True
                return node
            return _resolve(dict(defs[ref_name]), ancestors | {ref_name})
        return {key: _resolve(value, ancestors) for key, value in node.items() if key != "$defs"}

    result = _resolve(schema)
    if recursive:
        result["$defs"] = defs
    return result


@dataclass(frozen=True)
class ToolDefinition:
    """Declarative tool definition shared between pipeline and codegen."""

    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel] | None = None

    def parameters_json_schema(self) -> UncheckedJsonObject:
        schema = self.input_schema.model_json_schema()
        schema["additionalProperties"] = False
        return _inline_refs(schema)

    def result_json_schema(self) -> UncheckedJsonObject | None:
        if self.output_schema is None:
            return None
        schema = self.output_schema.model_json_schema(mode="serialization")
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
        return _inline_refs(schema)
