"""Named types for JSON-safe values and explicitly unchecked boundaries."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | JsonArray | JsonObject
type JsonArray = list[JsonValue]
type JsonObject = dict[str, JsonValue]

# Deliberately unchecked JSON-shaped data at parsing and library boundaries.
# CUSTOM004 makes this the visible, searchable escape hatch instead of allowing
# anonymous ``dict[str, Any]`` annotations throughout the codebase.
type UncheckedJsonObject = Annotated[
    dict[str, Any],
    Field(
        description="An unchecked JSON object carries data across a boundary before domain validation."
    ),
]
