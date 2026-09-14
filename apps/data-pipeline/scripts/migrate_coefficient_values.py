"""Offline conversion of nested coefficient variants to direct operand values."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, overload

from pydantic import FiniteFloat, TypeAdapter

from nof1_causal_lab.artifacts.identity import ParameterId

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject


@overload
def convert_value(value: JsonObject) -> float | ParameterId: ...


@overload
def convert_value(value: None) -> None: ...


def convert_value(value: JsonObject | None) -> float | ParameterId | None:
    """Read an explicitly retired fixed or parameter coefficient record."""
    if value is None:
        return None
    match value:
        case {"kind": "fixed", "value": literal} if len(value) == 2:
            return TypeAdapter(FiniteFloat).validate_python(literal)
        case {"kind": "parameter", "parameter_id": identity} if len(value) == 2:
            return TypeAdapter(ParameterId).validate_python(identity)
        case _:
            raise ValueError(f"Invalid retired coefficient: {value!r}")


def convert_payload[T](payload: T) -> T:
    """Flatten expression operands in retained models, snapshots, and journal payloads."""
    value = deepcopy(payload)

    def visit(item):
        if isinstance(item, dict):
            if item.get("kind") == "coefficient" and "coefficient" in item:
                if "value" in item:
                    raise ValueError("An operand cannot contain both retired and current values")
                item["value"] = convert_value(item.pop("coefficient"))
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--write", action="store_true", help="Replace the supplied JSON files")
    args = parser.parse_args()
    changed = 0
    for path in args.paths:
        original = json.loads(path.read_text())
        converted = convert_payload(original)
        if converted == original:
            continue
        changed += 1
        if args.write:
            path.write_text(
                json.dumps(converted, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            )
    print(f"{'Converted' if args.write else 'Would convert'} {changed} file(s)")


if __name__ == "__main__":
    main()
