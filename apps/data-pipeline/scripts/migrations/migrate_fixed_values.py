"""Inline fixed parameter values into coefficient slots in a new format-7 study."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.migrations.study_rewrite import rewrite_study


def inline_fixed_values(payload):
    """Translate model definitions and nested authoring payloads, preserving other values."""
    if isinstance(payload, list):
        return [inline_fixed_values(item) for item in payload]
    if not isinstance(payload, dict):
        return payload

    parameters = payload.get("parameters", [])
    fixed = (
        {
            parameter["id"]: parameter["value"]
            for parameter in parameters
            if _is_parameter(parameter) and parameter.get("value") is not None
        }
        if isinstance(parameters, list)
        else {}
    )
    counts = {identity: 0 for identity in fixed}

    def inline(value):
        if isinstance(value, list):
            return [inline(item) for item in value]
        if not isinstance(value, dict):
            return value
        if value.get("kind") == "coefficient" and value.get("value") in fixed:
            identity = value["value"]
            counts[identity] += 1
            return {**value, "value": fixed[identity]}
        return {key: inline(item) for key, item in value.items()}

    if fixed:
        payload = inline(payload)
        for identity, count in counts.items():
            if count > 1:
                raise ValueError(
                    f"Fixed parameter {identity!r} references multiple slots; "
                    "inlining would untie them"
                )
        payload["parameters"] = [
            parameter for parameter in parameters if parameter.get("id") not in fixed
        ]
    return {
        key: inline_fixed_values(value)
        for key, value in payload.items()
        if not (_is_parameter(payload) and key == "value")
    }


def _is_parameter(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("id"), str)
        and value["id"].startswith("parameter:")
        and "name" in value
        and "description" in value
    )


def migrate_workspace(source: Path, destination: Path) -> dict[str, str]:
    """Reject shared fixed slots during rewriting; preserve all saved numerical data."""
    return rewrite_study(
        source,
        destination,
        inline_fixed_values,
        mapping_name="fixed-values-revisions.json",
        source_format=6,
        target_format=7,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = migrate_workspace(args.source.resolve(), args.destination.resolve())
    print(f"Migrated {len(mapping)} Git objects into {args.destination}")


if __name__ == "__main__":
    main()
