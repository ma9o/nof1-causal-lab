"""Convert retired construct execution declarations using explicit scientific revisions.

Null usage fields can be discarded. An active declaration requires a reviewed
replacement construct: Delta exactness, source recording, and initial/dynamics
laws cannot be inferred from an execution-selection flag. This offline tool is
never imported by the application and writes a separate destination.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from nof1_causal_lab.artifacts.model_spec import ModelSpec

type ConstructUsagePayload = dict[str, Any]
type ConstructRevisions = dict[str, ConstructUsagePayload]


def convert_construct_usage(
    payload: ConstructUsagePayload, *, revisions: ConstructRevisions | None = None
) -> ConstructUsagePayload:
    """Remove retired fields, requiring explicit replacements for active declarations."""
    result = deepcopy(payload)
    revisions = revisions or {}
    required: set[str] = set()
    used: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if "temporal_status" in value and "usage" in value:
                usage = value.pop("usage")
                if usage is not None:
                    identity = value["id"]
                    if identity not in revisions:
                        required.add(identity)
                    else:
                        replacement = revisions[identity]
                        if replacement.get("id") != identity or "usage" in replacement:
                            raise ValueError("Revisions must preserve construct IDs and omit usage")
                        value.clear()
                        value.update(deepcopy(replacement))
                        used.add(identity)
            for item in value.values():
                visit(item)

    visit(result)
    if required:
        raise ValueError(
            "Active usage declarations need reviewed scientific revisions: "
            f"{sorted(required)}. Specify measurement laws and source recording; "
            "retain required initial and transition laws."
        )
    if unused := revisions.keys() - used:
        raise ValueError(f"Revisions do not correspond to active declarations: {sorted(unused)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--revisions", type=Path, help="Construct ID to complete revised construct")
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve():
        parser.error("Use a separate destination to preserve the source model")
    revisions = json.loads(args.revisions.read_text()) if args.revisions else None
    converted = convert_construct_usage(json.loads(args.source.read_text()), revisions=revisions)
    from scripts.migrate_distribution_references import convert_distribution_references

    model = ModelSpec.model_validate(convert_distribution_references(converted))
    args.destination.write_text(model.model_dump_json(indent=2) + "\n")
    print("Converted model")


if __name__ == "__main__":
    main()
