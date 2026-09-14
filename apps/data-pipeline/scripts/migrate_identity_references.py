"""Offline conversion of fixed-kind references to scalar IDs in retained JSON histories."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def convert_payload[T](payload: T) -> T:
    """Preserve entity IDs, model versions, journal sequences, and provenance pins."""
    value = deepcopy(payload)

    def visit(item):
        if isinstance(item, list):
            for child in item:
                visit(child)
        elif isinstance(item, dict):
            for key in ("default_outcome", "target", "outcome"):
                reference = item.get(key)
                if (
                    isinstance(reference, dict)
                    and set(reference) == {"kind", "id"}
                    and reference["kind"] == "construct"
                    and (key != "target" or "mode" in item)
                ):
                    item[key] = reference["id"]
            workspace = item.get("workspace")
            if (
                isinstance(workspace, dict)
                and set(workspace) == {"kind", "id"}
                and workspace["kind"] == "model"
            ):
                item["workspace_id"] = item.pop("workspace")["id"]
            if "issue_type" in item and "subject" in item:
                subject = item.pop("subject")
                if subject is None:
                    item["indicator_id"] = None
                elif subject["kind"] == "indicator":
                    item["indicator_id"] = subject["id"]
                elif subject["kind"] == "construct":
                    item["indicator_id"] = None
                    item["message"] += f" (construct {subject['id']})"
                else:
                    raise ValueError(f"Unsupported validation issue subject: {subject!r}")
            for child in item.values():
                visit(child)

    visit(value)
    return value


def main():
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
