"""Retire operation IDs, stored progress events and episode naming in a new format-8 study copy.

Records lose `operation_id` and `resume`, artifact producers name the action that made
them, each commit's `logs/transition.json` becomes `logs/attempt.json`, the stored
`logs/events.json` is dropped, and the repository moves from `episode/` to `study/`.
Unknown or absent producers are kept as they are.

Usage: uv run python -m scripts.migrations.migrate_format_8 SOURCE DESTINATION
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject

PRODUCERS = {
    "run:posterior": "fit",
    "run:raw_data": "prepare_data",
    "run:measurements": "prepare_data",
    "run:simulated_measurements": "prepare_data",
    "run:latent_structure": "edit_model",
    "run:measurement_structure": "edit_model",
    "run:statistical_model_spec": "edit_model",
}
RENAMED_ENTRIES: dict[str, str | None] = {"transition.json": "attempt.json", "events.json": None}


def _producer(metadata: JsonObject) -> JsonObject:
    produced_by = metadata.get("produced_by")
    if not isinstance(produced_by, str) or produced_by not in PRODUCERS:
        return metadata
    return {**metadata, "produced_by": PRODUCERS[produced_by]}


def update_file(_tree: str, name: str, payload: Any) -> Any:
    if name == "transition.json":
        record = {
            key: value for key, value in payload.items() if key not in {"operation_id", "resume"}
        }
        record["produced"] = [_producer(item) for item in record.get("produced", [])]
        return record
    if name == "meta.json":
        return _producer(payload)
    return payload


def migrate(source: Path, destination: Path) -> dict[str, str]:
    return rewrite_study(
        source,
        destination,
        lambda value: value,
        mapping_name="format-8-revisions.json",
        source_format=7,
        target_format=8,
        update_file=update_file,
        rename_entry=lambda name: RENAMED_ENTRIES.get(name, name),
        layout=("episode", "study"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = migrate(args.source, args.destination)
    print(f"Rewrote {len(mapping)} objects into {args.destination}")


if __name__ == "__main__":
    main()
