"""Remove legacy edge timing and recompute identification in a new format-6 study."""

from __future__ import annotations

import argparse
import json
from functools import cache
from pathlib import Path

import pygit2

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.identification import identify_model
from nof1_causal_lab.utils.arrays import read_array
from scripts.migrations.study_rewrite import remove_edge_timing, rewrite_study


def migrate_workspace(source: Path, destination: Path) -> dict[str, str]:
    """Preserve equations and numerical arrays; recertify the new temporal query."""
    repo = pygit2.Repository(str(source / "episode/history.git"))

    def read(tree_id: str, filename: str):
        tree = repo[pygit2.Oid(hex=tree_id)].peel(pygit2.Tree)
        return json.loads(tree[filename].peel(pygit2.Blob).data)

    @cache
    def identification(model_id: str):
        model = ModelSpec.model_validate(
            remove_edge_timing(read(model_id, "model.json")),
            context={
                "distribution_array_loader": lambda ref: read_array(
                    str(source / "store/arrays"), ref
                )
            },
        )
        return identify_model(model).model_dump(mode="json")

    def update_file(tree_id: str, filename: str, payload):
        if filename == "identification_report.json":
            return identification(read(tree_id, "meta.json")["derived_from"]["model"])
        return payload

    return rewrite_study(
        source,
        destination,
        remove_edge_timing,
        mapping_name="edge-timing-revisions.json",
        source_format=5,
        target_format=6,
        update_file=update_file,
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
