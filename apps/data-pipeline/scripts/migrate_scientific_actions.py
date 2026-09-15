"""Offline copy migration: separate retained predictive checks from legacy fit logs.

Usage: uv run python scripts/migrate_scientific_actions.py SOURCE DESTINATION
Use --check to inventory changes without creating a destination. Source data is never edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from typing import TypedDict


class MigrationReport(TypedDict):
    migration: str
    source: str
    changed_files: dict[str, str]
    predictive_archives: list[str]


def migrate(source: Path, destination: Path, *, check: bool = False) -> MigrationReport:
    if destination.exists():
        raise ValueError("Migration destination must not exist")
    if not source.is_dir():
        raise ValueError("Source workspace does not exist")
    if source.resolve() in destination.resolve().parents:
        raise ValueError("Destination must be outside the source workspace")
    changed = {}
    source_hashes: dict[str, str] = {}
    archives = {}
    for path in sorted((source / "episode" / "journal").glob("*.json")):
        record = json.loads(path.read_text())
        report = record.get("diagnostics", {}).get("report")
        if not isinstance(report, dict) or "ppc" not in report:
            continue
        if record["move"].get("operation_id") != "posterior":
            raise ValueError(f"Unexpected predictive field outside a fit: {path}")
        checks = report.pop("ppc")
        if checks is not None:
            from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorPredictiveChecks

            PosteriorPredictiveChecks.model_validate(checks)
            produced_model = next(
                (item for item in record["produced"] if item["artifact_id"] == "model"), None
            )
            archives[path.name] = {
                "inference_seq": record["seq"],
                "model_version": produced_model["version"] if produced_model else None,
                "panel_version": record["diagnostics"].get("input_pins", {}).get("panel"),
                "checks": checks,
                "arrays_available": False,
                "interpretation": "Retained legacy fit measurements; no generated arrays or simulation design were recorded.",
            }
        relative = path.relative_to(source).as_posix()
        source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        changed[relative] = {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "record": record,
        }
    manifest: MigrationReport = {
        "migration": "four-scientific-actions-v1",
        "source": str(source.resolve()),
        "changed_files": source_hashes,
        "predictive_archives": sorted(archives),
    }
    if check:
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".scientific-actions-migration-", dir=destination.parent
    ) as temp:
        target = Path(temp) / "workspace"
        shutil.copytree(source, target, symlinks=True)
        for relative, item in changed.items():
            (target / relative).write_text(json.dumps(item["record"], indent=2) + "\n")
        archive = target / "episode" / "predictive-archive"
        archive.mkdir(parents=True, exist_ok=True)
        for name, value in archives.items():
            if (archive / name).exists():
                raise ValueError(f"Predictive archive collision: {name}")
            (archive / name).write_text(json.dumps(value, indent=2) + "\n")
        (target / "scientific-actions-migration.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        target.rename(destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(migrate(args.source, args.destination, check=args.check), indent=2))
