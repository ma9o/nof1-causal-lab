"""Project DEMO fixtures from the tracked Git bundle and numerical blobs.

The bundle preserves the illustrative study's model, action logs and traces.
Its inference report is retained without the original joint samples; this
command never invents a conditioned ModelSpec or runs fitting or simulation.
Prior plot viewports use the production reader's deterministic native draw.

Run ``bun run fixture:demo`` or ``bun run fixture:demo:check`` from the repo root.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.snapshots import ModelReader
from nof1_causal_lab.utils import data as data_module
from scripts.project_study_fixture import read_fixture_files

DEMO_ROOT = Path(__file__).resolve().parents[3] / "data" / "DEMO"


def build_outputs():
    # A clean restore proves the fixture does not depend on a developer's local
    # repository, generated JSON projections or retired numbered directories.
    with (
        TemporaryDirectory(prefix="nof1-model-fixture-") as directory,
        patch.object(data_module, "_DATA_URI", directory),
    ):
        workspace = Path(directory) / "DEMO"
        history = workspace / "episode/history.git"
        history.parent.mkdir(parents=True)
        subprocess.run(
            ["git", "clone", "--mirror", str(DEMO_ROOT / "episode/history.bundle"), str(history)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "--git-dir", str(history), "config", "nof1.format", "4"],
            check=True,
            capture_output=True,
        )
        shutil.copytree(DEMO_ROOT / "store", workspace / "store")
        reader = ModelReader("DEMO")
        repository = StudyRepository("DEMO")
        commits = {0: reader.records[0].parent_ids[0]}
        commits.update({record.seq: record.commit_id for record in reader.records})
        outputs = {
            DEMO_ROOT / "fixture" / name: json.loads(content)
            for name, content in read_fixture_files(repository, reader.state).items()
        }
        outputs.update(
            {
                DEMO_ROOT / "fixture/model_snapshot.json": reader.snapshot().model_dump(
                    mode="json"
                ),
                DEMO_ROOT / "fixture/model_history.json": {
                    str(seq): ModelReader("DEMO", at=commit).snapshot().model_dump(mode="json")
                    for seq, commit in commits.items()
                },
            }
        )
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()
    mismatches = []
    for path, value in outputs.items():
        rendered = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.check:
            if path.read_text() != rendered:
                mismatches.append(str(path))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered)
    if mismatches:
        raise SystemExit("Read fixtures are stale: " + ", ".join(mismatches))
    print(
        f"Restored the Git bundle and composed {len(outputs)} DEMO projections; no numerical runs"
    )


if __name__ == "__main__":
    main()
