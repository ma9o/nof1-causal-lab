"""Build DEMO and workbench fixtures from the saved bundle, or project a study.

``bun run fixture:build`` regenerates every fixture from the authoritative Git
bundle and blobs; ``bun run fixture:check`` verifies them without writing.
The ``project`` subcommand supports workspace promotion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest.mock import patch

from nof1_causal_lab.actions.revisions import (
    compare_model_definitions,
    compare_model_graph,
    compare_parameters,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.execution import is_stale
from nof1_causal_lab.machine.git_objects import object_tree
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.inference import inference_is_current
from nof1_causal_lab.machine.snapshots import ModelReader
from nof1_causal_lab.machine.store import ArtifactStore, trace_log_path
from nof1_causal_lab.models.model_inputs import input_fingerprints
from nof1_causal_lab.utils import data as data_module

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_lab.machine.artifacts import EpisodeState

ROOT = Path(__file__).resolve().parents[4]
DEMO_ROOT = ROOT / "data" / "DEMO"
WORKBENCH_OUTPUT = ROOT / "apps/web/src/components/__fixtures__/workbench-comparisons.json"

ARTIFACTS: dict[ArtifactId, str] = {
    "model": "model.json",
    "identification_report": "identification_report.json",
    "validation_report": "validation_report.json",
}
TRACES = {
    "raw_data": "raw-data",
    "latent_structure": "latent-structure",
    "measurement_structure": "measurement-structure",
    "measurements": "measurement-chunk-",
    "statistical_model_spec": "model-spec-",
}


def read_fixture_files(repository: StudyRepository, state: EpisodeState) -> dict[str, bytes]:
    """Project retained payloads and logs from the selected Git ancestry."""
    records = repository.records(repository.head())
    store = ArtifactStore(repository.workspace_id)
    files = {}
    for aid, filename in ARTIFACTS.items():
        files[f"artifacts/{aid}.json"] = repository.read_file(state.current[aid].revision, filename)
    for aid, prefix in TRACES.items():
        record = next(record for record in reversed(records) if record.operation_id == aid)
        trace = next(trace for trace in sorted(record.trace_ids) if trace.startswith(prefix))
        files[f"traces/{aid}.json"] = repository.read_file(
            record.commit_id, f"logs/{trace_log_path(trace)}"
        )
    for operation, filename in [
        ("statistical_model_spec", "model_authoring.json"),
        ("posterior", "inference.json"),
    ]:
        record = next(record for record in reversed(records) if record.operation_id == operation)
        files[filename] = (json.dumps(record.diagnostics, indent=2) + "\n").encode()
        if operation == "posterior":
            # Archived summaries may survive after their original arrays are lost.
            tree = object_tree(repository.repo, record.commit_id)
            if "logs/predictive_checks.json" in tree:
                files["predictive_checks.json"] = repository.read_file(
                    record.commit_id, "logs/predictive_checks.json"
                )
    # Check external payload closure as well as the native Git objects.
    for aid, info in state.current.items():
        for filename in store.filenames(aid, info.revision):
            store.file_path(aid, info.revision, filename)
    return files


def project(source: Path, destination: Path | None = None):
    data_module._DATA_URI = str(source.parent)
    repository = StudyRepository(source.name)
    state = repository.state(repository.head())
    required: tuple[ArtifactId, ...] = (
        "raw_data",
        "model",
        "identification_report",
        "panel",
        "validation_report",
    )
    missing = [aid for aid in required if not state.has(aid)]
    if missing:
        raise ValueError(
            f"Source workspace is incomplete; missing current artifacts: {', '.join(missing)}."
        )
    stale = [aid for aid in required if is_stale(state, aid)]
    if stale:
        raise ValueError(f"Source workspace has stale current artifacts: {', '.join(stale)}.")
    if not inference_is_current(state):
        raise ValueError("Source workspace has no completed inference for its current model.")
    files = read_fixture_files(repository, state)
    if destination is not None:
        for name, content in files.items():
            path = destination / "fixture" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        subprocess.run(
            [
                "git",
                "--git-dir",
                str(destination / "episode" / "history.git"),
                "bundle",
                "create",
                str(destination / "episode" / "history.bundle"),
                "--all",
            ],
            check=True,
            capture_output=True,
        )
    return {"artifacts": list(ARTIFACTS), "traces": list(TRACES)}


def workbench_comparisons(snapshot, history):
    free = ModelSpec.model_validate(snapshot["model"]["value"])
    parameter = next(
        item
        for item in free.parameters
        if item.name == "beta_perceived_stress_burden_internalizing_symptom_burden"
    )
    fixed = parameter.model_copy(
        update={
            "value": 0,
            "distribution": None,
            "distribution_transform": "identity",
            "reference_interval_days": None,
        }
    )
    pinned = free.revised(
        parameters=tuple(fixed if item.id == parameter.id else item for item in free.parameters),
        distributions={
            key: law for key, law in free.distributions.items() if key != parameter.distribution
        },
    )
    models = {
        history[str(seq)]["context"]["state"]["current"]["model"][
            "revision"
        ]: ModelSpec.model_validate(history[str(seq)]["model"]["value"])
        for seq in (2, 3, 4, 7)
    }
    models.update(
        {format(n, "x").rjust(40, "a"): model for n, model in [(5, free), (6, free), (7, pinned)]}
    )
    comparisons = {}
    for before_version, left in models.items():
        for after_version, right in models.items():
            parameters = compare_parameters(left, right)
            before = input_fingerprints(left)
            comparisons[f"{before_version}:{after_version}"] = {
                "definition_changes": [
                    item.model_dump(mode="json") for item in compare_model_definitions(left, right)
                ],
                "parameters": [item.model_dump(mode="json") for item in parameters],
                "graph": compare_model_graph(left, right, parameters).model_dump(mode="json"),
                "changed_inputs": [
                    key for key, value in input_fingerprints(right).items() if before[key] != value
                ],
            }
    return {
        "pinned_model": pinned.model_dump(mode="json"),
        "pinned_inputs": input_fingerprints(pinned),
        "comparisons": comparisons,
    }


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
        outputs[WORKBENCH_OUTPUT] = workbench_comparisons(
            outputs[DEMO_ROOT / "fixture/model_snapshot.json"],
            outputs[DEMO_ROOT / "fixture/model_history.json"],
        )
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="Generate DEMO and workbench fixtures from the Git bundle"
    )
    build.add_argument("--check", action="store_true")
    projection = commands.add_parser(
        "project", help="Validate and project a workspace for promotion"
    )
    projection.add_argument("source", type=Path)
    projection.add_argument("destination", type=Path, nargs="?")
    args = parser.parse_args()
    if args.command == "project":
        print(json.dumps(project(args.source.resolve(), args.destination)))
        return

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
        raise SystemExit("Fixtures are stale: " + ", ".join(mismatches))
    print(f"Restored the Git bundle and composed {len(outputs)} DEMO and workbench fixtures")


if __name__ == "__main__":
    main()
