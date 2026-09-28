"""Project a complete local Git study through the production state and history readers."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from nof1_causal_lab.machine.execution import is_stale
from nof1_causal_lab.machine.git_objects import object_tree
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.inference import inference_is_current
from nof1_causal_lab.machine.store import ArtifactStore, trace_log_path
from nof1_causal_lab.utils import data as data_module

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_lab.machine.artifacts import EpisodeState

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
        record = next(
            record
            for record in reversed(records)
            if record.operation_id == aid
        )
        trace = next(trace for trace in sorted(record.trace_ids) if trace.startswith(prefix))
        files[f"traces/{aid}.json"] = repository.read_file(
            record.commit_id, f"logs/{trace_log_path(trace)}"
        )
    for operation, filename in [
        ("statistical_model_spec", "model_authoring.json"),
        ("posterior", "inference.json"),
    ]:
        record = next(
            record
            for record in reversed(records)
            if record.operation_id == operation
        )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    args = parser.parse_args()
    print(json.dumps(project(args.source.resolve(), args.destination)))
