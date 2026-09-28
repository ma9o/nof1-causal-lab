"""Offline conversion of numbered artifacts and journals to native Git objects.

Run with the study offline. Original artifacts, journals and traces remain untouched;
an existing first-generation Git repository is retained as history.before-objects.git.
Only a fully built replacement repository is published. Runtime readers never migrate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pyarrow.parquet as pq
import pygit2

from nof1_causal_lab.machine.git_objects import read_file
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord, trace_log_path
from nof1_causal_lab.utils import data as data_module
from scripts.migrate_action_history import convert_transition


class ArtifactConversion:
    """Resolve the legacy dependency DAG once, preserving every retained payload."""

    def __init__(self, workspace: Path, store: ArtifactStore):
        self.workspace = workspace
        self.store = store
        self.converted = {}

    def artifact(self, artifact_id, version):
        key = (artifact_id, version)
        if key not in self.converted:
            directory = self.workspace / "store" / artifact_id / f"v{version}"
            meta = json.loads((directory / "meta.json").read_bytes())
            pins = {
                aid: self.artifact(aid, number).revision
                for aid, number in meta["derived_from"].items()
            }
            self.converted[key] = self.store.write_artifact(
                artifact_id,
                derived_from=pins,
                produced_by=meta["produced_by"],
                created_at=meta["created_at"],
                json_files={
                    path.name: self.value(json.loads(path.read_bytes()))
                    for path in directory.glob("*.json")
                    if path.name != "meta.json"
                },
                parquet_files={
                    path.name: pq.read_table(path) for path in directory.glob("*.parquet")
                },
            )
        return self.converted[key]

    def value(self, value):
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {"workspace_id", "version"}:
            return {
                "workspace_id": value["workspace_id"],
                "revision": self.artifact("model", value["version"]).revision,
                "path": "model.json",
            }
        result = {}
        for key, item in value.items():
            if key in {"input_pins", "input_versions", "derived_from"}:
                result["input_revisions" if key == "input_versions" else key] = {
                    aid: self.artifact(aid, number).revision for aid, number in item.items()
                }
            elif key in {
                "model_version",
                "panel_version",
                "raw_data_version",
                "comparison_panel_version",
                "expected_model_version",
                "expected_version",
            }:
                aid = "panel" if "panel" in key else "raw_data" if "raw_data" in key else "model"
                result[key.replace("version", "revision")] = (
                    self.artifact(aid, item).revision if item else None
                )
            else:
                result[key] = self.value(item)
        return result

    def record(self, raw):
        converted = self.value({key: value for key, value in raw.items() if key != "produced"})
        converted["produced"] = [
            self.artifact(info["artifact_id"], info["version"]) for info in raw["produced"]
        ]
        if "move" in converted:
            converted = convert_transition(converted)
        return TransitionRecord.model_validate(converted)


def migrate(workspace: Path) -> int:
    workspace = workspace.resolve()
    destination = workspace / "episode/history.git"
    backup = destination.with_name("history.before-objects.git")
    old = pygit2.Repository(str(destination)) if destination.exists() else None
    if old is not None and "nof1.format" in old.config and old.config.get_int("nof1.format") >= 2:
        raise FileExistsError(f"Study already uses native Git objects: {destination}")
    if backup.exists():
        raise FileExistsError(f"Migration backup already exists: {backup}")
    entries = []
    old_heads = {}
    if old is not None:
        old_heads = {name: str(old.branches[name].target) for name in old.branches.local}
        for ref in sorted(
            (ref for ref in old.references if ref.startswith("refs/attempts/")),
            key=lambda ref: int(ref.rsplit("/", 1)[1]),
        ):
            commit = old[old.references[ref].target].peel(pygit2.Commit)
            entries.append(
                (
                    str(commit.id),
                    str(commit.parent_ids[0]),
                    json.loads(read_file(old, str(commit.id), "logs/transition.json")),
                    commit,
                )
            )
    else:
        journal = workspace / "episode/journal"
        if not journal.is_dir():
            raise ValueError(f"No journal to migrate: {journal}")
        entries = [
            (None, None, json.loads(path.read_bytes()), None)
            for path in sorted(journal.glob("*.json"))
        ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        TemporaryDirectory(prefix=".history-", dir=destination.parent) as directory,
        patch.object(data_module, "_DATA_URI", str(workspace.parent)),
    ):
        path = Path(directory) / "history.git"
        repository = StudyRepository(workspace.name, repository_path=path)
        store = ArtifactStore(workspace.name, repository_path=path)
        conversion = ArtifactConversion(workspace, store)
        commits = {}
        if old is not None:
            for oid in old_heads.values():
                for commit in old.walk(pygit2.Oid(hex=oid)):
                    if not commit.parent_ids:
                        commits[str(commit.id)] = repository.head()
        for old_id, old_parent, raw, commit in entries:
            record = conversion.record(raw)
            parent = commits[old_parent] if old_parent is not None else repository.head()
            if record.branch not in repository.branches():
                repository.create_branch(record.branch, at=parent)
            logs = {}
            if commit is not None:
                assert old is not None
                logs["events.json"] = read_file(old, str(commit.id), "logs/events.json")
            for trace in record.trace_ids:
                if commit is not None:
                    assert old is not None
                    logs[trace_log_path(trace)] = read_file(
                        old, str(commit.id), f"logs/{trace_log_path(trace)}"
                    )
                else:
                    logs[trace_log_path(trace)] = (
                        workspace / "episode/traces" / f"{record.seq:06d}" / f"{trace}.json"
                    ).read_bytes()
            new_id = repository.append(record, expected_head=parent, logs=logs)
            if old_id is not None:
                commits[old_id] = new_id
        for name, oid in old_heads.items():
            if name not in repository.branches():
                repository.create_branch(name, at=commits[oid])
            if repository.head(name) != commits[oid]:
                raise ValueError(f"Converted branch head differs: {name}")
        # Preserve unpublished artifacts too; they are not selected by any commit.
        for meta in sorted((workspace / "store").glob("*/v*/meta.json")):
            info = json.loads(meta.read_bytes())
            conversion.artifact(info["artifact_id"], info["version"])
        for head in repository.branches().values():
            repository.state(head)
        store.repo.free()
        repository.repo.free()
        if old is not None:
            old.free()
            destination.rename(backup)
        path.rename(destination)
    return len(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    print(f"Migrated {migrate(args.workspace)} attempts; original files preserved")
