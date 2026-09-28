"""Offline migration from run/write histories to scientific action records.

Writes a separate repository. The source remains untouched; stop study execution
before replacing its repository with the validated destination.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pygit2

from nof1_causal_lab.machine.git_objects import write_tree


def convert_transition(record):
    """Translate the retired command and retain only inputs that were recorded."""
    command = record.pop("move")
    operation = command.get("operation_id")
    if command["kind"] == "write":
        if command["artifact_id"] != "model":
            raise ValueError(f"Unsupported historical write: {command['artifact_id']}")
        action = "edit_model"
    else:
        action = {
            "latent_structure": "edit_model",
            "measurement_structure": "edit_model",
            "statistical_model_spec": "edit_model",
            "raw_data": "prepare_data",
            "measurements": "prepare_data",
            "simulated_measurements": "prepare_data",
            "posterior": "fit",
            "simulate": "simulate",
        }[operation]
    pins = record["diagnostics"].get("input_pins", {}) | command.get("input_revisions", {})
    inputs = {key + "_revision": value for key, value in pins.items()}
    if action == "edit_model":
        inputs = (
            {"expected_revision": command["expected_model_revision"]}
            if "expected_model_revision" in command
            else {}
        )
    elif action == "prepare_data":
        inputs["source"] = {
            "raw_data": "files",
            "measurements": "raw_data",
            "simulated_measurements": "simulation",
        }[operation]
    record.update(action=action, inputs=inputs, operation_id=operation)
    return record


def _restore_check_targets(record, tree: pygit2.Tree):
    """Name archived construct checks from their retained model, without new measurements."""
    checks = record["diagnostics"].get("prior_predictive", {}).get("diagnostics", [])
    unnamed = [check for check in checks if "target" not in check]
    if not unnamed:
        return
    model = json.loads(tree["artifacts/model/model.json"].peel(pygit2.Blob).data)
    names = {}

    def collect(value):
        if isinstance(value, dict):
            if "id" in value and "name" in value:
                names[value["id"]] = value["name"]
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(model)
    for check in unnamed:
        check["target"] = names[check["construct_id"]]


def migrate_repository(source: Path, destination: Path) -> dict[str, str]:
    if destination.exists():
        raise FileExistsError(destination)
    original = pygit2.Repository(str(source))
    if original.config.get_int("nof1.format") != 2:
        raise ValueError("Expected a format-2 study repository")
    shutil.copytree(source, destination)
    repo = pygit2.Repository(str(destination))
    refs = {name: str(repo.references[name].target) for name in repo.references}
    artifacts = {oid for name, oid in refs.items() if name.startswith("refs/artifacts/")}
    commits: set[str] = set()
    for name, oid in refs.items():
        if name.startswith(("refs/heads/", "refs/attempts/")):
            commits.update(str(commit.id) for commit in repo.walk(pygit2.Oid(hex=oid)))
    for oid in commits:
        commit = repo[pygit2.Oid(hex=oid)].peel(pygit2.Commit)
        if "artifacts" in commit.tree:
            artifact_tree = commit.tree["artifacts"].peel(pygit2.Tree)
            artifacts.update(str(entry.id) for entry in artifact_tree)
    mapped: dict[str, str] = {}
    active: set[str] = set()

    def files(tree: pygit2.Tree, prefix: str = "") -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        for entry in tree:
            obj = repo[entry.id]
            assert entry.name is not None
            name = prefix + entry.name
            if isinstance(obj, pygit2.Tree):
                result.update(files(obj, name + "/"))
            else:
                result[name] = obj.peel(pygit2.Blob).data
        return result

    def rewrite(value):
        if isinstance(value, dict):
            return {
                rewrite(key): rewrite(item) for key, item in value.items() if key != "provenance"
            }
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            if value in artifacts or value in commits:
                return migrate(value)
            return value
        return value

    def migrate(oid: str) -> str:
        if oid in mapped:
            return mapped[oid]
        if oid in active:
            raise ValueError(f"Cyclic stored reference at {oid}")
        active.add(oid)
        obj = repo[pygit2.Oid(hex=oid)]
        if isinstance(obj, pygit2.Commit):
            parents = [pygit2.Oid(hex=migrate(str(parent.id))) for parent in obj.parents]
            content = files(obj.tree)
            for name in list(content):
                if name.startswith("artifacts/"):
                    del content[name]
            log_path = "logs/transition.json"
            if log_path in content:
                record = convert_transition(json.loads(content[log_path]))
                _restore_check_targets(record, obj.tree)
                content[log_path] = json.dumps(record).encode()
            for name, raw in list(content.items()):
                if name.endswith(".json"):
                    content[name] = json.dumps(rewrite(json.loads(raw)), sort_keys=True).encode()
            tree = repo.TreeBuilder(write_tree(repo, content))
            if "artifacts" in obj.tree:
                builder = repo.TreeBuilder()
                for entry in obj.tree["artifacts"].peel(pygit2.Tree):
                    artifacts.add(str(entry.id))
                    builder.insert(
                        entry.name, pygit2.Oid(hex=migrate(str(entry.id))), pygit2.GIT_FILEMODE_TREE
                    )
                tree.insert("artifacts", builder.write(), pygit2.GIT_FILEMODE_TREE)
            message = obj.message
            if log_path in content:
                record = json.loads(content[log_path])
                message = f"{record['action']} ({record['status']})"
            result = repo.create_commit(
                None, obj.author, obj.committer, message, tree.write(), parents
            )
        else:
            content = files(obj.peel(pygit2.Tree))
            for name, raw in list(content.items()):
                if name.endswith(".json"):
                    content[name] = json.dumps(rewrite(json.loads(raw)), sort_keys=True).encode()
            result = write_tree(repo, content)
        mapped[oid] = str(result)
        active.remove(oid)
        return str(result)

    for name, oid in refs.items():
        result = migrate(oid)
        if name.startswith("refs/artifacts/"):
            repo.references.delete(name)
            name = name.rsplit("/", 1)[0] + "/" + result
        repo.references.create(name, pygit2.Oid(hex=result), force=True)
    repo.config["nof1.format"] = 3
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = migrate_repository(args.source, args.destination)
    args.destination.with_suffix(".mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(f"Migrated {len(mapping)} stored objects into {args.destination}")


if __name__ == "__main__":
    main()
