"""Rewrite a stopped study's Git graph into a new copy with translated JSON payloads."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

import pygit2

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_inputs import input_fingerprints
from nof1_causal_lab.utils.arrays import read_array

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def rewrite_study(
    source: Path,
    destination: Path,
    update: Callable[[Any], Any],
    *,
    mapping_name: str,
    source_format: int,
    target_format: int,
    update_file: Callable[[str, str, Any], Any] | None = None,
    rename_entry: Callable[[str], str | None] | None = None,
    layout: tuple[str, str] = ("study", "study"),
) -> dict[str, str]:
    """Copy a stopped study and rewrite its Git graph and stored revision references.

    `update` translates every stored JSON payload and `update_file` translates one by
    name. `rename_entry` renames a stored file, or drops it by returning None. `layout`
    names the directory holding the repository before and after the copy. Changed
    objects get new Git identities and references to them follow. Model owners
    receive translated input fingerprints; saved readers retain their consumed inputs.
    """
    source_layout, destination_layout = layout
    if destination.exists() or destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Choose a new destination outside the source workspace")
    original = pygit2.Repository(str(source / source_layout / "history.git"))
    if original.config.get_int("nof1.format") != source_format:
        raise ValueError(f"Expected a format-{source_format} study")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("cache", "scratch"))
    if source_layout != destination_layout:
        (destination / source_layout).rename(destination / destination_layout)
    repo = pygit2.Repository(str(destination / destination_layout / "history.git"))
    refs = {name: str(repo.references[name].target) for name in repo.references}
    revisions = set(refs.values())
    models = {oid for name, oid in refs.items() if name.startswith("refs/artifacts/model/")}
    for name, oid in refs.items():
        if name.startswith(("refs/heads/", "refs/attempts/")):
            for commit in repo.walk(pygit2.Oid(hex=oid)):
                revisions.add(str(commit.id))
                if "artifacts" in commit.tree:
                    revisions.update(
                        str(entry.id) for entry in commit.tree["artifacts"].peel(pygit2.Tree)
                    )
                if "artifacts/model" in commit.tree:
                    models.add(str(commit.tree["artifacts/model"].id))
    # Model owners receive their translated fingerprints. Saved readers keep
    # their original inputs, so the existing freshness comparison marks them
    # stale when the model's scientific meaning changed.
    model_inputs: dict[str, dict[str, str]] = {}
    for oid in models:
        tree = repo[pygit2.Oid(hex=oid)].peel(pygit2.Tree)
        model = ModelSpec.model_validate(
            update(json.loads(tree["model.json"].peel(pygit2.Blob).data)),
            context={
                "distribution_array_loader": lambda ref: read_array(
                    str(destination / "store/arrays"), ref
                )
            },
        )
        model_inputs[oid] = input_fingerprints(model)
    mapping: dict[str, str] = {}
    active: set[str] = set()

    def rewrite(value):
        if isinstance(value, dict):
            return {rewrite(key): rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str) and value in revisions:
            return migrate(value)
        return value

    def migrate(oid):
        if oid in mapping:
            return mapping[oid]
        if oid in active:
            raise ValueError(f"Cyclic stored reference: {oid}")
        active.add(oid)
        obj = repo[pygit2.Oid(hex=oid)]
        if isinstance(obj, pygit2.Commit):
            parents = [pygit2.Oid(hex=migrate(str(parent.id))) for parent in obj.parents]
            tree = pygit2.Oid(hex=migrate(str(obj.tree_id)))
            result = repo.create_commit(None, obj.author, obj.committer, obj.message, tree, parents)
        else:
            tree = obj.peel(pygit2.Tree)
            builder = repo.TreeBuilder(tree)
            for entry in tree:
                assert entry.name is not None
                name = rename_entry(entry.name) if rename_entry is not None else entry.name
                if name != entry.name:
                    builder.remove(entry.name)
                if name is None:
                    continue
                child = repo[entry.id]
                if isinstance(child, pygit2.Tree):
                    replacement = pygit2.Oid(hex=migrate(str(entry.id)))
                elif entry.name.endswith(".json"):
                    payload = json.loads(child.peel(pygit2.Blob).data)
                    owner_payload = (
                        {**payload, "model_inputs": model_inputs[oid]}
                        if entry.name == "meta.json" and oid in model_inputs
                        else payload
                    )
                    updated = rewrite(
                        update(
                            update_file(oid, entry.name, owner_payload)
                            if update_file is not None
                            else owner_payload
                        )
                    )
                    replacement = (
                        entry.id
                        if updated == payload
                        else repo.create_blob(json.dumps(updated, sort_keys=True).encode())
                    )
                else:
                    replacement = entry.id
                builder.insert(name, replacement, entry.filemode)
            result = builder.write()
        mapping[oid] = str(result)
        active.remove(oid)
        return str(result)

    replacements = {name: migrate(oid) for name, oid in refs.items()}
    for name, oid in replacements.items():
        target = name.rsplit("/", 1)[0] + "/" + oid if name.startswith("refs/artifacts/") else name
        if target != name:
            repo.references.delete(name)
        repo.references.create(target, pygit2.Oid(hex=oid), force=True)
    (destination / mapping_name).write_text(json.dumps(mapping, indent=2) + "\n")
    repo.config["nof1.format"] = target_format
    return mapping
