"""Offline action migration preserves branches, failed attempts, and exact input links."""

import json

import pygit2
import pytest
from scripts.migrate_action_history import migrate_repository

from nof1_causal_lab.machine.git_objects import read_file, write_tree
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore
from tests.helpers import make_model

pytestmark = pytest.mark.contract


def test_migration_preserves_science_and_rewrites_all_reference_kinds(tmp_path):
    source, destination = tmp_path / "old.git", tmp_path / "new.git"
    repo = pygit2.init_repository(str(source), bare=True)
    repo.config["nof1.format"] = 2
    signature = pygit2.Signature("test", "study@local", 1, 0)
    root = repo.create_commit(None, signature, signature, "init", write_tree(repo, {}), [])
    model = make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")

    def artifact(parent=None):
        metadata = {
            "artifact_id": "model",
            "provenance": "llm",
            "derived_from": {"model": parent} if parent else {},
            "produced_by": "edit_model",
            "created_at": "2026-01-01T00:00:00Z",
        }
        oid = write_tree(
            repo,
            {"model.json": json.dumps(model).encode(), "meta.json": json.dumps(metadata).encode()},
        )
        repo.references.create(f"refs/artifacts/model/{oid}", oid)
        return {**metadata, "revision": str(oid)}

    def commit(seq, parent, info, command, *, branch="main", status="applied", diagnostics=None):
        record = {
            "seq": seq,
            "ts": "2026-01-01T00:00:00Z",
            "branch": branch,
            "status": status,
            "move": command,
            "produced": [info] if status == "applied" else [],
            "retracted": [],
            "trace_ids": ["author"],
            "resume": None,
            "diagnostics": diagnostics or {},
        }
        artifacts = repo.TreeBuilder()
        artifacts.insert("model", pygit2.Oid(hex=info["revision"]), pygit2.GIT_FILEMODE_TREE)
        tree = repo.TreeBuilder(
            write_tree(
                repo,
                {
                    "logs/transition.json": json.dumps(record).encode(),
                    "logs/traces/author.json": b'{"messages": [], "model": "test"}',
                },
            )
        )
        tree.insert("artifacts", artifacts.write(), pygit2.GIT_FILEMODE_TREE)
        oid = repo.create_commit(None, signature, signature, "old command", tree.write(), [parent])
        repo.references.create(f"refs/attempts/{seq}", oid)
        if status == "applied":
            repo.references.create(f"refs/heads/{branch}", oid, force=True)
        return oid

    first = artifact()
    common = commit(
        1,
        root,
        first,
        {
            "kind": "write",
            "artifact_id": "model",
            "provenance": "llm",
            "expected_model_revision": None,
        },
    )
    second = artifact(first["revision"])
    main = commit(
        2,
        common,
        second,
        {
            "kind": "write",
            "artifact_id": "model",
            "provenance": "human",
            "expected_model_revision": first["revision"],
        },
    )
    fork = commit(
        3,
        common,
        first,
        {
            "kind": "run",
            "operation_id": "simulate",
            "input_revisions": {"model": first["revision"]},
        },
        branch="alternative",
        diagnostics={"reference": {"revision": str(common), "path": "logs/transition.json"}},
    )
    rejected = commit(
        4,
        main,
        second,
        {
            "kind": "run",
            "operation_id": "posterior",
            "input_revisions": {"model": second["revision"]},
        },
        status="rejected",
    )
    original_refs = {name: str(repo.references[name].target) for name in repo.references}
    mapping = migrate_repository(source, destination)
    assert repo.config.get_int("nof1.format") == 2
    assert {name: str(repo.references[name].target) for name in repo.references} == original_refs
    old_record = json.loads(read_file(repo, str(common), "logs/transition.json"))
    assert "move" in old_record
    assert old_record["produced"][0]["provenance"] == "llm"

    migrated = StudyRepository("MIGRATED", repository_path=destination)
    store = ArtifactStore("MIGRATED", repository_path=destination)
    assert migrated.branches() == {"main": mapping[str(main)], "alternative": mapping[str(fork)]}
    records = migrated.attempts()
    assert [record.action for record in records] == ["edit_model", "edit_model", "simulate", "fit"]
    assert records[1].parent_ids == records[2].parent_ids == [mapping[str(common)]]
    assert records[3].commit_id == mapping[str(rejected)]
    assert records[3].status == "rejected"
    assert records[1].inputs["expected_revision"] == mapping[first["revision"]]
    assert records[2].diagnostics["reference"]["revision"] == mapping[str(common)]
    for info in (first, second):
        revision = mapping[info["revision"]]
        assert store.read_json_file("model", revision, "model.json") == model
        assert "provenance" not in store.read_meta("model", revision).model_dump()
    assert store.read_meta("model", mapping[second["revision"]]).derived_from == {
        "model": mapping[first["revision"]]
    }
    assert (
        migrated.read_file(records[0].commit_id, "logs/traces/author.json")
        == b'{"messages": [], "model": "test"}'
    )
    with pytest.raises(FileExistsError):
        migrate_repository(source, destination)
