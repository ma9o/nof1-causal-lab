"""Offline schema migration keeps data definitions and history without scientific runs."""

import hashlib
import json

import polars as pl
import pygit2
import pytest
from scripts.migrations.migrate_data_preparation import migrate_workspace

from nof1_causal_lab.artifacts.data_preparation import FileSourceRef
from nof1_causal_lab.machine.git_objects import read_file, write_tree
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.snapshots import ModelReader
from tests.helpers import make_model

pytestmark = pytest.mark.contract


def test_migration_moves_scoring_to_data_and_preserves_original_workspace(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data

    source = tmp_path / "old" / "study"
    destination = tmp_path / "new" / "study"
    (source / "episode").mkdir(parents=True)
    repo = pygit2.init_repository(str(source / "episode/history.git"), bare=True)
    repo.config["nof1.format"] = 3
    signature = pygit2.Signature("test", "study@local", 1, 0)
    model = make_model(["stress"])
    payload = model.model_dump(mode="json")

    def add_scoring(value):
        if isinstance(value, dict):
            if "measurement_dtype" in value and "construct_polarity" in value:
                value.update(
                    how_to_measure="Extract the stated stress rating",
                    recording="samples",
                    extraction_mode="semantic",
                    source_columns=["diary"],
                    computed_rule=None,
                )
            for child in value.values():
                add_scoring(child)
        elif isinstance(value, list):
            for child in value:
                add_scoring(child)

    add_scoring(payload)

    def artifact(identity, files, pins):
        meta = {
            "artifact_id": identity,
            "derived_from": pins,
            "produced_by": "run:measurements" if identity == "panel" else "write:model",
            "created_at": "2026-01-01T00:00:00Z",
            "model_inputs": {},
            "consumed_model_inputs": {},
        }
        oid = write_tree(repo, {**files, "meta.json": json.dumps(meta).encode()})
        repo.references.create(f"refs/artifacts/{identity}/{oid}", oid)
        return {**meta, "revision": str(oid)}

    definition = artifact("model", {"model.json": json.dumps(payload).encode()}, {})
    table = pl.DataFrame(
        {
            "indicator_id": [model.indicators[0].id] * 2,
            "value": [1.0, 3.0],
            "anchor_time": ["2026-01-01", "2026-01-03"],
            "support_kind": model.indicators[0].support_kind.value,
            "summary_operator": model.indicators[0].summary_operator.value,
            "anchor_policy": model.indicators[0].anchor_policy.value,
            "observation_window": model.measurement_clock,
            "support_start": ["2025-12-31", "2026-01-02"],
            "support_end": ["2026-01-01", "2026-01-03"],
        }
    )
    parquet = source / "observations.parquet"
    table.write_parquet(parquet)
    blob_id = hashlib.sha256(parquet.read_bytes()).hexdigest()
    blob = source / "store/blobs" / blob_id
    blob.parent.mkdir(parents=True)
    parquet.rename(blob)
    panel = artifact(
        "panel",
        {"external.json": json.dumps({"panel.parquet": blob_id}).encode()},
        {"model": definition["revision"]},
    )
    record = {
        "seq": 1,
        "ts": "2026-01-01T00:00:00Z",
        "action": "prepare_data",
        "operation_id": "measurements",
        "inputs": {},
        "status": "applied",
        "produced": [definition, panel],
        "trace_ids": [],
        "resume": None,
    }
    artifacts = repo.TreeBuilder()
    for value in [definition, panel]:
        artifacts.insert(
            value["artifact_id"], pygit2.Oid(hex=value["revision"]), pygit2.GIT_FILEMODE_TREE
        )
    tree = repo.TreeBuilder(write_tree(repo, {"logs/transition.json": json.dumps(record).encode()}))
    tree.insert("artifacts", artifacts.write(), pygit2.GIT_FILEMODE_TREE)
    head = repo.create_commit(
        "refs/heads/main", signature, signature, "prepare data", tree.write(), []
    )
    repo.references.create("refs/attempts/1", head)
    original_refs = {name: str(repo.references[name].target) for name in repo.references}
    mapping = migrate_workspace(source, destination, files=("diary.csv",))
    assert repo.config.get_int("nof1.format") == 3
    assert {name: str(repo.references[name].target) for name in repo.references} == original_refs
    assert (
        "how_to_measure"
        in json.loads(read_file(repo, definition["revision"], "model.json"))["edges"][0]["cause"][
            "indicators"
        ][0]
    )
    monkeypatch.setattr(data, "_DATA_URI", str(destination.parent))
    history = StudyRepository("study")
    assert history.head() == mapping[str(head)]
    snapshot = ModelReader("study").snapshot()
    assert snapshot.model is not None
    assert snapshot.data.metadata is not None
    assert snapshot.data.metadata.value.preparation is not None
    assert isinstance(snapshot.data.metadata.value.source, FileSourceRef)
    assert snapshot.model.value == model
    assert (
        snapshot.data.metadata.value.preparation.variables[0].how_to_measure
        == "Extract the stated stress rating"
    )
    assert snapshot.data.metadata.value.source.files == ("diary.csv",)
    assert snapshot.data.measurements is not None
    assert snapshot.data.profile is not None
    assert history.state(history.head()).current["panel"].derived_from == {}
    assert (destination / "store/blobs" / blob_id).read_bytes() == blob.read_bytes()
