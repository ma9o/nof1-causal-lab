"""Offline timing migration preserves declarations and recomputes identification."""

import json

import pygit2
import pytest
from scripts.migrations.migrate_edge_timing import migrate_workspace

from nof1_causal_lab.machine.git_objects import read_file, write_tree
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.models.identification import identify_model
from nof1_causal_lab.models.model_inputs import input_fingerprints
from tests.helpers import make_model

pytestmark = pytest.mark.contract


def test_migration_removes_flag_and_recomputes_query_without_changing_source(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data

    source, destination = tmp_path / "old" / "study", tmp_path / "new" / "study"
    monkeypatch.setattr(data, "_DATA_URI", str(source.parent))
    store = ArtifactStore(source.name)
    model = make_model(["X", "Y"], [("X", "Y")])
    model = model.revised(default_outcome=model.constructs[-1].id)
    legacy = model.model_dump(mode="json")
    legacy["edges"][0]["lagged"] = False
    revision = write_tree(
        store.repo,
        {
            "model.json": json.dumps(legacy).encode(),
            "meta.json": json.dumps({"model_inputs": input_fingerprints(model)}).encode(),
        },
    )
    store.repo.references.create(f"refs/artifacts/model/{revision}", revision)
    report_revision = write_tree(
        store.repo,
        {
            "identification_report.json": b'{"outcome": null, "treatments": {}}',
            "meta.json": json.dumps({"derived_from": {"model": str(revision)}}).encode(),
        },
    )
    store.repo.references.create(
        f"refs/artifacts/identification_report/{report_revision}", report_revision
    )
    store.repo.config["nof1.format"] = 5
    before_refs = {name: str(store.repo.references[name].target) for name in store.repo.references}

    mapping = migrate_workspace(source, destination)

    assert before_refs == {
        name: str(store.repo.references[name].target) for name in store.repo.references
    }
    assert json.loads(read_file(store.repo, str(revision), "model.json")) == legacy
    updated = pygit2.Repository(str(destination / "episode/history.git"))
    assert updated.config.get_int("nof1.format") == 6
    assert json.loads(read_file(updated, mapping[str(revision)], "model.json")) == model.model_dump(
        mode="json"
    )
    assert json.loads(
        read_file(updated, mapping[str(report_revision)], "identification_report.json")
    ) == identify_model(model).model_dump(mode="json")
    assert (
        json.loads(read_file(updated, mapping[str(report_revision)], "meta.json"))["derived_from"][
            "model"
        ]
        == mapping[str(revision)]
    )
