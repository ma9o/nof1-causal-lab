"""Fixed-value migration preserves literals, observations and stored revision identity."""

import copy
import json

import pygit2
import pytest
from scripts.migrations.migrate_fixed_values import migrate_workspace

from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.git_objects import read_file, write_tree
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.models.model_inputs import input_fingerprints
from tests.helpers import graph_constructs, make_model

pytestmark = pytest.mark.contract
FIXED = scientific_id("parameter", "fixed")
FREE = scientific_id("parameter", "free")


def _legacy_model():
    payload = make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")
    first, second = graph_constructs(payload)
    first["coefficients"] = [
        {"kind": "coefficient", "role": "initial_mean", "value": FIXED},
        {"kind": "coefficient", "role": "initial_scale", "value": 1.0},
    ]
    second["coefficients"] = [{"kind": "coefficient", "role": "initial_mean", "value": FREE}]
    payload["parameters"] = [
        {"id": FIXED, "name": "fixed", "description": "Known", "value": 0.0},
        {"id": FREE, "name": "free", "description": "Unknown", "value": None},
    ]
    return payload


def test_migration_inlines_and_rekeys_without_changing_source(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data

    source, destination = tmp_path / "old/study", tmp_path / "new/study"
    monkeypatch.setattr(data, "_DATA_URI", str(source.parent))
    store = ArtifactStore(source.name)
    legacy = _legacy_model()
    expected = copy.deepcopy(legacy)
    graph_constructs(expected)[0]["coefficients"][0]["value"] = 0.0
    expected["parameters"] = [
        {key: value for key, value in legacy["parameters"][1].items() if key != "value"}
    ]
    model = ModelSpec.model_validate(expected)
    old_inputs = {purpose: "old-" + value for purpose, value in input_fingerprints(model).items()}
    revision = write_tree(
        store.repo,
        {
            "model.json": json.dumps(legacy).encode(),
            "meta.json": json.dumps({"model_inputs": old_inputs}).encode(),
        },
    )
    store.repo.references.create(f"refs/artifacts/model/{revision}", revision)
    observations = [{"value": 4.0}, {"value": None}]
    tree = write_tree(
        store.repo,
        {
            "artifacts/model/model.json": json.dumps(legacy).encode(),
            "artifacts/model/meta.json": json.dumps({"model_inputs": old_inputs}).encode(),
            "transition.json": json.dumps(
                {
                    "inputs": {"model": legacy},
                    "revision": str(revision),
                    "model_inputs": old_inputs,
                    "observations": observations,
                    "parameter": legacy["parameters"][1],
                }
            ).encode(),
        },
    )
    parent = store.repo.head.peel(pygit2.Commit)
    commit = store.repo.create_commit(
        "refs/heads/main", parent.author, parent.committer, "Author model", tree, [parent.id]
    )
    array = source / "store/arrays/saved.bin"
    array.parent.mkdir(parents=True)
    array.write_bytes(b"saved numerical data")
    store.repo.config["nof1.format"] = 6
    before_refs = {name: str(store.repo.references[name].target) for name in store.repo.references}

    mapping = migrate_workspace(source, destination)

    assert store.repo.config.get_int("nof1.format") == 6
    assert before_refs == {
        name: str(store.repo.references[name].target) for name in store.repo.references
    }
    assert json.loads(read_file(store.repo, str(revision), "model.json")) == legacy
    updated = pygit2.Repository(str(destination / "episode/history.git"))
    assert updated.config.get_int("nof1.format") == 7
    assert (destination / "store/arrays/saved.bin").read_bytes() == array.read_bytes()
    assert f"refs/artifacts/model/{mapping[str(revision)]}" in updated.references
    assert str(updated.head.target) == mapping[str(commit)]
    converted = json.loads(read_file(updated, mapping[str(revision)], "model.json"))
    assert converted == expected
    assert ModelSpec.model_validate(converted) == model
    transition = json.loads(read_file(updated, mapping[str(commit)], "transition.json"))
    assert transition["inputs"]["model"] == expected
    assert transition["revision"] == mapping[str(revision)]
    assert transition["model_inputs"] == input_fingerprints(model)
    assert transition["observations"] == observations
    assert "value" not in transition["parameter"]
