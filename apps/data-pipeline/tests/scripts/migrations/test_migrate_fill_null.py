"""Offline field migration preserves study history, source bytes and numerical results."""

import json
from typing import TYPE_CHECKING

import polars as pl
import pygit2
import pytest
from scripts.migrations.migrate_fill_null import migrate_workspace

from nof1_causal_lab.machine.artifacts import ArtifactRecord
from nof1_causal_lab.machine.git_objects import read_file, write_tree
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from nof1_causal_lab.models.model_inputs import input_fingerprints
from tests.helpers import make_model

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import UncheckedJsonObject

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("legacy_form", ["recording", "nested"])
def test_migrate_fill_null_preserves_history_and_observations(tmp_path, monkeypatch, legacy_form):
    from nof1_causal_lab.utils import data

    source = tmp_path / "old" / "study"
    destination = tmp_path / "new" / "study"
    monkeypatch.setattr(data, "_DATA_URI", str(source.parent))
    store = ArtifactStore(source.name)
    history = StudyRepository(source.name)
    variables: list[UncheckedJsonObject] = [
        {
            "id": f"indicator:{recording}",
            "name": recording,
            "measurement_dtype": "continuous",
            "aggregation": "sum" if recording == "events" else "last",
            "observation_window": "1d",
            "recording": recording,
        }
        for recording in ("samples", "changes", "events")
    ]
    if legacy_form == "nested":
        for variable in variables:
            variable["fill_null"] = {
                "samples": None,
                "changes": {"strategy": "forward", "limit": 2},
                "events": {"value": 0},
            }[variable.pop("recording")]
    metadata = {"source": {"file": "panel.parquet"}, "variables": variables}
    panel = pl.DataFrame({"value": [1.0, None, 3.0]})
    artifact = store.write_artifact(
        "panel",
        derived_from={},
        produced_by="run:observation_table",
        parquet_files={"panel.parquet": panel},
        json_files={"metadata.json": metadata},
    )
    model = make_model(["dose"])
    current_inputs = input_fingerprints(model)
    old_inputs = {purpose: f"archived-{purpose}" for purpose in current_inputs}
    model_meta = {
        **artifact.model_dump(mode="json", exclude={"revision"}),
        "artifact_id": "model",
        "produced_by": "write:model",
        "model_inputs": old_inputs,
    }
    model_revision = write_tree(
        store.repo,
        {
            "model.json": model.model_dump_json().encode(),
            "meta.json": json.dumps(model_meta).encode(),
        },
    )
    store.repo.references.create(f"refs/artifacts/model/{model_revision}", model_revision)
    model_artifact = ArtifactRecord.model_validate({**model_meta, "revision": str(model_revision)})
    first = history.append(
        TransitionRecord(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            action="prepare_data",
            status="applied",
            inputs={"input": metadata},
            produced=[model_artifact, artifact],
            trace_ids=[],
            resume=None,
        )
    )
    history.create_branch("retained", at=first)
    last = history.append(
        TransitionRecord(
            seq=2,
            ts="2026-01-02T00:00:00Z",
            action="prepare_data",
            status="rejected",
            inputs={"panel_revision": artifact.revision, "model_inputs": old_inputs},
            trace_ids=[],
            resume=None,
            reason="Retained rejected attempt",
        )
    )
    refs = {name: str(history.repo.references[name].target) for name in history.repo.references}
    history.repo.config["nof1.format"] = 4
    migrated = migrate_workspace(source, destination)
    assert {
        name: str(history.repo.references[name].target) for name in history.repo.references
    } == refs
    assert json.loads(history.read_file(artifact.revision, "metadata.json")) == metadata
    # This historical migration emits format 4. Inspect that archive directly;
    # current readers intentionally accept only the latest format.
    updated = pygit2.Repository(str(destination / "episode/history.git"))
    assert updated.config.get_int("nof1.format") == 4
    assert {name: str(updated.branches[name].target) for name in updated.branches} == {
        "main": migrated[first],
        "retained": migrated[first],
    }
    attempt = updated[updated.references["refs/attempts/2"].target].peel(pygit2.Commit)
    assert str(attempt.id) == migrated[last]
    assert [str(parent) for parent in attempt.parent_ids] == [migrated[first]]
    attempt_log = json.loads(read_file(updated, migrated[last], "logs/transition.json"))
    assert attempt_log["inputs"]["panel_revision"] == migrated[artifact.revision]
    assert attempt_log["inputs"]["model_inputs"] == current_inputs
    assert (
        json.loads(read_file(updated, migrated[model_artifact.revision], "meta.json"))[
            "model_inputs"
        ]
        == current_inputs
    )
    revised = json.loads(read_file(updated, migrated[artifact.revision], "metadata.json"))
    assert [item["fill_null"] for item in revised["variables"]] == [
        None,
        "forward",
        0.0,
    ]
    assert revised["variables"][1].get("fill_null_limit") == (
        2 if legacy_form == "nested" else None
    )
    external = json.loads(read_file(updated, migrated[artifact.revision], "external.json"))
    assert pl.read_parquet(destination / "store/blobs" / external["panel.parquet"]).equals(panel)
    inputs = json.loads(read_file(updated, migrated[first], "logs/transition.json"))["inputs"]
    assert inputs["input"]["variables"][1]["fill_null"] == "forward"
    for original in (source / "store").rglob("*"):
        if original.is_file():
            assert (
                destination / original.relative_to(source)
            ).read_bytes() == original.read_bytes()
