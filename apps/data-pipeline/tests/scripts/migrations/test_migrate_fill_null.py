"""Offline field migration preserves study history, source bytes and numerical results."""

import json
from typing import TYPE_CHECKING

import polars as pl
import pytest
from scripts.migrations.migrate_fill_null import migrate_workspace

from nof1_causal_lab.actions.data_checks import read_data_metadata
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.machine.artifacts import ArtifactRecord
from nof1_causal_lab.machine.git_objects import write_tree
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
    migrated = migrate_workspace(source, destination)
    assert {
        name: str(history.repo.references[name].target) for name in history.repo.references
    } == refs
    assert json.loads(history.read_file(artifact.revision, "metadata.json")) == metadata
    monkeypatch.setattr(data, "_DATA_URI", str(destination.parent))
    updated = StudyRepository(destination.name)
    updated_store = ArtifactStore(destination.name)
    assert updated.branches() == {"main": migrated[first], "retained": migrated[first]}
    attempt = updated.read_attempt(2)
    assert attempt is not None
    assert attempt.commit_id == migrated[last]
    assert attempt.parent_ids == [migrated[first]]
    assert attempt.inputs["panel_revision"] == migrated[artifact.revision]
    assert attempt.inputs["model_inputs"] == current_inputs
    assert (
        updated_store.read_meta("model", migrated[model_artifact.revision]).model_inputs
        == current_inputs
    )
    revised = read_data_metadata(updated_store, GitOid(migrated[artifact.revision]))
    assert [item.model_dump()["fill_null"] for item in revised.variables] == [
        None,
        "forward",
        0.0,
    ]
    assert revised.variables[1].fill_null_limit == (2 if legacy_form == "nested" else None)
    assert updated_store.read_parquet_file(
        "panel", migrated[artifact.revision], "panel.parquet"
    ).equals(panel)
    inputs = json.loads(updated.read_file(migrated[first], "logs/transition.json"))["inputs"]
    assert inputs["input"]["variables"][1]["fill_null"] == "forward"
    for original in (source / "store").rglob("*"):
        if original.is_file():
            assert (
                destination / original.relative_to(source)
            ).read_bytes() == original.read_bytes()
