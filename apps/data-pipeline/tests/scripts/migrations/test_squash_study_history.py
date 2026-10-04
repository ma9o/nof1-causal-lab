"""Compacting fact-only history preserves input closure, saved bytes and leaf limitations."""

from datetime import UTC, date, datetime
from uuid import uuid4

import pygit2
import pytest
from scripts.migrations.squash_study_history import copy_squashed, plan_squash

from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.identity import GitOid, GitRef
from nof1_causal_lab.artifacts.posterior import InferenceEvidence
from nof1_causal_lab.artifacts.simulation import SimulationEvidence, SimulationSpec
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, AttemptRecord, DataDiffAttempt, DataPreparationResult, EditAttempt, ModelFitResult, ModelSimulationResult, Raised
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.study.view_models import DataDiffRequest, PanelRef
from nof1_causal_lab.utils import data
from tests.action_fixtures import applied_record, question_root
from tests.model_fixtures import load_model_fixture, x_y_model

pytestmark = pytest.mark.contract


@pytest.fixture
def study(tmp_path, monkeypatch):
    source, destination = tmp_path / "original/study", tmp_path / "squashed/study"
    monkeypatch.setattr(data, "_DATA_URI", str(source.parent))
    question_root("study")
    return source, destination, ArtifactStore("study"), StudyRepository("study")


def _append(history, result):
    return history.append(applied_record(result, seq=history.latest_seq() + 1,
        attempt_id=uuid4(), ts="2026-09-30T12:00:00Z"),
        logs={"notes.json": b'[ { "saved": true } ]\n'}).commit_id


def _edit(store, history, model=None):
    parent = history.state(history.head()).get("model")
    info = store.write_artifact("model", derived_from={"model": parent.revision} if parent else {},
        produced_by="edit_model", json_files={"model.json": (model or x_y_model()).model_dump(mode="json", round_trip=True)})
    return _append(history, Applied(result=None, effects=ActionEffects(produced=(info,))))


def _prepare(store, history):
    raw = store.write_artifact("raw_data", derived_from={}, produced_by="prepare_data")
    panel = store.write_artifact("panel", derived_from={"raw_data": raw.revision}, produced_by="prepare_data",
        json_files={"metadata.json": {"source": {"files": ["diary.csv"]}}})
    return _append(history, Applied(result=DataPreparationResult(), effects=ActionEffects(produced=(raw, panel))))


def _fit(store, history):
    state = history.state(history.head())
    model = load_model_fixture("causal_proofs/conditioned_treatment_outcome.json")
    pins = {key: state.current[key].revision for key in ("model", "panel")}
    info = store.write_artifact("model", derived_from=pins, produced_by="fit",
        json_files={"model.json": model.model_dump(mode="json", round_trip=True)})
    result = ModelFitResult(model=GitRef(workspace_id="study", revision=pins["model"], path="model.json"),
        panel=GitRef(workspace_id="study", revision=pins["panel"], path="panel.parquet"),
        evidence=InferenceEvidence(distribution=next(iter(model.law_layouts)), time_origin=None, duration_seconds=1))
    return _append(history, Applied(result=result, effects=ActionEffects(produced=(info,))))


def _simulate(history):
    evidence = SimulationEvidence(model=GitRef(workspace_id="study", revision=history.state(history.head()).current["model"].revision, path="model.json"),
        design=SimulationSpec(start=date(2026, 1, 1), horizon="1d"), time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        times=(0, 1), draws=1, seed=history.latest_seq(), state_ids=(), parameter_draws={}, latent_paths="latents", observations="observations",
        observation_layout={"variables": (), "support_start_times": "starts", "support_end_times": "ends", "mask": "mask"})
    return _append(history, Applied(result=ModelSimulationResult(evidence=evidence), effects=ActionEffects()))


def test_compaction_keeps_named_fit_inputs_original_facts_and_suffix_bytes(study):
    source, destination, store, history = study
    unused = _edit(store, history)
    selected = _edit(store, history)
    prepared = _prepare(store, history)
    fitted = _fit(store, history)
    boundary = _edit(store, history, load_model_fixture("causal_proofs/conditioned_treatment_outcome.json"))
    failed = history.append(AttemptRecord(seq=history.latest_seq() + 1, ts="2026-09-30T12:01:00Z",
        attempt=EditAttempt(action="edit_model", request=None, outcome=Raised(error_type="Saved", error_message="failure")))).commit_id
    array = source / "store/arrays/saved"
    array.parent.mkdir(parents=True)
    array.write_bytes(b"unchanged native bytes")
    original = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    mapping = copy_squashed(plan_squash(source, at=boundary), destination)
    assert mapping[unused] is None
    assert all(mapping[commit] is not None for commit in (selected, prepared, fitted, boundary, failed))
    moved = StudyRepository("study", repository_path=destination / "study/history.git")
    assert history.state(boundary) == moved.state(mapping[boundary])
    for record in history.attempts():
        target = mapping[record.commit_id]
        if target is not None:
            assert history.repo[record.commit_id].tree["logs"].id == moved.repo[target].tree["logs"].id
            assert "checks.json" not in moved.repo[target].peel(pygit2.Commit).tree
    assert (destination / "store/arrays/saved").read_bytes() == array.read_bytes()
    assert original == {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}


@pytest.mark.parametrize("stale", (False, True))
def test_only_the_latest_simulation_of_the_boundary_model_is_retained(study, stale):
    source, destination, store, history = study
    _edit(store, history)
    first = _simulate(history)
    last = _simulate(history)
    if stale:
        _edit(store, history)
    boundary = _prepare(store, history)
    mapping = copy_squashed(plan_squash(source, at=boundary), destination)
    assert mapping[first] is None
    assert (mapping[last] is None) == stale


def test_recorded_comparisons_and_nonnew_destinations_are_refused(study):
    source, destination, store, history = study
    boundary = _edit(store, history)
    with pytest.raises(ValueError, match="new destination"):
        copy_squashed(plan_squash(source, at=boundary), source)
    request = DataDiffRequest(left=PanelRef(revision=boundary), right=PanelRef(revision=boundary))
    history.append(AttemptRecord(seq=history.latest_seq() + 1, ts="2026-09-30T12:01:00Z",
        attempt=DataDiffAttempt(action="data_diff", request=request, outcome=Applied(result=None, effects=ActionEffects()))))
    with pytest.raises(ValueError, match="sequential branch|Recorded comparisons"):
        plan_squash(source, at=boundary)
    assert not destination.exists()
