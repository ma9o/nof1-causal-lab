"""Saved-effect compaction preserves state and the reader's fresh findings."""

from pathlib import Path

import json
from uuid import uuid4

import numpyro.distributions as dist
import pygit2
import pytest
from scripts.migrations.squash_study_history import copy_squashed, plan_squash

from nof1_causal_lab.artifacts.checks import SpecificationReport
from nof1_causal_lab.artifacts.identity import GitOid, GitRef
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.simulation import SimulationReport, SimulationSpec
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import AttemptRecord
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.state import RetractedArtifact
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils import data
from tests.integration.runner_fixtures import scientific_model

pytestmark = pytest.mark.contract


@pytest.fixture
def study(tmp_path, monkeypatch):
    source = tmp_path / "original" / "study"
    destination = tmp_path / "squashed" / "study"
    monkeypatch.setattr(data, "_DATA_URI", str(source.parent))
    return source, destination, ArtifactStore("study"), StudyRepository("study")


def _append(history, action="edit_model", **fields):
    seq = history.latest_seq() + 1
    return history.append(
        AttemptRecord(
            seq=seq,
            attempt_id=uuid4(),
            ts=f"2026-09-30T12:00:{seq:02d}Z",
            action=action,
            status=fields.pop("status", "applied"),
            trace_ids=[],
            **fields,
        ),
        logs={"notes.json": b'[ { "saved": true } ]\n', "worker.txt": b"original log\n"},
    )


def _model(store, history, *, model=None, fit=False, pins=None):
    current = history.state(history.head()).get("model")
    return store.write_artifact(
        "model",
        derived_from=pins if pins is not None else ({"model": current.revision} if current else {}),
        produced_by="fit" if fit else "edit_model",
        json_files={
            "model.json": (
                model or ModelSpec(question=f"Question {history.latest_seq() + 1}")
            ).model_dump(mode="json")
        },
    )


def _checks(seq):
    return ModelCheckReport(
        input_keys={"specification": str(seq)}, specification=SpecificationReport(findings=())
    )


def _edit(store, history, *, model=None, retracted=()):
    artifact = _model(store, history, model=model)
    return _append(
        history,
        produced=[artifact],
        checks=_checks(history.latest_seq() + 1),
        retracted=list(retracted),
    )


def _prepare(store, history):
    raw = store.write_artifact(
        "raw_data",
        derived_from={},
        produced_by="prepare_data",
        json_files={"data.json": {"seq": history.latest_seq() + 1}},
    )
    panel = store.write_artifact(
        "panel",
        derived_from={"raw_data": raw.revision},
        produced_by="prepare_data",
        json_files={"metadata.json": {"source": {"files": ["diary.csv"]}}},
    )
    profile = store.write_artifact(
        "data_profile",
        derived_from={"panel": panel.revision},
        produced_by="check:data",
        json_files={"profile.json": {}},
    )
    return _append(
        history,
        "prepare_data",
        produced=[raw, panel, profile],
        diagnostics={"input_pins": {"raw_data": raw.revision}},
    )


def _fit(store, history, *, model=None, pins=None):
    if pins is None:
        state = history.state(history.head())
        pins = {key: state.current[key].revision for key in ("model", "panel")}
    artifact = _model(store, history, model=model, fit=True, pins=pins)
    report = InferenceReport(
        time_origin=None,
        inference_metadata={
            "method": "marginal_particle_gibbs",
            "n_samples": 3,
            "duration_seconds": 1,
        },
    )
    return _append(
        history,
        "fit",
        produced=[artifact],
        checks=_checks(history.latest_seq() + 1),
        diagnostics={"input_pins": pins, "report": report.model_dump(mode="json")},
    )


def _simulate(history, *, model_revision=None):
    state = history.state(history.head())
    pins = {"model": model_revision or state.current["model"].revision}
    if state.has("panel"):
        pins["panel"] = state.current["panel"].revision
    report = SimulationReport(
        model=GitRef(workspace_id="study", revision=pins["model"], path="model.json"),
        design=SimulationSpec(end=1),
        time_origin=None,
        times=(0, 1),
        draws=1,
        seed=history.latest_seq() + 1,
        state_ids=(),
        parameter_draws={},
        latent_paths="saved-latents",
        observations="saved-observations",
        observation_layout={
            "variables": [],
            "support_start_times": "starts",
            "support_end_times": "ends",
            "mask": "mask",
        },
        predictive={"states": {}, "indicators": {}, "fit_reliability": "not_fitted"},
    )
    return _append(
        history,
        "simulate",
        diagnostics={"input_pins": pins, "report": report.model_dump(mode="json")},
    )


def _kept_seqs(plan):
    return [r.seq for r in plan.records if r.commit_id in plan.kept]


def _refs(repo):
    return {name: str(repo.references[name].target) for name in repo.references}


def _space_checks(history, revision):
    """Retain valid JSON with bytes that a Pydantic reserialization would change."""
    repo = history.repo
    original = repo[revision].peel(pygit2.Commit)
    tree = repo.TreeBuilder(original.tree)
    payload = json.loads(history.read_file(revision, "checks.json"))
    tree.insert(
        "checks.json",
        repo.create_blob(json.dumps(payload, indent=4).encode()),
        pygit2.GIT_FILEMODE_BLOB,
    )
    rewritten = repo.create_commit(
        None,
        original.author,
        original.committer,
        original.message,
        tree.write(),
        original.parent_ids,
    )
    for ref, target in _refs(repo).items():
        if target == revision:
            repo.references[ref].set_target(rewritten)
    return GitOid(str(rewritten))


def test_example_preserves_boundary_suffix_objects_and_attempt_identity(study, monkeypatch):
    source, destination, store, history = study
    _edit(store, history)  # 1
    _edit(store, history)  # 2
    _prepare(store, history)  # 3: raw data is consumed within this same action.
    _fit(store, history)  # 4
    edit = _edit(store, history)  # 5: its authorship base is exempt.
    stale_simulation = _simulate(history)  # 6
    boundary = _space_checks(history, _fit(store, history))  # 7
    failure = _append(history, status="rejected", reason="Saved rejection")  # 8
    later = _edit(store, history)  # 9
    last_failure = _append(history, status="raised", error_type="SavedError")  # 10
    arrays = source / "store/arrays"
    arrays.mkdir(parents=True)
    (arrays / "saved").write_bytes(b"saved numerical bytes")
    refs_before = _refs(history.repo)
    original_files = {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    fresh_report = ModelReader("study", at=boundary).inference_report
    assert fresh_report is not None
    assert fresh_report.source.validity == "fresh"
    assert _kept_seqs(plan_squash(source, at=boundary)) == [3, 5, 7, 8, 9, 10]

    mapping = copy_squashed(plan_squash(source, at=boundary), destination)
    retained = {old: GitOid(new) for old, new in mapping.items() if new is not None}
    monkeypatch.setattr(data, "_DATA_URI", str(destination.parent))
    squashed = StudyRepository("study")
    assert [r.seq for r in squashed.attempts()] == [3, 5, 7, 8, 9, 10]
    assert mapping[stale_simulation] is None
    assert json.loads((destination / "squash-mapping.json").read_text()) == mapping
    for old in (boundary, failure, later, last_failure):
        new = mapping[old]
        assert new is not None
        assert history.state(old) == squashed.state(new)
        assert history.repo[old].tree["artifacts"].id == squashed.repo[new].tree["artifacts"].id
        assert history.read_file(old, "checks.json") == squashed.read_file(new, "checks.json")
    for record in history.attempts():
        new = mapping[record.commit_id]
        if new is None:
            assert f"refs/actions/{record.attempt_id}" not in squashed.repo.references
            continue
        original = history.repo[record.commit_id].peel(pygit2.Commit)
        rewritten = squashed.repo[new].peel(pygit2.Commit)
        assert original.tree["logs"].id == rewritten.tree["logs"].id
        assert (original.author, original.committer, original.message) == (
            rewritten.author,
            rewritten.committer,
            rewritten.message,
        )
        assert record.attempt_id is not None
        receipt = squashed.dispatched_attempt(record.attempt_id)
        assert receipt is not None
        assert receipt.commit_id == new
    assert squashed.record(retained[failure]).parent_ids == [retained[boundary]]
    assert squashed.record(retained[last_failure]).parent_ids == [retained[later]]
    # The authored base survives as an artifact, while the visible parent changes.
    edited = history.state(edit).current["model"]
    assert store.read_meta("model", edited.derived_from["model"])
    assert squashed.record(retained[edit]).parent_ids != history.record(edit).parent_ids
    after = ModelReader("study", at=retained[boundary])
    assert after.inference_report is not None
    assert after.inference_report.value == fresh_report.value
    assert after.inference_report.source.validity == "fresh"
    assert after.simulation() is None
    assert {k: v for k, v in _refs(squashed.repo).items() if k.startswith("refs/artifacts/")} == {
        k: v for k, v in refs_before.items() if k.startswith("refs/artifacts/")
    }
    assert (destination / "store/arrays/saved").read_bytes() == b"saved numerical bytes"
    assert _refs(history.repo) == refs_before
    assert original_files == {
        p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize("stale", [False, True])
def test_only_latest_fresh_simulation_survives_and_old_findings_do_not_resurface(
    study, monkeypatch, stale
):
    source, destination, store, history = study
    first = _edit(store, history)  # 1
    _simulate(history)  # 2
    _simulate(history)  # 3, both discarded if stale at R.
    if stale:
        _edit(store, history)  # 4
    _append(history, status="rejected")
    boundary = _prepare(store, history)  # R is always retained; failures before it are dropped.
    before = ModelReader("study", at=boundary).simulation()
    plan = plan_squash(source, at=boundary)
    assert _kept_seqs(plan) == ([4, 6] if stale else [1, 3, 5])
    mapping = copy_squashed(plan_squash(source, at=boundary), destination)
    monkeypatch.setattr(data, "_DATA_URI", str(destination.parent))
    rewritten = mapping[boundary]
    assert rewritten is not None
    after = ModelReader("study", at=GitOid(rewritten)).simulation()
    if stale:
        assert mapping[first] is None
        assert after is None
    else:
        assert after is not None
        assert after.source.validity == "fresh"
        assert before is not None
        assert before.value == after.value


@pytest.mark.parametrize(('inherit', 'scientific_model_payload'), [
    pytest.param('none', 'squash_study_history/inherited_laws_keep_their_fit_but_reset_laws_do_not_scientific_model_none.json', id='none'),
    pytest.param('all', 'squash_study_history/inherited_laws_keep_their_fit_but_reset_laws_do_not_scientific_model_all.json', id='all'),
    pytest.param('some', 'squash_study_history/inherited_laws_keep_their_fit_but_reset_laws_do_not_scientific_model_some.json', id='some'),
])
def test_inherited_laws_keep_their_fit_but_reset_laws_do_not(study, inherit, scientific_model_payload):
    source, destination, store, history = study
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / scientific_model_payload).read_text())
    _edit(store, history, model=model)  # 1
    _prepare(store, history)  # 2
    # Saved laws are explicit test values; no inference or simulation executes.
    conditioned = model.revised(
        distributions={key: dist.Normal(42, 0.2) for key in model.distributions}
    )
    fitted = _fit(store, history, model=conditioned)  # 3
    edited = (model if inherit == "none" else conditioned).revised(question="Revised question")
    if inherit == "some":
        key = next(iter(model.distributions))
        edited = edited.revised(
            distributions={**edited.distributions, key: model.distributions[key]}
        )
    _edit(store, history, model=edited)  # 4
    boundary = _simulate(history)  # 5
    assert _kept_seqs(plan_squash(source, at=boundary)) == (
        [2, 4, 5] if inherit == "none" else [1, 2, 3, 4, 5]
    )
    mapping = copy_squashed(plan_squash(source, at=boundary), destination)
    assert (mapping[fitted] is not None) == (inherit != "none")


@pytest.mark.parametrize("fit_in_suffix", [False, True])
def test_closure_uses_original_edit_data_and_explicit_historical_fit_pins(study, fit_in_suffix):
    source, _, store, history = study
    _edit(store, history)  # 1
    prepared = _prepare(store, history)  # 2
    edit = _edit(store, history)  # 3 consumes panel/profile 2.
    _prepare(store, history)  # 4 becomes current; does not replace edit's consumed data.
    before_fit = _edit(store, history)  # 5 supersedes 3, but a later fit explicitly pins 3.
    pins = {
        "model": history.state(edit).current["model"].revision,
        "panel": history.state(prepared).current["panel"].revision,
    }
    boundary = _fit(store, history, pins=pins)  # 6
    assert _kept_seqs(plan_squash(source, at=before_fit if fit_in_suffix else boundary)) == (
        [2, 3, 4, 5, 6] if fit_in_suffix else [2, 3, 4, 6]
    )


def test_reused_report_does_not_pull_its_old_model_producer_and_retractions_are_last_writes(study):
    source, destination, store, history = study
    first = _edit(store, history)  # 1
    old_model = history.state(first).current["model"]
    report = store.write_artifact(
        "identification_report",
        produced_by="check:model",
        derived_from={"model": old_model.revision},
        json_files={"report.json": {}},
    )
    _append(history, produced=[report])  # 2
    current = _model(store, history)
    _append(history, produced=[current, report], checks=_checks(3))  # 3 re-emits the report.
    retract = RetractedArtifact(artifact_id="validation_report", reason_ref="saved finding")
    _append(history, retracted=[retract])  # 4 redundant absence still counts as a write.
    boundary = _simulate(history)  # 5
    assert _kept_seqs(plan_squash(source, at=boundary)) == [3, 4, 5]
    mapping = copy_squashed(plan_squash(source, at=boundary), destination)
    squashed = StudyRepository("study", repository_path=destination / "study/history.git")
    rewritten = mapping[boundary]
    assert rewritten is not None
    assert history.state(boundary) == squashed.state(rewritten)
    assert mapping[first] is None


@pytest.mark.parametrize(
    "unsupported",
    ["format", "legacy", "report_only", "branch", "deleted_branch", "replicate", "missing_input"],
)
def test_refusals_happen_before_destination_creation(study, unsupported):
    source, destination, store, history = study
    boundary = _edit(store, history)
    if unsupported == "format":
        history.repo.config["nof1.format"] = 5
    elif unsupported == "legacy":
        _append(history, diagnostics={"prior_predictive": {"samples": {}, "diagnostics": []}})
    elif unsupported == "report_only":
        _append(history, diagnostics={"retention": "report_only"})
    elif unsupported in {"branch", "deleted_branch"}:
        history.create_branch("other", at=boundary)
        if unsupported == "deleted_branch":
            _append(history, branch="other")
            history.repo.references.delete("refs/heads/other")
    elif unsupported == "replicate":
        # Even an unselected catalog panel survives the copy and must be refused.
        store.write_artifact(
            "panel",
            derived_from={},
            produced_by="prepare_data",
            json_files={"metadata.json": {"source": {"revision": boundary, "replicate": 0}}},
        )
    elif unsupported == "missing_input":
        uncommitted = _model(store, history)
        _simulate(history, model_revision=uncommitted.revision)
    with pytest.raises(
        ValueError, match=r"Migrate|Legacy|branch|simulation replicate|no recorded producer"
    ):
        copy_squashed(plan_squash(source, at=boundary), destination)
    assert not destination.exists()


def test_boundary_must_be_applied_and_destination_must_be_new(study):
    source, destination, store, history = study
    boundary = _edit(store, history)
    failure = _append(history, status="raised")
    with pytest.raises(ValueError, match="applied action"):
        copy_squashed(plan_squash(source, at=failure), destination)
    with pytest.raises(ValueError, match="new destination"):
        copy_squashed(plan_squash(source, at=boundary), source)
    with pytest.raises(ValueError, match="new destination"):
        copy_squashed(plan_squash(source, at=boundary), source / "nested" / "study")
    assert not destination.exists()
