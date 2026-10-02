"""Format conversions preserve the Git graph, scientific freshness and external bytes."""

import hashlib
import json
from pathlib import Path

import pygit2
import pytest
from scripts.migrations.migrate_format_11 import convert_attempt, convert_study
from scripts.migrations.migrate_format_12 import convert_payload
from scripts.migrations.migrate_format_12 import convert_study as convert_format_12
from scripts.migrations.migrate_format_13 import convert_payload as compose_payload
from scripts.migrations.migrate_format_13 import convert_study as convert_format_13

from nof1_causal_lab.artifacts.identity import GitOid, scientific_id
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact
from nof1_causal_lab.study.git_objects import read_file, write_tree
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.study.view_models import PanelRef
from nof1_causal_lab.utils import data

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    ("source_format", "converter"), [(11, convert_format_12), (12, convert_format_13)]
)
def test_representation_rewrite_preserves_fresh_and_stale_consumers_refs_and_bytes(
    tmp_path, monkeypatch, source_format, converter
):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path / "source"))
    store = ArtifactStore("study")
    history = StudyRepository("study")
    model = ModelSpec.model_validate_json(
        (Path(__file__).parents[2] / "fixtures/models/common/x_y_model.json").read_text()
    )
    owner = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    # Make the old owner's inference fingerprint differ from the new representation.
    old_meta = json.loads(read_file(store.repo, owner.revision, "meta.json"))
    old_meta["model_inputs"]["belief"] = "old-representation"
    from nof1_causal_lab.actions.model_checks import CHECK_POLICY_VERSION

    checks = {
        "input_keys": {
            "specification": scientific_id(
                "check",
                [
                    CHECK_POLICY_VERSION,
                    "specification",
                    [old_meta["model_inputs"]["compilation"], "old-representation"],
                ],
            ),
            "identification": "stale-check-key",
        },
    }
    old_owner = write_tree(
        store.repo,
        {
            "model.json": (
                Path(__file__).parent / f"fixtures/model_format_{source_format}.json"
            ).read_bytes(),
            "meta.json": json.dumps(old_meta).encode(),
        },
    )
    store.repo.references.create("refs/artifacts/model/" + str(old_owner), old_owner)
    fresh = store.write_artifact(
        "validation_report",
        derived_from={"model": GitOid(str(old_owner))},
        produced_by="check",
        json_files={"validation_report.json": {"indicators": {}, "dataset_issues": []}},
    )
    fresh_meta = json.loads(read_file(store.repo, fresh.revision, "meta.json"))
    fresh_meta["consumed_model_inputs"] = {"belief": "old-representation"}
    fresh_oid = write_tree(
        store.repo,
        {
            "meta.json": json.dumps(fresh_meta).encode(),
            "validation_report.json": b'{"indicators": {}, "dataset_issues": []}',
        },
    )
    stale_meta = {**fresh_meta, "consumed_model_inputs": {"belief": "already-stale"}}
    stale_oid = write_tree(
        store.repo,
        {
            "meta.json": json.dumps(stale_meta).encode(),
            "validation_report.json": b'{"indicators": {}, "dataset_issues": []}',
        },
    )
    signature = pygit2.Signature("scientist", "study@local", 123, 0)
    root = history.head()
    builder = store.repo.TreeBuilder()
    builder.insert("model", old_owner, pygit2.GIT_FILEMODE_TREE)
    builder.insert("validation_report", fresh_oid, pygit2.GIT_FILEMODE_TREE)
    artifacts = builder.write()
    builder = store.repo.TreeBuilder()
    builder.insert("artifacts", artifacts, pygit2.GIT_FILEMODE_TREE)
    builder.insert(
        "publication.json",
        store.repo.create_blob(json.dumps({**old_meta, "revision": str(old_owner)}).encode()),
        pygit2.GIT_FILEMODE_BLOB,
    )
    builder.insert(
        "checks.json", store.repo.create_blob(json.dumps(checks).encode()), pygit2.GIT_FILEMODE_BLOB
    )
    tree = builder.write()
    first = store.repo.create_commit(
        None, signature, signature, "scientific inputs", tree, [store.repo[root].id]
    )
    builder = store.repo.TreeBuilder()
    builder.insert("model", old_owner, pygit2.GIT_FILEMODE_TREE)
    builder.insert("validation_report", stale_oid, pygit2.GIT_FILEMODE_TREE)
    artifacts2 = builder.write()
    builder = store.repo.TreeBuilder()
    builder.insert("artifacts", artifacts2, pygit2.GIT_FILEMODE_TREE)
    tree2 = builder.write()
    second = store.repo.create_commit(None, signature, signature, "stale result", tree2, [first])
    store.repo.references.create("refs/heads/main", second, force=True)
    store.repo.references.create("refs/heads/review", first)
    store.repo.references.create("refs/attempts/00000001", second)
    store.repo.config["nof1.format"] = source_format
    arrays = tmp_path / "source/study/store/arrays"
    arrays.mkdir(parents=True)
    retained = arrays / "retained.bin"
    retained.write_bytes(bytes(range(256)) * 17)
    source_refs = {name: str(store.repo.references[name].target) for name in store.repo.references}
    before = hashlib.sha256(retained.read_bytes()).hexdigest()
    dest = tmp_path / "target/study"
    mapping = converter(tmp_path / "source/study", dest)
    converted = pygit2.Repository(str(dest / "study/history.git"))
    assert converted.config.get_int("nof1.format") == source_format + 1
    for name, oid in source_refs.items():
        target = (
            name.rsplit("/", 1)[0] + "/" + mapping[oid]
            if name.startswith("refs/artifacts/")
            else name
        )
        assert str(converted.references[target].target) == mapping[oid]
    converted_model = ModelSpec.model_validate(
        compose_payload(json.loads(read_file(converted, mapping[str(old_owner)], "model.json")))
    )
    assert converted_model.model_dump(mode="json") == model.model_dump(mode="json")
    if source_format == 12:
        report = ValidationReportArtifact.model_validate_json(
            read_file(converted, mapping[str(fresh_oid)], "validation_report.json")
        )
        assert report.data.is_valid
        assert report.is_valid
        assert report.preflight.findings == ()
    new_owner = json.loads(read_file(converted, mapping[str(old_owner)], "meta.json"))
    publication = json.loads(read_file(converted, mapping[str(first)], "publication.json"))
    assert publication["revision"] == mapping[str(old_owner)]
    assert publication["model_inputs"] == new_owner["model_inputs"]
    updated_checks = json.loads(read_file(converted, mapping[str(first)], "checks.json"))
    assert updated_checks["input_keys"] == {
        "specification": scientific_id(
            "check",
            [
                CHECK_POLICY_VERSION,
                "specification",
                [new_owner["model_inputs"]["compilation"], new_owner["model_inputs"]["belief"]],
            ],
        ),
        "identification": "stale-check-key",
    }
    new_fresh = json.loads(read_file(converted, mapping[str(fresh_oid)], "meta.json"))
    new_stale = json.loads(read_file(converted, mapping[str(stale_oid)], "meta.json"))
    assert new_fresh["consumed_model_inputs"]["belief"] == new_owner["model_inputs"]["belief"]
    assert new_stale["consumed_model_inputs"] == {"belief": "already-stale"}
    assert new_fresh["derived_from"]["model"] == mapping[str(old_owner)]
    for oid in [first, second]:
        original = store.repo[oid].peel(pygit2.Commit)
        updated = converted[mapping[str(oid)]].peel(pygit2.Commit)
        assert original.author == updated.author
        assert original.committer == updated.committer
        assert original.message == updated.message
        assert [mapping[str(p.id)] for p in original.parents] == [
            str(p.id) for p in updated.parents
        ]
    assert hashlib.sha256((dest / "store/arrays/retained.bin").read_bytes()).hexdigest() == before
    assert {
        name: str(store.repo.references[name].target) for name in store.repo.references
    } == source_refs
    assert store.repo.config.get_int("nof1.format") == source_format
    assert retained.read_bytes() == (dest / "store/arrays/retained.bin").read_bytes()


def _archived(action, *, inputs=None, diagnostics=None, **fields):
    return {
        "seq": 1,
        "branch": "main",
        "ts": "2026-10-01T00:00:00Z",
        "trace_ids": [],
        "action": action,
        "inputs": inputs or {},
        "diagnostics": diagnostics or {},
        "produced": [],
        "retracted": [],
        "status": "applied",
        **fields,
    }


def test_attempt_conversion_keeps_retained_reports_and_absence_without_inventing_requests():
    from nof1_causal_lab.study.records import Applied, FitAttempt
    from tests.inference_fixtures import _report

    model = ModelSpec.model_validate_json(
        (Path(__file__).parents[2] / "fixtures/models/common/x_y_model.json").read_text()
    )
    report = _report(model).model_dump(mode="json")
    report["detail"]["initial_latent_delta"] = [[1.0, 2.0], [3.0, 4.0]]
    old = _archived(
        "fit",
        diagnostics={
            "input_pins": {"model": "1" * 40, "panel": "2" * 40},
            "report": report,
            "retention": "report_only",
            "engine_evidence": report["engine"]["evidence"],
            "retained_axes": {"draws": 17},
            "foreign_pin": "measured",
        },
    )
    converted, retained = convert_attempt(old, "study")
    record = converted
    assert isinstance(record.attempt, FitAttempt)
    assert isinstance(record.attempt.outcome, Applied)
    assert record.attempt.request is None
    assert record.attempt_id is None
    result = record.attempt.outcome.result
    assert result.retention == "report_only"
    assert result.report.model_dump(mode="json") == report
    assert retained == {"measurements": {"retained_axes": {"draws": 17}, "foreign_pin": "measured"}}
    prepared, extra = convert_attempt(
        _archived(
            "prepare_data",
            diagnostics={
                "workers": [
                    {"worker_id": 3, "status": "completed", "n_windows": 4, "n_extractions": 3}
                ]
            },
        ),
        "study",
    )
    from nof1_causal_lab.study.records import PrepareAttempt

    assert isinstance(prepared.attempt, PrepareAttempt)
    assert prepared.attempt.outcome.status == "applied"
    worker = prepared.attempt.outcome.result.workers[0]
    assert worker.n_llm_calls is None
    assert worker.reused is None
    assert prepared.attempt.outcome.result.n_observations is None
    assert extra == {}
    # A historical complete request outside today's grammar remains an archival fact.
    proposal = {"question": "Retained", "unknown_field": "original authoring value"}
    edit, retained = convert_attempt(
        _archived("edit_model", inputs={"expected_revision": None, "model": proposal}), "study"
    )
    assert edit.attempt.request is None
    assert retained["request_fragment"] == {"expected_revision": None, "model": proposal}
    assert retained["request_unavailable_reason"] == "recorded_request_outside_current_schema"
    for status, fields in [
        ("rejected", {"reason": "Actual rejection"}),
        ("raised", {"error_type": "ActualError", "error_message": "Actual failure"}),
    ]:
        value, _ = convert_attempt(_archived("edit_model", status=status, **fields), "study")
        assert value.attempt.outcome.status == status
        assert "result" not in value.attempt.outcome.model_dump()


def test_comparison_conversion_has_one_report_owner_and_preserves_leaf_identity(
    tmp_path, monkeypatch
):
    from nof1_causal_lab.study.git_objects import write_tree
    from nof1_causal_lab.study.view_models import DataDiffReport

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path / "source"))
    history = StudyRepository("study")
    repo = history.repo
    head = history.head()
    ref = PanelRef(revision=head)
    report = DataDiffReport(left=(ref,), right=(ref,), variables=()).model_dump(mode="json")
    record = _archived(
        "data_diff",
        inputs={"left": ref.model_dump(mode="json"), "right": ref.model_dump(mode="json")},
    )
    sig = pygit2.Signature("scientist", "study@local", 123, 0)
    tree = write_tree(
        repo,
        {
            "logs/attempt.json": json.dumps(record).encode(),
            "logs/data-diff.json": json.dumps(report).encode(),
            "logs/notes.txt": b"original log",
        },
    )
    leaf = repo.create_commit(None, sig, sig, "comparison", tree, [repo[head].id])
    repo.references.create("refs/attempts/1", leaf)
    repo.config["nof1.format"] = 10
    refs = {name: str(repo.references[name].target) for name in repo.references}
    source = tmp_path / "source/study"
    destination = tmp_path / "target/study"
    mapping = convert_study(source, destination)
    converted = pygit2.Repository(str(destination / "study/history.git"))
    target = converted[mapping[str(leaf)]].peel(pygit2.Commit)
    assert "data-diff.json" not in target.tree["logs"].peel(pygit2.Tree)
    assert read_file(converted, mapping[str(leaf)], "logs/notes.txt") == b"original log"
    current = json.loads(read_file(converted, mapping[str(leaf)], "logs/attempt.json"))
    assert current["attempt"]["outcome"]["result"]["report"]["left"][0]["revision"] == mapping[head]
    assert str(converted.head.target) == mapping[head]
    assert {name: str(repo.references[name].target) for name in repo.references} == refs
    assert repo.config.get_int("nof1.format") == 10


def test_format_12_preserves_null_changes_and_independent_dispositions():
    from pydantic import TypeAdapter

    from nof1_causal_lab.json_types import JsonValue
    from nof1_causal_lab.study.view_models import Change

    added = convert_payload({"path": "/question", "change": "added", "before": None, "after": None})
    assert isinstance(added, dict)
    change = TypeAdapter(Change[JsonValue]).validate_python(added["change"])
    assert change.kind == "added"
    assert change.after is None
    assert "before" not in change.model_dump()
    before, after = {"name": "First"}, {"name": "Renamed"}
    before_disposition = {"disposition": "retained"}
    after_disposition = {"disposition": "excluded"}
    assert convert_payload(
        {
            "construct_id": "construct:x",
            "change": "unchanged",
            "before": before,
            "after": after,
            "before_disposition": before_disposition,
            "after_disposition": after_disposition,
        }
    ) == {
        "construct_id": "construct:x",
        "change": {"kind": "unchanged", "before": before, "after": after},
        "before_disposition": before_disposition,
        "after_disposition": after_disposition,
    }


def test_format_12_preserves_intervals_and_extraction_outcomes_without_impossible_fields():
    from pydantic import TypeAdapter

    from nof1_causal_lab.actions.temporal.messages import ExtractionChunkResult
    from nof1_causal_lab.artifacts.parameter_spec import IntervalEffectTransformSpec, ParameterSpec

    for interval in (None, 7.5):
        parameter = ParameterSpec.model_validate(
            convert_payload(
                {
                    "id": "parameter:" + "a" * 64,
                    "name": "effect",
                    "description": "interval effect",
                    "distribution_transform": "dt_effect_to_ct_rate",
                    "reference_interval_days": interval,
                }
            )
        )
        assert isinstance(parameter.transform, IntervalEffectTransformSpec)
        assert parameter.transform.interval_days == (
            "model_clock" if interval is None else interval
        )
    completed = convert_payload(
        {
            "worker_id": 2,
            "status": "completed",
            "n_windows": 1,
            "n_extractions": 1,
            "result_ref": "result.json",
            "error": None,
        }
    )
    failed = convert_payload(
        {
            "worker_id": 3,
            "status": "failed",
            "n_windows": 1,
            "n_extractions": 0,
            "result_ref": None,
            "error": "extraction failed",
        }
    )
    adapter = TypeAdapter(ExtractionChunkResult)
    success = adapter.validate_python(completed)
    failure = adapter.validate_python(failed)
    assert success.status == "completed"
    assert success.result_ref == "result.json"
    assert "error" not in success.model_dump()
    assert failure.status == "failed"
    assert failure.error == "extraction failed"
    assert "result_ref" not in failure.model_dump()
    from nof1_causal_lab.study.records import ExtractionWorkerResult

    retained = TypeAdapter(ExtractionWorkerResult).dump_python(success, mode="json")
    assert retained["status"] == "completed"
    assert "result_ref" not in retained
