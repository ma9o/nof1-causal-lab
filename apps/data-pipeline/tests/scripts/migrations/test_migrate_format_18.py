"""Format 18 keeps scientific facts, original outcomes and comparison topology."""

import json

import pygit2
import pytest
from scripts.migrations.migrate_format_18 import convert_payload, migrate

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.study.git_objects import open_repository, read_file, write_tree
from nof1_causal_lab.study.records import AttemptRecord
from tests.inference_fixtures import _report
from tests.model_fixtures import load_model_fixture

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("retained_engine", [True, False])
def test_conversion_preserves_atoms_blobs_leaf_parents_and_original_messages(
    tmp_path, retained_engine
):
    source, destination = tmp_path / "original/STUDY", tmp_path / "converted/STUDY"
    repo = open_repository("STUDY", source / "study/history.git")
    repo.config["nof1.format"] = 17
    signature = pygit2.Signature("Scientist", "scientist@example.com", 1, 0)
    root = repo.head.target
    prior = load_model_fixture("causal_proofs/treatment_outcome.json")
    posterior = load_model_fixture("causal_proofs/conditioned_treatment_outcome.json")
    legacy_model = posterior.model_dump(mode="json")
    legacy_model.pop("law_layouts")
    layout_id, layout = next(iter(posterior.law_layouts.items()))

    def artifact(aid, files, pins=None):
        tree = write_tree(
            repo,
            {
                **files,
                "meta.json": json.dumps(
                    {
                        "artifact_id": aid,
                        "derived_from": pins or {},
                        "produced_by": "fit" if aid == "model" and pins else None,
                        "created_at": "2026-10-01T00:00:00Z",
                        "model_inputs": {},
                        "consumed_model_inputs": {},
                    }
                ).encode(),
            },
        )
        repo.references.create(f"refs/artifacts/{aid}/{tree}", tree)
        return str(tree)

    base = artifact("model", {"model.json": prior.model_dump_json(round_trip=True).encode()})
    panel = artifact(
        "panel",
        {
            "metadata.json": b'{"kind":"file","source":{"files":["x.csv"]},"preparation":{},"variables":[],"time_origin":null}',
            "external.json": b'{"panel.parquet":"retained-table"}',
        },
    )
    fitted = artifact(
        "model", {"model.json": json.dumps(legacy_model).encode()}, {"model": base, "panel": panel}
    )
    finding = artifact(
        "data_profile", {"data_profile.json": b'{"indicators":{}}'}, {"panel": panel}
    )
    native_report = _report(posterior).model_dump(mode="json", round_trip=True)
    native_report["core"]["inference_diagnostics"] = None
    if not retained_engine:
        native_report["core"]["engine"] = {
            "kind": "not_evaluated",
            "subject": "production_engine",
            "reason": "ARCHIVED_ENGINE_NOT_RETAINED",
            "detail": "Not retained",
        }
    element = next(iter(layout.labels))
    native_report["core"]["posterior_marginals"] = [
        {"subject": {"element_id": element}, "parameter": "Production label"}
    ]
    messages = [{"timestamp": "2026-10-01T00:00:00Z", "level": "info", "label": "ACTION_COMPLETED"}]

    def commit(seq, action, result, parent, produced=(), request=None, ref=None):
        infos = [
            {
                "artifact_id": aid,
                "revision": revision,
                "derived_from": {},
                "produced_by": None,
                "created_at": "",
                "model_inputs": {},
                "consumed_model_inputs": {},
            }
            for aid, revision in produced
        ]
        record = {
            "seq": seq,
            "attempt_id": None,
            "ts": "2026-10-01T00:00:00Z",
            "messages": messages,
            "trace_ids": ["trace"],
            "attempt": {
                "action": action,
                "request": request,
                "outcome": {
                    "status": "applied",
                    "result": result,
                    "effects": {"produced": infos, "retracted": [], "checks": {"input_keys": {}}},
                },
            },
        }
        files = {
            "logs/attempt.json": json.dumps(record).encode(),
            "checks.json": b'{"input_keys":{}}',
            "logs/traces/trace.json": b'{"model":"original","messages":[{"role":"assistant","content":"retained","tool_calls":[{"id":"call","type":"function","name":"tool","arguments":"{}"}]}]}',
        }
        tree = repo.TreeBuilder(write_tree(repo, files))
        artifacts = repo.TreeBuilder()
        for aid, revision in produced:
            artifacts.insert(aid, pygit2.Oid(hex=revision), pygit2.GIT_FILEMODE_TREE)
        tree.insert("artifacts", artifacts.write(), pygit2.GIT_FILEMODE_TREE)
        oid = repo.create_commit(ref, signature, signature, action, tree.write(), [parent])
        repo.references.create(f"refs/attempts/{seq:08d}", oid)
        return oid

    fit = commit(
        1,
        "fit",
        {
            "model": {"workspace_id": "STUDY", "revision": base, "path": "model.json"},
            "panel": {"workspace_id": "STUDY", "revision": panel, "path": "panel.parquet"},
            "retention": "joint",
            "report": native_report,
        },
        root,
        (("model", fitted), ("panel", panel), ("data_profile", finding)),
        ref="refs/heads/main",
    )
    archived = commit(
        2,
        "fit",
        {
            "model": {"workspace_id": "STUDY", "revision": base, "path": "model.json"},
            "panel": {"workspace_id": "STUDY", "revision": panel, "path": "panel.parquet"},
            "retention": "report_only",
            "report": native_report,
        },
        fit,
        ref="refs/heads/main",
    )
    request = {
        "action": "data_diff",
        "left": {"kind": "panel", "revision": str(fit)},
        "right": {"kind": "panel", "revision": str(archived)},
    }
    leaf = commit(
        3,
        "data_diff",
        {"action": "data_diff", "report": {"stored": True}},
        archived,
        request=request,
    )
    array, table = source / "store/arrays/retained-array", source / "store/blobs/retained-table"
    array.parent.mkdir(parents=True)
    table.parent.mkdir(parents=True)
    array.write_bytes(bytes(range(256)))
    table.write_bytes(b"exact panel bytes")
    before = {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    mapping = migrate(source, destination)
    converted = pygit2.Repository(str(destination / "study/history.git"))
    assert converted.config.get_int("nof1.format") == 18
    assert str(converted.head.target) == mapping[str(archived)]
    assert converted[mapping[str(leaf)]].peel(pygit2.Commit).parent_ids == [
        pygit2.Oid(hex=mapping[str(archived)])
    ]
    assert not any(name.startswith("refs/artifacts/data_profile/") for name in converted.references)
    raw = json.loads(read_file(converted, mapping[fitted], "model.json"))
    restored = ModelSpec.model_validate(raw)
    assert raw["distributions"] == legacy_model["distributions"]
    assert restored.law_layouts[layout_id].labels[element] == "Production label"
    assert restored.law_layouts[layout_id].construct_labels == layout.construct_labels
    assert "time_points" not in raw
    for old in (fit, archived, leaf):
        tree = converted[mapping[str(old)]].peel(pygit2.Commit).tree
        assert "checks.json" not in tree
        assert "data_profile" not in tree["artifacts"].peel(pygit2.Tree)
        record = AttemptRecord.model_validate_json(
            read_file(converted, mapping[str(old)], "logs/attempt.json")
        )
        assert [message.model_dump(mode="json") for message in record.messages] == messages
        assert "checks" not in record.attempt.outcome.effects.model_dump()
    actual = AttemptRecord.model_validate_json(
        read_file(converted, mapping[str(fit)], "logs/attempt.json")
    )
    assert actual.attempt.outcome.result.evidence.distribution == layout_id
    assert actual.attempt.outcome.result.evidence.num_chains is None
    assert (actual.attempt.outcome.result.evidence.engine is not None) == retained_engine
    assert actual.attempt.outcome.result.model.revision == mapping[base]
    for old in (archived, leaf):
        result = json.loads(read_file(converted, mapping[str(old)], "logs/attempt.json"))[
            "attempt"
        ]["outcome"]["result"]
        assert result is None
    retained_request = json.loads(read_file(converted, mapping[str(leaf)], "logs/attempt.json"))[
        "attempt"
    ]["request"]
    assert retained_request["left"]["revision"] == mapping[str(fit)]
    trace = json.loads(read_file(converted, mapping[str(fit)], "logs/traces/trace.json"))
    assert trace["messages"][0]["tool_calls"] == [{"id": "call", "name": "tool", "arguments": "{}"}]
    assert (destination / array.relative_to(source)).read_bytes() == array.read_bytes()
    assert (destination / table.relative_to(source)).read_bytes() == table.read_bytes()
    assert before == {
        path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }


def test_owner_specific_field_removals_preserve_authored_arguments_and_foreign_wire_tags():
    payload = {
        "normalized": {"id": "call", "type": "function", "name": "tool", "arguments": "{}"},
        "wire": {"id": "call", "type": "function", "function": {"name": "tool", "arguments": "{}"}},
        "authored": {"name": "method", "parameters": {"method": "my hypothesis", "mean": 2}},
        "worker": {"worker_id": 1, "status": "completed", "n_llm_calls": 8, "reused": True},
        "failure": {"worker_id": 2, "status": "failed", "n_llm_calls": 8, "reused": False},
        "sampler": {
            "settings": {},
            "parameter_kernel": "native",
            "latent_smoother": "dsmc",
            "latent_transition_kind": "euler_maruyama",
        },
        "prior": {
            "parameter": None,
            "code": "WARN",
            "suggested_adjustment": "Revise",
            "is_valid": True,
            "repair_scope": "old",
        },
        "point": {"value": 1, "probability": 1, "count": 3},
        "profile": {"n_obs": 3, "measurement_dtype": "continuous", "variance": 1, "mean": 2},
        "native": {"n_draws": 3, "state_ids": [], "parameter_shapes": {}, "latent_shape": []},
    }
    converted = convert_payload(payload)
    assert converted["normalized"] == {"id": "call", "name": "tool", "arguments": "{}"}
    assert converted["wire"] == payload["wire"]
    assert converted["authored"] == payload["authored"]
    assert converted["worker"] == {"worker_id": 1, "status": "completed"}
    assert converted["failure"] == payload["failure"]
    assert converted["sampler"] == {"settings": {}, "parameter_kernel": "native"}
    assert converted["prior"] == {
        "parameter": None,
        "code": "WARN",
        "suggested_adjustment": "Revise",
    }
    assert converted["point"] == {"value": 1, "probability": 1}
    assert converted["profile"] == {"n_obs": 3}
    assert converted["native"] == {"n_draws": 3, "state_ids": []}
