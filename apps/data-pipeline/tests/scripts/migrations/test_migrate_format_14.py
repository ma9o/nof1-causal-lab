"""Format 14 roots every lineage in its question and dates simulation designs."""

import json

import pygit2
import pytest
from scripts.migrations.migrate_format_14 import convert_payload, graft_question

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.json_types import JsonValue
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import (
    Applied,
    AttemptRecord,
    EditAttempt,
    Rejected,
)
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils import data
from tests.action_fixtures import applied_record
from tests.helpers import make_model

pytestmark = pytest.mark.contract


def _object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def test_payloads_move_the_question_off_models_and_date_simulations():
    model = make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")
    old_model = {**model, "question": "Why?", "default_outcome": "construct:y"}
    assert convert_payload({"model": old_model}) == {"model": model}
    report = {
        "time_origin": "2023-11-16T00:00:00Z",
        "times": [907.0, 928.0, 967.0],
        "design": {
            "start": None,
            "end": 967.0,
            "interventions": [
                {"target": "construct:dose", "time": 907.0, "value": 10.0},
                {"target": "construct:dose", "time": 928.0, "value": 5.0},
            ],
        },
    }
    attempt = _object(
        convert_payload(
            {
                "action": "simulate",
                "request": {"action": "simulate", "model_revision": "a" * 40, **report["design"]},
                "outcome": {"status": "applied", "result": {"report": report}},
            }
        )
    )
    design = {
        "start": "2026-05-11",
        "horizon": "60d",
        "interventions": [
            {"target": "construct:dose", "after": None, "value": 10.0},
            {"target": "construct:dose", "after": "3w", "value": 5.0},
        ],
    }
    assert attempt["request"] == {"action": "simulate", "model_revision": "a" * 40, **design}
    dated = _object(_object(_object(attempt["outcome"])["result"])["report"])
    assert dated["design"] == design
    assignments = dated["assignments"]
    assert isinstance(assignments, list)
    assert [_object(item)["time"] for item in assignments] == [907.0, 928.0]
    battery = {"input_key": "k", "model_revision": "a" * 40, "law": {}, "draws": 4, "design": {}}
    assert "design" not in _object(convert_payload(battery))
    with pytest.raises(ValueError, match="no recorded origin"):
        convert_payload({"action": "simulate", "request": {}, "outcome": {"status": "raised"}})


def test_question_roots_every_lineage_and_every_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    history = StudyRepository("STUDY")
    root = history.head()
    model = ArtifactStore("STUDY").write_artifact(
        "model", derived_from={}, produced_by="edit_model", json_files={"model.json": {}}
    )
    applied = history.append(
        applied_record(Applied(result=None, effects=ActionEffects(produced=(model,))), seq=1)
    )
    rejected = history.append(
        AttemptRecord(
            seq=2,
            ts="2026-01-01T00:00:00Z",
            attempt=EditAttempt(
                action="edit_model",
                request=EditModelRequest(expected_revision=None, model=ModelSpec()),
                outcome=Rejected(reason="revision_conflict", detail="Stale base"),
            ),
        )
    )
    mapping = graft_question(history.repo, QuestionSpec(text="Does X change Y?"))
    grafted = StudyRepository("STUDY")
    commits = [
        commit
        for commit in grafted.repo.walk(
            pygit2.Oid(hex=grafted.head()),
            pygit2.enums.SortMode.TOPOLOGICAL | pygit2.enums.SortMode.REVERSE,
        )
        if "logs" in commit.tree
    ]
    records = [
        json.loads(grafted.read_file(str(commit.id), "logs/attempt.json")) for commit in commits
    ]
    assert [(item["seq"], item["attempt"]["action"]) for item in records] == [
        (0, "set_question"),
        (1, "edit_model"),
    ]
    root_attempt = grafted.repo.references["refs/attempts/0"].peel(pygit2.Commit)
    assert tuple(str(oid) for oid in root_attempt.parent_ids) == (root,)
    assert commits[1].parent_ids == [root_attempt.id]
    question = grafted.state(str(root_attempt.id)).current["question"]
    for commit in (grafted.head(), mapping[rejected.commit_id]):
        assert grafted.state(commit).current["question"] == question
    assert (
        grafted.state(grafted.head()).current["model"]
        == history.state(applied.commit_id).current["model"]
    )
    retained = grafted.read_attempt(2)
    assert retained is not None
    assert retained.commit_id == mapping[rejected.commit_id]
    tree = root_attempt.tree
    assert "artifacts/question/question.json" in tree
