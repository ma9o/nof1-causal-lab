"""Retained question edits become model revisions without changing scientific values."""

import json

from scripts.migrate_model_question import fold_questions

from nof1_causal_lab.machine.moves import WriteArtifact, is_stale
from nof1_causal_lab.machine.snapshots import ModelReader
from nof1_causal_lab.machine.store import EpisodeJournal, derive_current_state
from tests.helpers import make_model


def test_question_edits_preserve_model_history_and_extraction_context(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    workspace = tmp_path / "QUESTIONS"
    graph = make_model(["X", "Y"], [("X", "Y")])
    original = graph.model_dump(mode="json", exclude={"question"})
    events = [
        ("question", 1, {}, {"text": "Does X change Y?"}),
        ("model", 1, {"question": 1}, original),
        ("panel", 1, {"model": 1, "question": 1}, None),
        ("question", 2, {}, {"text": "How does X change Y?"}),
        ("model", 2, {"model": 1}, original),
    ]
    for seq, (aid, version, pins, payload) in enumerate(events, 1):
        directory = workspace / "store" / aid / f"v{version}"
        directory.mkdir(parents=True)
        info = {
            "artifact_id": aid,
            "version": version,
            "derived_from": pins,
            "provenance": "human",
            "produced_by": None,
            "created_at": f"2026-09-14T12:00:0{seq}Z",
        }
        (directory / "meta.json").write_text(json.dumps(info))
        if payload is not None:
            (directory / f"{aid}.json").write_text(json.dumps(payload))
        journal = workspace / "episode/journal"
        journal.mkdir(parents=True, exist_ok=True)
        (journal / f"{seq:06d}.json").write_text(
            json.dumps(
                {
                    "seq": seq,
                    "ts": info["created_at"],
                    "move": {"kind": "run", "operation_id": "measurements"}
                    if aid == "panel"
                    else {"kind": "write", "artifact_id": aid, "provenance": "human"},
                    "status": "applied",
                    "produced": [info],
                    "retracted": [],
                    "diagnostics": {},
                    "trace_ids": [],
                    "resume": None,
                }
            )
        )

    revisions = fold_questions(workspace)
    assert revisions == {"question/v1": 1, "model/v1": 2, "question/v2": 3, "model/v2": 4}
    assert not (workspace / "store/question").exists()
    initial = ModelReader("QUESTIONS", at_seq=1).model
    assert initial is not None
    assert initial.question == "Does X change Y?"
    assert initial.edges == ()
    for seq in (2, 4, 5):
        model = ModelReader("QUESTIONS", at_seq=seq).model
        assert model is not None
        assert model.model_dump(mode="json", exclude={"question"}) == original
        assert model.question == ("Does X change Y?" if seq == 2 else "How does X change Y?")
    records = EpisodeJournal("QUESTIONS").read_all()
    assert records[-1].produced[0].derived_from == {"model": 3}
    assert isinstance(records[-1].move, WriteArtifact)
    assert records[-1].move.expected_model_version == 3
    assert records[2].produced[0].derived_from == {"model": 2}
    assert is_stale(derive_current_state("QUESTIONS"), "panel")
