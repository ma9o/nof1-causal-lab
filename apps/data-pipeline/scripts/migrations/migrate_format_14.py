"""Root stopped format-13 studies in their question and date their simulation designs.

Usage: uv run python -m scripts.migrations.migrate_format_14 SOURCE DESTINATION
Only the new destination is written. No numerical values are regenerated.

The question takes the models' text and the head model's default outcome, with no
queries, and is set by a grafted `set_question` root (sequence 0) under every lineage.
Simulation designs become a start date, a horizon and intervention offsets, using
each report's recorded origin; a design whose times are not whole dates and seconds
after that origin cannot be translated, and the conversion stops.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pygit2

from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.study.git_objects import write_tree
from scripts.migrations.study_rewrite import rewrite_study

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonValue

_UNITS = (("w", 604800), ("d", 86400), ("h", 3600), ("m", 60), ("s", 1))


def convert_model(value: JsonValue) -> JsonValue:
    """The question and its outcome belong to the study question, not the model."""
    if isinstance(value, Mapping) and {"edges", "distributions"} <= value.keys():
        return {
            key: item for key, item in value.items() if key not in {"question", "default_outcome"}
        }
    return value


def convert_payload(value: JsonValue) -> JsonValue:
    """Translate one retained JSON record throughout."""
    if isinstance(value, (list, tuple)):
        return [convert_payload(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    converted = {key: convert_payload(item) for key, item in value.items()}
    converted = convert_model(converted)
    assert isinstance(converted, dict)
    if converted.get("action") == "simulate" and "outcome" in converted:
        return _dated_attempt(converted)
    if {"input_key", "model_revision", "law", "draws"} <= converted.keys():
        # The predictive battery always spans its bound panel.
        converted.pop("design", None)
    return converted


def _duration(seconds: float) -> str:
    whole = round(seconds)
    if whole <= 0 or abs(whole - seconds) > 1e-6:
        raise ValueError(f"{seconds} seconds is not a positive whole duration")
    unit, size = next((unit, size) for unit, size in _UNITS if whole % size == 0)
    return f"{whole // size}{unit}"


def _dated_attempt(attempt: dict[str, JsonValue]) -> dict[str, JsonValue]:
    outcome = attempt["outcome"]
    if not isinstance(outcome, Mapping) or outcome.get("status") != "applied":
        raise ValueError("A failed simulation keeps no recorded origin to date its request")
    result = outcome["result"]
    assert isinstance(result, Mapping)
    report = result["report"]
    assert isinstance(report, Mapping)
    dated = dated_report(report)
    design = dated["design"]
    assert isinstance(design, Mapping)
    request = attempt["request"]
    assert isinstance(request, Mapping)
    return {
        **attempt,
        "request": {
            **{key: item for key, item in request.items() if key not in {"start", "end"}},
            **design,
        },
        "outcome": {**outcome, "result": {**result, "report": dated}},
    }


def dated_report(report: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """A report's design as a start date, horizon and offsets after its recorded origin."""
    if report.get("time_origin") is None:
        raise ValueError("A calendar-free simulation has no origin to date its design")
    origin = datetime.fromisoformat(str(report["time_origin"]))
    times = report["times"]
    assert isinstance(times, list)
    design = report["design"]
    assert isinstance(design, Mapping)
    start_day, end_day = float(str(times[0])), float(str(design["end"]))
    start = origin + timedelta(days=start_day)
    if start.astimezone(UTC).time() != datetime.min.time():
        raise ValueError(f"The simulation start {start.isoformat()} is not a whole date")
    events = design["interventions"]
    assert isinstance(events, list)
    spec = SimulationSpec.model_validate(
        {
            "start": start.astimezone(UTC).date().isoformat(),
            "horizon": _duration((end_day - start_day) * 86400),
            "interventions": [
                {
                    "target": event["target"],
                    "after": None
                    if float(str(event["time"])) == start_day
                    else _duration((float(str(event["time"])) - start_day) * 86400),
                    "value": event["value"],
                }
                for event in events
                if isinstance(event, Mapping)
            ],
        }
    )
    assignments = spec.assignments(origin)
    if (spec.start_day(origin), spec.end_day(origin)) != (start_day, end_day) or any(
        abs(placed.time - float(str(event["time"]))) > 1e-9
        for placed, event in zip(assignments, events, strict=True)
        if isinstance(event, Mapping)
    ):
        raise ValueError("The dated design does not reproduce the recorded model times")
    return {
        **report,
        "design": spec.model_dump(mode="json"),
        "assignments": [item.model_dump(mode="json") for item in assignments],
    }


def study_question(source: Path) -> QuestionSpec:
    """The single question text the models carried, with the head model's outcome."""
    repo = pygit2.Repository(str(source / "study/history.git"))
    texts: list[str] = []
    for name in repo.references:
        if name.startswith("refs/artifacts/model/"):
            tree = repo.references[name].peel(pygit2.Tree)
            model = json.loads(tree["model.json"].peel(pygit2.Blob).data)
            if model.get("question"):
                texts.append(str(model["question"]).strip())
    if len(set(texts)) != 1:
        raise ValueError(f"Expected one question text across models, found {sorted(set(texts))}")
    head = repo.references["refs/heads/main"].peel(pygit2.Commit).tree
    outcome = (
        json.loads(head["artifacts/model/model.json"].peel(pygit2.Blob).data).get("default_outcome")
        if "artifacts/model" in head
        else None
    )
    return QuestionSpec(text=texts[0], outcome=outcome)


def graft_question(repo: pygit2.Repository, question: QuestionSpec) -> dict[str, str]:
    """Insert a set_question root under every lineage, with the question in every tree."""
    commits = {
        str(commit.id): commit
        for name in repo.references
        if name.startswith(("refs/heads/", "refs/attempts/", "refs/actions/"))
        for commit in repo.walk(repo.references[name].target)
    }
    (root,) = (commit for commit in commits.values() if not commit.parents)
    first = min(
        (
            json.loads(commit.tree["logs/attempt.json"].peel(pygit2.Blob).data)
            for commit in commits.values()
            if "logs" in commit.tree
        ),
        key=lambda record: record["seq"],
    )
    metadata = {
        "artifact_id": "question",
        "derived_from": {},
        "produced_by": "set_question",
        "created_at": first["ts"],
        "model_inputs": {},
        "consumed_model_inputs": {},
    }
    payload = json.dumps(question.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    artifact = write_tree(
        repo,
        {
            "question.json": payload.encode(),
            "meta.json": json.dumps(metadata, sort_keys=True).encode(),
        },
    )
    repo.references.create(f"refs/artifacts/question/{artifact}", artifact, force=True)
    record = {
        "seq": 0,
        "ts": first["ts"],
        "attempt_id": None,
        "branch": "main",
        "messages": [],
        "trace_ids": [],
        "attempt": {
            "action": "set_question",
            "request": {"action": "set_question", "question": question.model_dump(mode="json")},
            "outcome": {
                "status": "applied",
                "result": {
                    "action": "set_question",
                    "produced": [{**metadata, "revision": str(artifact)}],
                    "retracted": [],
                    "checks": None,
                },
            },
        },
    }
    signature = pygit2.Signature(
        "nof1-causal-lab", "study@local", int(datetime.fromisoformat(first["ts"]).timestamp()), 0
    )
    lineage_root = repo.create_commit(
        None,
        signature,
        signature,
        "set_question (applied)",
        _with_question(repo, root.tree, artifact, record),
        [root.id],
    )
    mapping: dict[str, str] = {}

    def graft(commit: pygit2.Commit) -> pygit2.Oid:
        if str(commit.id) in mapping:
            return pygit2.Oid(hex=mapping[str(commit.id)])
        parents = [
            lineage_root if parent.id == root.id else graft(parent) for parent in commit.parents
        ]
        oid = repo.create_commit(
            None,
            commit.author,
            commit.committer,
            commit.message,
            _with_question(repo, commit.tree, artifact, None),
            parents,
        )
        mapping[str(commit.id)] = str(oid)
        return oid

    targets = {
        name: graft(repo.references[name].peel(pygit2.Commit))
        for name in repo.references
        if name.startswith(("refs/heads/", "refs/attempts/", "refs/actions/"))
    }
    with repo.transaction() as transaction:
        for name, oid in [*targets.items(), ("refs/attempts/0", lineage_root)]:
            transaction.lock_ref(name)
            transaction.set_target(name, oid)
    return mapping


def _with_question(
    repo: pygit2.Repository,
    tree: pygit2.Tree,
    artifact: pygit2.Oid,
    record: Mapping[str, JsonValue] | None,
) -> pygit2.Oid:
    builder = repo.TreeBuilder(tree)
    artifacts = (
        repo.TreeBuilder(tree["artifacts"].peel(pygit2.Tree))
        if "artifacts" in tree
        else repo.TreeBuilder()
    )
    artifacts.insert("question", artifact, pygit2.GIT_FILEMODE_TREE)
    builder.insert("artifacts", artifacts.write(), pygit2.GIT_FILEMODE_TREE)
    if record is not None:
        log = json.dumps(record, sort_keys=True).encode()
        builder.insert("logs", write_tree(repo, {"attempt.json": log}), pygit2.GIT_FILEMODE_TREE)
    return builder.write()


def convert_study(source: Path, destination: Path) -> dict[str, str]:
    question = study_question(source)
    mapping = rewrite_study(
        source,
        destination,
        convert_payload,
        mapping_name="format-14-revisions.json",
        source_format=13,
        target_format=14,
        preserve_model_meaning=True,
    )
    grafted = graft_question(pygit2.Repository(str(destination / "study/history.git")), question)
    mapping = {old: grafted.get(new, new) for old, new in mapping.items()}
    (destination / "format-14-revisions.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    mapping = convert_study(args.source, args.destination)
    print(f"Converted format 13 → 14: {len(mapping)} mapped Git objects")


if __name__ == "__main__":
    main()
