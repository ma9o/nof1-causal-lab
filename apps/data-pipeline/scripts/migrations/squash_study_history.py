"""Compact a stopped study's saved action effects into a new copy.

Usage: uv run python -m scripts.migrations.squash_study_history SOURCE DESTINATION --at REVISION

The rule keeps the smallest dependency closure of mandatory last writers, not
the globally fewest equivalent actions. Nothing is fitted, simulated or checked
again. Artifact refs and authored ancestry remain an archive, even for dropped
actions. Only current, single-branch histories without simulated panels qualify.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pygit2
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.data_preparation import DataSourceRef, SimulationReplicateRef
from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.predictive_provenance import FittedLawProvenance, MixedLawProvenance
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.lineage import law_provenance
from nof1_causal_lab.study.records import (
    Applied,
    DataPreparationResult,
    ModelFitResult,
    ModelSimulationResult,
    StudyRevision,
)
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils.arrays import read_array

_PRIMARY = {"question", "model", "panel", "raw_data"}


@dataclass(frozen=True)
class HistorySquashPlan:
    source: Path
    root: GitOid
    records: tuple[StudyRevision, ...]
    kept: frozenset[GitOid]


def plan_squash(source: Path, *, at: str) -> HistorySquashPlan:
    """Validate the supported scope and close last writers over scientific inputs."""
    source = source.resolve()
    path = source / "study/history.git"
    if not path.is_dir():
        raise ValueError(f"No study repository at {path}")
    history = StudyRepository(source.name, repository_path=path)
    repo = history.repo
    heads = {name for name in repo.references if name.startswith("refs/heads/")}
    if heads != {"refs/heads/main"} or any(ref.startswith("refs/remotes/") for ref in repo.references):
        raise ValueError("Squashing requires the single main study history")
    head = history.head()
    boundary = GitOid(str(repo.revparse_single(at).peel(pygit2.Commit).id))
    records = history.attempts()
    ancestry = history.records(head)
    if not ancestry or boundary not in {record.commit_id for record in ancestry}:
        raise ValueError("The squash boundary must be an applied action on the study branch")
    root = ancestry[0].parent_ids[0]
    root_commit = repo[root].peel(pygit2.Commit)
    if root_commit.parent_ids or len(root_commit.tree):
        raise ValueError("Expected an empty study root")
    # Also catches successful attempts on a deleted branch, and old failures
    # whose execution base cannot be remapped by this sequential-workflow rule.
    current = root
    for record in records:
        if record.parent_ids != (current,):
            raise ValueError(f"Attempt {record.record.seq} is outside the single sequential branch")
        outcome = record.record.attempt.outcome
        if isinstance(outcome, Applied) and record.record.attempt.action == "data_diff":
            raise ValueError("Recorded comparisons cannot be squashed")
        if record.record.attempt.outcome.status == "applied":
            current = record.commit_id
    recorded_commits = {r.commit_id for r in records}
    if [r.commit_id for r in records if r.record.attempt.outcome.status == "applied"] != [
        r.commit_id for r in ancestry
    ] or any(
        str(repo.references[ref].target) not in recorded_commits
        for ref in repo.references
        if ref.startswith("refs/actions/")
    ):
        raise ValueError("Study actions must all belong to the single recorded branch")

    store = ArtifactStore(source.name, repository_path=path)
    # The catalog survives intact, including panels from actions we might drop.
    for revision in (
        ref.rsplit("/", 1)[1] for ref in repo.references if ref.startswith("refs/artifacts/panel/")
    ):
        metadata = store.read_json_file("panel", revision, "metadata.json")
        if isinstance(
            TypeAdapter(DataSourceRef).validate_python(metadata["source"]), SimulationReplicateRef
        ):
            raise ValueError("Panels prepared from a simulation replicate are unsupported")

    producers: dict[tuple[ArtifactId, GitOid], StudyRevision] = {}
    for record in ancestry:
        outcome = record.record.attempt.outcome
        assert isinstance(outcome, Applied)
        for artifact in outcome.effects.produced:
            if artifact.artifact_id not in _PRIMARY:
                continue
            key = (artifact.artifact_id, artifact.revision)
            if key in producers:
                raise ValueError(f"Scientific input has multiple publishers: {key}")
            producers[key] = record

    @cache
    def fitted_owner(revision: GitOid) -> GitOid | None:
        model = ModelSpec.model_validate(
            store.read_json_file("model", revision, "model.json"),
            context={
                "distribution_array_loader": lambda ref: read_array(
                    str(source / "store/arrays"), ref
                )
            },
        )
        provenance = law_provenance(store, store.read_meta("model", revision), model, None)
        return (
            provenance.fitted_model_revision
            if isinstance(provenance, (FittedLawProvenance, MixedLawProvenance))
            else None
        )

    def producer(
        artifact_id: ArtifactId, revision: GitOid, consumer: StudyRevision
    ) -> StudyRevision:
        found = producers.get((artifact_id, revision))
        if found is None or found.record.seq > consumer.record.seq:
            raise ValueError(
                f"Attempt {consumer.record.seq}: {artifact_id} {revision} has no recorded producer"
            )
        return found

    def dependencies(record: StudyRevision) -> set[GitOid]:
        pins: set[tuple[ArtifactId, GitOid]] = set()
        outcome = record.record.attempt.outcome
        assert isinstance(outcome, Applied)
        effects = outcome.effects
        match outcome.result:
            case ModelFitResult(model=model, panel=panel):
                pins.update((("model", model.revision), ("panel", panel.revision)))
            case ModelSimulationResult(evidence=evidence):
                pins.add(("model", evidence.model.revision))
                if evidence.origin_panel_revision is not None:
                    pins.add(("panel", evidence.origin_panel_revision))
            case DataPreparationResult():
                pass
            case None:
                pass
            case _:
                raise ValueError("Recorded comparisons cannot be squashed")
        for artifact in effects.produced:
            if artifact.artifact_id == "model" and record.record.attempt.action == "edit_model":
                pins.update(
                    (key, revision)
                    for key, revision in artifact.derived_from.items()
                    if key != "model"
                )
            elif artifact.artifact_id in _PRIMARY - {"model"}:
                pins.update(artifact.derived_from.items())
        result = {producer(key, revision, record).commit_id for key, revision in pins}
        models = {revision for key, revision in pins if key == "model"}
        models.update(item.revision for item in effects.produced if item.artifact_id == "model")
        for revision in models:
            owner = fitted_owner(revision)
            if owner is not None:
                fit = producer("model", owner, record)
                if fit.record.attempt.outcome.result is None:
                    raise ValueError("A fitted-law owner requires retained numerical evidence")
                result.add(fit.commit_id)
        return result

    boundary_record = history.record(boundary)
    prefix = [r for r in ancestry if r.record.seq <= boundary_record.record.seq]
    last_writers: dict[str, GitOid] = {}
    for record in prefix:
        outcome = record.record.attempt.outcome
        assert isinstance(outcome, Applied)
        effects = outcome.effects
        for artifact in (*effects.retracted, *effects.produced):
            last_writers[artifact.artifact_id] = record.commit_id
    kept = {root, boundary, *last_writers.values()}
    kept.update(r.commit_id for r in records if r.record.seq > boundary_record.record.seq)
    latest_simulation = next(
        (r for r in reversed(prefix) if r.record.attempt.action == "simulate"), None
    )
    model = history.state(boundary).get("model")
    if latest_simulation is not None and model is not None:
        outcome = latest_simulation.record.attempt.outcome
        assert isinstance(outcome, Applied)
        assert isinstance(outcome.result, ModelSimulationResult)
        report = outcome.result.evidence
        if report.model.revision == model.revision:
            kept.add(latest_simulation.commit_id)

    by_commit = {r.commit_id: r for r in records}
    pending = list(kept - {root})
    while pending:
        record = by_commit[pending.pop()]
        if record.record.attempt.outcome.status != "applied":
            continue
        for dependency in dependencies(record) - kept:
            kept.add(dependency)
            pending.append(dependency)
    return HistorySquashPlan(source, root, tuple(records), frozenset(kept))


def copy_squashed(plan: HistorySquashPlan, destination: Path) -> dict[str, str | None]:
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(plan.source):
        raise ValueError("Choose a new destination outside the source workspace")
    if destination.name != plan.source.name:
        raise ValueError("Keep the study directory name (its logical workspace ID) unchanged")
    shutil.copytree(plan.source, destination, ignore=shutil.ignore_patterns("cache", "scratch"))
    repo = pygit2.Repository(str(destination / "study/history.git"))
    for ref in list(repo.references):
        if not ref.startswith("refs/artifacts/"):
            repo.references.delete(ref)
    mapping: dict[str, str | None] = {plan.root: plan.root}
    head = plan.root
    for record in plan.records:
        if record.commit_id not in plan.kept:
            mapping[record.commit_id] = None
            continue
        parent = (
            head
            if record.record.attempt.outcome.status == "applied"
            else mapping[record.parent_ids[0]]
        )
        assert parent is not None
        previous = repo[parent].peel(pygit2.Commit).tree
        original = repo[record.commit_id].peel(pygit2.Commit)
        artifacts = (
            repo.TreeBuilder(previous["artifacts"].id)
            if "artifacts" in previous
            else repo.TreeBuilder()
        )
        outcome = record.record.attempt.outcome
        if isinstance(outcome, Applied):
            effects = outcome.effects
            for artifact in effects.retracted:
                if artifacts.get(artifact.artifact_id) is not None:
                    artifacts.remove(artifact.artifact_id)
            for artifact in effects.produced:
                artifacts.insert(
                    artifact.artifact_id,
                    pygit2.Oid(hex=artifact.revision),
                    pygit2.GIT_FILEMODE_TREE,
                )
        tree = repo.TreeBuilder()
        tree.insert("artifacts", artifacts.write(), pygit2.GIT_FILEMODE_TREE)
        tree.insert("logs", original.tree["logs"].id, pygit2.GIT_FILEMODE_TREE)
        rewritten = repo.create_commit(
            None,
            original.author,
            original.committer,
            original.message,
            tree.write(),
            [pygit2.Oid(hex=parent)],
        )
        mapping[record.commit_id] = str(rewritten)
        repo.references.create(f"refs/attempts/{record.record.seq}", rewritten)
        if record.record.attempt_id is not None:
            repo.references.create(f"refs/actions/{record.record.attempt_id}", rewritten)
        if record.record.attempt.outcome.status == "applied":
            head = GitOid(str(rewritten))
    repo.references.create("refs/heads/main", pygit2.Oid(hex=head))
    repo.set_head("refs/heads/main")
    (destination / "squash-mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--at", required=True, help="Applied boundary commit (OID or Git revision)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show retained/dropped attempts without copying"
    )
    args = parser.parse_args()
    plan = plan_squash(args.source, at=args.at)
    for record in plan.records:
        disposition = "keep" if record.commit_id in plan.kept else "drop"
        print(
            f"{disposition:4} {record.record.seq:4} {record.record.attempt.action:12} {record.record.attempt.outcome.status:8} {record.commit_id}"
        )
    if not args.dry_run:
        copy_squashed(plan, args.destination)
        print(f"Squashed into {args.destination}; source unchanged; see squash-mapping.json")


if __name__ == "__main__":
    main()
