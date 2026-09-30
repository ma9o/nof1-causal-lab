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
from typing import TYPE_CHECKING

import pygit2
from pydantic import TypeAdapter

from nof1_causal_lab.actions.predictive_checks import fitted_law_report, law_provenance
from nof1_causal_lab.artifacts.data_preparation import DataSourceRef, SimulationReplicateRef
from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.utils.arrays import read_array

if TYPE_CHECKING:
    from nof1_causal_lab.machine.history_models import StudyRevision

_PRIMARY = {"model", "panel", "data_profile", "raw_data"}
_PINS = TypeAdapter(dict[ArtifactId, GitOid])


@dataclass(frozen=True)
class HistorySquashPlan:
    source: Path
    branch: str
    root: GitOid
    records: tuple[StudyRevision, ...]
    kept: frozenset[GitOid]


def plan_squash(source: Path, *, at: str) -> HistorySquashPlan:
    """Validate the supported scope and close last writers over scientific inputs."""
    source = source.resolve()
    path = source / "episode/history.git"
    if not path.is_dir():
        raise ValueError(f"No study repository at {path}")
    history = StudyRepository(source.name, repository_path=path)
    repo = history.repo
    branches = history.branches()
    if len(branches) != 1 or any(ref.startswith("refs/remotes/") for ref in repo.references):
        raise ValueError("Squashing requires exactly one study branch")
    branch, head = next(iter(branches.items()))
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
        if record.branch != branch or record.parent_ids != [current]:
            raise ValueError(f"Attempt {record.seq} is outside the single sequential branch")
        if (
            record.operation_id == "statistical_model_spec"
            or record.diagnostics.get("retention") == "report_only"
        ):
            raise ValueError(f"Legacy action at attempt {record.seq} is unsupported")
        if record.status == "applied":
            current = record.commit_id
    recorded_commits = {r.commit_id for r in records}
    if [r.commit_id for r in records if r.status == "applied"] != [
        r.commit_id for r in ancestry
    ] or any(
        str(repo.references[ref].target) not in recorded_commits
        for ref in repo.references
        if ref.startswith("refs/actions/")
    ):
        raise ValueError("Study actions must all belong to the single recorded branch")

    store = ArtifactStore(source.name, repository_path=path)
    # The catalog survives intact, including panels from actions we might drop.
    for revision in store.list_revisions("panel"):
        metadata = store.read_json_file("panel", revision, "metadata.json")
        if isinstance(
            TypeAdapter(DataSourceRef).validate_python(metadata["source"]), SimulationReplicateRef
        ):
            raise ValueError("Panels prepared from a simulation replicate are unsupported")

    producers: dict[tuple[ArtifactId, GitOid], StudyRevision] = {}
    for record in ancestry:
        for artifact in record.produced:
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
        return law_provenance(
            store, store.read_meta("model", revision), model, None
        ).fitted_model_revision

    def producer(
        artifact_id: ArtifactId, revision: GitOid, consumer: StudyRevision
    ) -> StudyRevision:
        found = producers.get((artifact_id, revision))
        if found is None or found.seq > consumer.seq:
            raise ValueError(
                f"Attempt {consumer.seq}: {artifact_id} {revision} has no recorded producer"
            )
        return found

    def dependencies(record: StudyRevision) -> set[GitOid]:
        pins: set[tuple[ArtifactId, GitOid]] = set()
        if record.action in {"fit", "simulate", "prepare_data"}:
            pins.update(_PINS.validate_python(record.diagnostics["input_pins"]).items())
        if record.action == "edit_model":
            parent = history.state(record.parent_ids[0])
            pins.update(
                (key, parent.current[key].revision)
                for key in ("panel", "data_profile")
                if parent.has(key)
            )
        for artifact in record.produced:
            if artifact.artifact_id in _PRIMARY - {"model"}:
                pins.update(artifact.derived_from.items())
        result = {producer(key, revision, record).commit_id for key, revision in pins}
        models = {revision for key, revision in pins if key == "model"}
        models.update(item.revision for item in record.produced if item.artifact_id == "model")
        for revision in models:
            owner = fitted_owner(revision)
            if owner is not None:
                fit = producer("model", owner, record)
                fitted_law_report([fit], owner)
                result.add(fit.commit_id)
        return result

    boundary_record = history.record(boundary)
    prefix = [r for r in ancestry if r.seq <= boundary_record.seq]
    last_writers: dict[str, GitOid] = {}
    for record in prefix:
        for artifact in (*record.retracted, *record.produced):
            last_writers[artifact.artifact_id] = record.commit_id
        if record.checks is not None:
            last_writers["checks"] = record.commit_id
    kept = {root, boundary, *last_writers.values()}
    kept.update(r.commit_id for r in records if r.seq > boundary_record.seq)
    latest_simulation = next((r for r in reversed(prefix) if r.operation_id == "simulate"), None)
    model = history.state(boundary).get("model")
    if latest_simulation is not None and model is not None:
        report = SimulationReport.model_validate(latest_simulation.diagnostics["report"])
        if report.model.revision == model.revision:
            kept.add(latest_simulation.commit_id)

    by_commit = {r.commit_id: r for r in records}
    pending = list(kept - {root})
    while pending:
        record = by_commit[pending.pop()]
        if record.status != "applied":
            continue
        for dependency in dependencies(record) - kept:
            kept.add(dependency)
            pending.append(dependency)
    return HistorySquashPlan(source, branch, root, tuple(records), frozenset(kept))


def _copy_squashed(plan: HistorySquashPlan, destination: Path) -> dict[str, str | None]:
    destination = destination.resolve()
    if destination.exists() or destination.is_relative_to(plan.source):
        raise ValueError("Choose a new destination outside the source workspace")
    if destination.name != plan.source.name:
        raise ValueError("Keep the study directory name (its logical workspace ID) unchanged")
    shutil.copytree(plan.source, destination, ignore=shutil.ignore_patterns("cache", "scratch"))
    repo = pygit2.Repository(str(destination / "episode/history.git"))
    for ref in list(repo.references):
        if not ref.startswith("refs/artifacts/"):
            repo.references.delete(ref)
    mapping: dict[str, str | None] = {plan.root: plan.root}
    head = plan.root
    for record in plan.records:
        if record.commit_id not in plan.kept:
            mapping[record.commit_id] = None
            continue
        parent = head if record.status == "applied" else mapping[record.parent_ids[0]]
        assert parent is not None
        previous = repo[parent].peel(pygit2.Commit).tree
        original = repo[record.commit_id].peel(pygit2.Commit)
        artifacts = (
            repo.TreeBuilder(previous["artifacts"].id)
            if "artifacts" in previous
            else repo.TreeBuilder()
        )
        if record.status == "applied":
            for artifact in record.retracted:
                if artifacts.get(artifact.artifact_id) is not None:
                    artifacts.remove(artifact.artifact_id)
            for artifact in record.produced:
                artifacts.insert(
                    artifact.artifact_id,
                    pygit2.Oid(hex=artifact.revision),
                    pygit2.GIT_FILEMODE_TREE,
                )
        tree = repo.TreeBuilder()
        tree.insert("artifacts", artifacts.write(), pygit2.GIT_FILEMODE_TREE)
        checks = (
            original.tree if record.status == "applied" and record.checks is not None else previous
        )
        if "checks.json" in checks:
            tree.insert("checks.json", checks["checks.json"].id, pygit2.GIT_FILEMODE_BLOB)
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
        repo.references.create(f"refs/attempts/{record.seq}", rewritten)
        if record.attempt_id is not None:
            repo.references.create(f"refs/actions/{record.attempt_id}", rewritten)
        if record.status == "applied":
            head = GitOid(str(rewritten))
    repo.references.create(f"refs/heads/{plan.branch}", pygit2.Oid(hex=head))
    repo.set_head(f"refs/heads/{plan.branch}")
    (destination / "squash-mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    return mapping


def squash_study(source: Path, destination: Path, *, at: str) -> dict[str, str | None]:
    """Apply the deterministic retention rule to a new copy; leave the source untouched."""
    return _copy_squashed(plan_squash(source, at=at), destination)


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
            f"{disposition:4} {record.seq:4} {record.action:12} {record.status:8} {record.commit_id}"
        )
    if not args.dry_run:
        _copy_squashed(plan, args.destination)
        print(f"Squashed into {args.destination}; source unchanged; see squash-mapping.json")


if __name__ == "__main__":
    main()
