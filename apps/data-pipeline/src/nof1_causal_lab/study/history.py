"""Git trees select artifacts; commits own action logs, ancestry and branch heads."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.study.records import AttemptRecord

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID

import pygit2

from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.git_objects import open_repository, read_file, write_tree
from nof1_causal_lab.study.records import Applied, StudyRevision
from nof1_causal_lab.study.state import ArtifactRecord, StudyState


class BranchConflict(Exception):
    """The branch moved after the action selected its execution base."""


class StudyRepository:
    """One bare repository per local study; Git owns revision identity and topology."""

    def __init__(self, workspace_id: str, *, repository_path: Path | None = None) -> None:
        self.workspace_id = workspace_id
        self.repo = open_repository(workspace_id, repository_path)
        self.path = Path(self.repo.path)

    @staticmethod
    def _branch_ref(branch: str) -> str:
        ref = f"refs/heads/{branch}"
        if not branch or not pygit2.reference_is_valid_name(ref):
            raise StudyLookupError(f"Invalid branch name: {branch!r}")
        return ref

    def head(self, branch: str = "main") -> GitOid:
        ref = self._branch_ref(branch)
        if ref not in self.repo.references:
            raise StudyLookupError(f"Unknown study branch: {branch}")
        return GitOid(str(self.repo.references[ref].target))

    def _commit(self, revision: str) -> pygit2.Commit:
        oid = pygit2.Oid(hex=revision)
        if oid not in self.repo:
            raise StudyLookupError(f"Unknown study revision: {revision}")
        obj = self.repo[oid]
        if obj.type != pygit2.GIT_OBJECT_COMMIT:
            raise StudyLookupError("Select a study commit")
        return obj.peel(pygit2.Commit)

    def read_file(self, revision: str, name: str) -> bytes:
        return read_file(self.repo, revision, name)

    def state(self, revision: str) -> StudyState:
        tree = self._commit(revision).tree
        if "artifacts" not in tree:
            return StudyState()
        current = {}
        artifacts = tree["artifacts"].peel(pygit2.Tree)
        for artifact in artifacts:
            info = ArtifactRecord(
                **json.loads(read_file(self.repo, str(artifact.id), "meta.json")),
                revision=GitOid(str(artifact.id)),
            )
            if info.artifact_id != artifact.name:
                raise StudyLookupError("Artifact tree does not match its scientific identity")
            current[info.artifact_id] = info
        checks = (
            ModelCheckReport.model_validate_json(read_file(self.repo, revision, "checks.json"))
            if "checks.json" in tree
            else None
        )
        return StudyState(current=current, checks=checks)

    def resolve(self, *, branch: str = "main", at: GitOid | None = None) -> GitOid:
        revision = at if at is not None else self.head(branch)
        commit = self._commit(revision)
        if "logs" in commit.tree and not isinstance(
            self.record(revision).record.attempt.outcome, Applied
        ):
            raise StudyLookupError(f"Commit {revision} is an unsuccessful attempt")
        return revision

    def record(self, revision: str) -> StudyRevision:
        commit = self._commit(revision)
        from nof1_causal_lab.study.records import AttemptRecord

        return StudyRevision(
            record=AttemptRecord.model_validate_json(
                read_file(self.repo, revision, "logs/attempt.json")
            ),
            commit_id=GitOid(str(commit.id)),
            parent_ids=tuple(GitOid(str(parent)) for parent in commit.parent_ids),
        )

    def records(self, revision: str) -> list[StudyRevision]:
        """Read only the selected Git ancestry, in parent-before-child order."""
        return [
            self.record(str(commit.id))
            for commit in self.repo.walk(
                pygit2.Oid(hex=revision),
                pygit2.enums.SortMode.TOPOLOGICAL | pygit2.enums.SortMode.REVERSE,
            )
            if "logs" in commit.tree
        ]

    def attempts(self) -> list[StudyRevision]:
        refs = (ref for ref in self.repo.references if ref.startswith("refs/attempts/"))
        return [
            self.record(str(self.repo.references[ref].target))
            for ref in sorted(refs, key=lambda ref: int(ref.rsplit("/", 1)[1]))
        ]

    def read_attempt(self, seq: int) -> StudyRevision | None:
        """Workflow retry lookup; attempt numbers are not scientific revision identities."""
        ref = f"refs/attempts/{seq}"
        return (
            self.record(str(self.repo.references[ref].target))
            if ref in self.repo.references
            else None
        )

    def latest_seq(self) -> int:
        return max(
            (
                int(ref.rsplit("/", 1)[1])
                for ref in self.repo.references
                if ref.startswith("refs/attempts/")
            ),
            default=0,
        )

    def dispatched_attempt(self, attempt_id: UUID) -> StudyRevision | None:
        """Resolve a dispatch receipt independently of the current branch head."""
        ref = f"refs/actions/{attempt_id}"
        return (
            self.record(str(self.repo.references[ref].target))
            if ref in self.repo.references
            else None
        )

    def branches(self) -> dict[str, GitOid]:
        return {name: self.head(name) for name in sorted(self.repo.branches.local)}

    def create_branch(self, name: str, *, at: GitOid) -> GitOid:
        revision = self.resolve(at=at)
        if (
            "logs" in self._commit(revision).tree
            and self.record(revision).record.attempt.action == "data_diff"
        ):
            raise StudyLookupError("A data comparison is a read-only leaf; branch from its parent")
        self.repo.create_reference(self._branch_ref(name), pygit2.Oid(hex=revision))
        return revision

    def append(
        self,
        record: AttemptRecord,
        *,
        expected_head: str | None = None,
        logs: dict[str, bytes] | None = None,
    ) -> StudyRevision:
        """Atomically publish the action's tree and advance only a successful branch."""
        outcome = record.attempt.outcome
        advances = isinstance(outcome, Applied) and record.attempt.action != "data_diff"
        attempt_ref = f"refs/attempts/{record.seq}"
        branch_ref = self._branch_ref(record.branch)
        log = record.model_dump(mode="json")
        with self.repo.transaction() as transaction:
            transaction.lock_ref(attempt_ref)
            action_ref = (
                f"refs/actions/{record.attempt_id}" if record.attempt_id is not None else None
            )
            if action_ref is not None:
                transaction.lock_ref(action_ref)
                if (
                    action_ref in self.repo.references
                    and self.record(str(self.repo.references[action_ref].target)).record.seq
                    != record.seq
                ):
                    raise FileExistsError(
                        f"Action {record.attempt_id} already has a different attempt"
                    )
            existing = self.read_attempt(record.seq)
            if existing is not None:
                if existing.record.model_dump(mode="json") != log:
                    raise FileExistsError(f"Attempt {record.seq} already has different content")
                if expected_head is not None and existing.parent_ids != (expected_head,):
                    raise BranchConflict(f"Attempt {record.seq} has a different execution base")
                return existing
            transaction.lock_ref(branch_ref)
            head = self.head(record.branch)
            parent = expected_head if expected_head is not None else head
            if advances and parent != head:
                raise BranchConflict(f"Branch {record.branch} moved from {parent} to {head}")
            previous = self._commit(parent).tree
            artifacts = (
                self.repo.TreeBuilder(previous["artifacts"].id)
                if "artifacts" in previous
                else self.repo.TreeBuilder()
            )
            if advances and isinstance(outcome, Applied):
                for item in outcome.result.retracted:
                    if artifacts.get(item.artifact_id) is not None:  # pyright: ignore[reportUnnecessaryComparison] - pygit2 documents None for a missing entry, but its stub returns Object.
                        artifacts.remove(item.artifact_id)
                for item in outcome.result.produced:
                    artifacts.insert(
                        item.artifact_id, pygit2.Oid(hex=item.revision), pygit2.GIT_FILEMODE_TREE
                    )
            tree = self.repo.TreeBuilder()
            tree.insert("artifacts", artifacts.write(), pygit2.GIT_FILEMODE_TREE)
            if advances and isinstance(outcome, Applied) and outcome.result.checks is not None:
                tree.insert(
                    "checks.json",
                    self.repo.create_blob(outcome.result.checks.model_dump_json().encode()),
                    pygit2.GIT_FILEMODE_BLOB,
                )
            elif "checks.json" in previous:
                tree.insert("checks.json", previous["checks.json"].id, pygit2.GIT_FILEMODE_BLOB)
            log_files = {
                "attempt.json": json.dumps(log, sort_keys=True).encode(),
                **(logs or {}),
            }
            tree.insert("logs", write_tree(self.repo, log_files), pygit2.GIT_FILEMODE_TREE)
            timestamp = int(datetime.fromisoformat(record.ts).timestamp())
            signature = pygit2.Signature("nof1-causal-lab", "study@local", timestamp, 0)
            action = record.attempt.action
            oid = self.repo.create_commit(
                None,
                signature,
                signature,
                f"{action} ({outcome.status})",
                tree.write(),
                [pygit2.Oid(hex=parent)],
            )
            transaction.set_target(attempt_ref, oid)
            if action_ref is not None:
                transaction.set_target(action_ref, oid)
            if advances:
                transaction.set_target(branch_ref, oid, message=f"{action} ({outcome.status})")
        return StudyRevision(
            commit_id=GitOid(str(oid)), parent_ids=(GitOid(str(parent)),), record=record
        )
