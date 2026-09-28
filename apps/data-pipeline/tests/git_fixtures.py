"""Git identities for synthetic contracts and explicitly ordered test setup artifacts."""

from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore


def git_oid(label: int) -> GitOid:
    return GitOid(f"{label:040x}")


def artifact_revision(workspace: str, artifact: ArtifactId, ordinal: int) -> GitOid:
    return ArtifactStore(workspace).list_revisions(artifact)[ordinal - 1]


def commit_id(workspace: str, seq: int) -> GitOid:
    repository = StudyRepository(workspace)
    if seq == 0:
        head = repository.repo[repository.head()].peel(__import__("pygit2").Commit)
        while head.parents:
            head = head.parents[0]
        return GitOid(str(head.id))
    record = repository.read_attempt(seq)
    assert record is not None
    return record.commit_id
