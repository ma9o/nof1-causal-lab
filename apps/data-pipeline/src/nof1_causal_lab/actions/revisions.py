"""Read-only selection and comparison contracts for immutable scientific inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.actions.io import ModelDiffOutput
from nof1_causal_lab.artifacts.model_spec import ModelSpec

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import GitOid


def _model_revision(workspace_id: str, revision: GitOid) -> ModelSpec:
    """Read a saved spec; a checkpoint preceding model creation selects the empty spec."""
    import pygit2

    from nof1_causal_lab.study.errors import StudyLookupError
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import Applied
    from nof1_causal_lab.study.snapshots import ModelReader
    from nof1_causal_lab.study.store import ArtifactStore, read_model

    store = ArtifactStore(workspace_id)
    oid = pygit2.Oid(hex=revision)
    if oid not in store.repo:
        raise StudyLookupError(f"Unknown model revision {revision}")
    obj = store.repo[oid]
    if isinstance(obj, pygit2.Tree):
        return read_model(store, revision)
    if obj.type != pygit2.GIT_OBJECT_COMMIT:
        raise StudyLookupError("Select a model artifact tree or a study commit")
    repository = StudyRepository(workspace_id)
    if "logs" in obj.peel(pygit2.Commit).tree:
        selected = repository.record(revision)
        if not isinstance(selected.record.attempt.outcome, Applied):
            # Failure leaves record no scientific change; compare their exact execution parent.
            revision = selected.parent_ids[0]
    reader = ModelReader(workspace_id, at=revision)
    model = reader.model
    return ModelSpec.from_entities() if model is None else model


def model_diff(workspace_id: str, before_id: GitOid, after_id: GitOid) -> ModelDiffOutput:
    """Compare saved spec documents without compiling or evaluating scientific findings."""
    before = _model_revision(workspace_id, before_id)
    after = _model_revision(workspace_id, after_id)
    return ModelDiffOutput(changes=after.changes_from(before))


def read_model_diff(workspace_id: str, before_id: GitOid, after_id: GitOid) -> ModelDiffOutput:
    """Compute the comparison once during its owning action."""
    return model_diff(workspace_id, before_id, after_id)
