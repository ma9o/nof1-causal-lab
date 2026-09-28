"""Views of native Git commits and captured execution state."""

from pydantic import BaseModel, ConfigDict

from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.store import TransitionRecord


class StudyRevision(TransitionRecord):
    """One Git commit's parent links and its action log."""

    commit_id: GitOid
    parent_ids: list[GitOid]


class BranchBase(BaseModel):
    """Immutable execution base, captured before an action starts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commit_id: GitOid
    state: EpisodeState
    event_cursor: str | None = None
