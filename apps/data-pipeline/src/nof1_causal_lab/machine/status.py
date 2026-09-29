"""Public episode state and action outcomes shared by the workflow and HTTP facade."""

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.actions.results import RunningAction
from nof1_causal_lab.artifacts.identity import GitOid, ScientificActionId
from nof1_causal_lab.json_types import JsonObject

from .artifacts import ArtifactRecord, EpisodeState
from .execution import ArtifactFreshness, RetractedArtifact
from .store import JournalStatus


class ActionOutcome(BaseModel):
    """An action outcome reports the attempted transition and resulting committed state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    branch: str = "main"
    commit_id: GitOid | None = None
    seq: int
    status: JournalStatus
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    diagnostics: JsonObject = Field(default_factory=dict)
    produced: list[ArtifactRecord] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    state: EpisodeState


class EpisodeStatus(BaseModel):
    """Episode status reports committed artifacts, their freshness, actions, and any running one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    branch: str = "main"
    commit_id: GitOid | None = None
    seq: int
    state: EpisodeState
    artifacts: list[ArtifactFreshness]
    actions: list[ScientificActionId]
    running: RunningAction | None
