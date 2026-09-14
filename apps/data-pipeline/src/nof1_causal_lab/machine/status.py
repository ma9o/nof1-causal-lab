"""Public episode state and move outcomes shared by the workflow and HTTP facade."""

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.identity import OperationId
from nof1_causal_lab.json_types import JsonObject

from .artifacts import ArtifactVersionInfo, EpisodeState
from .moves import ArtifactFreshness, Move, RetractedArtifact
from .store import JournalStatus


class MoveOutcome(BaseModel):
    """A move outcome reports the attempted transition and resulting committed state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seq: int
    status: JournalStatus
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    diagnostics: JsonObject = Field(default_factory=dict)
    produced: list[ArtifactVersionInfo] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    state: EpisodeState


class EpisodeStatus(BaseModel):
    """Episode status reports committed artifacts, their freshness, and available moves."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    state: EpisodeState
    artifacts: list[ArtifactFreshness]
    next_operation: OperationId | None = None
    legal: list[Move]
    auto_running: bool = False
