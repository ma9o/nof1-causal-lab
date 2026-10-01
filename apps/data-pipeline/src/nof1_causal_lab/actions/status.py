"""Study status shared by the HTTP facade: committed artifacts, their freshness and any running attempt."""

from pydantic import BaseModel, ConfigDict

from nof1_causal_lab.actions.results import RunningAction
from nof1_causal_lab.artifacts.identity import GitOid, ScientificActionId
from nof1_causal_lab.study.state import ArtifactFreshness, StudyState


class StudyStatus(BaseModel):
    """Study status reports committed artifacts, their freshness, actions, and any running one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    branch: str = "main"
    commit_id: GitOid | None = None
    seq: int
    state: StudyState
    artifacts: list[ArtifactFreshness]
    actions: list[ScientificActionId]
    running: RunningAction | None
