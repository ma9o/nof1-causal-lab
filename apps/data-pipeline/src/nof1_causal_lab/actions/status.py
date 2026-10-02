"""Study status shared by the HTTP facade: committed artifacts, their freshness and any running attempt."""

from nof1_causal_lab.actions.results import RunningAction
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import GitOid, ScientificActionId
from nof1_causal_lab.study.state import ArtifactFreshness, StudyState


class StudyStatus(Value):
    """Study status reports committed artifacts, their freshness, actions, and any running one."""

    workspace_id: str
    branch: str = "main"
    commit_id: GitOid | None = None
    seq: int
    state: StudyState
    artifacts: tuple[ArtifactFreshness, ...]
    actions: tuple[ScientificActionId, ...]
    running: RunningAction | None
