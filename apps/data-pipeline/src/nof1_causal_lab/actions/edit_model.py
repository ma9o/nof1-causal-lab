"""Stage one explicit model edit before action-owned checks and atomic publication."""

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.execution import TransitionEffects
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.writes import write_model_revision


def edit_model(
    workspace_id: str, request: EditModelRequest, state: EpisodeState
) -> TransitionEffects:
    store = ArtifactStore(workspace_id)
    info = write_model_revision(
        store,
        state,
        request.model.model_dump(mode="json"),
        expected_model_revision=request.expected_revision,
        derived_from={"model": request.expected_revision} if request.expected_revision else {},
        produced_by="edit_model",
    )
    return TransitionEffects(produced=[info])
