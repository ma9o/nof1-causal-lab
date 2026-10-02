"""Stage one explicit model edit before action-owned checks and atomic publication."""

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.study.records import ModelEditResult
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.study.writes import write_model_revision


def edit_model(workspace_id: str, request: EditModelRequest, state: StudyState) -> ModelEditResult:
    store = ArtifactStore(workspace_id)
    info = write_model_revision(
        store,
        state,
        request.model,
        expected_model_revision=request.expected_revision,
        derived_from={"model": request.expected_revision} if request.expected_revision else {},
        produced_by="edit_model",
    )
    return ModelEditResult(
        produced=(info,),
        base=GitRef(
            workspace_id=workspace_id, revision=request.expected_revision, path="model.json"
        )
        if request.expected_revision is not None
        else None,
    )
