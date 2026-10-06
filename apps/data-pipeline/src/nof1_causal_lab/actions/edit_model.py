"""Stage one explicit model edit before action-owned checks and atomic publication."""

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.models.model_checks import question_edit_reason
from nof1_causal_lab.study.inputs import model_edit_state
from nof1_causal_lab.study.records import Applied, Rejected
from nof1_causal_lab.study.store import ArtifactStore, read_question
from nof1_causal_lab.study.writes import write_model_revision


def edit_model(workspace_id: str, request: EditModelRequest[GitOid]) -> Applied[None] | Rejected:
    """Save a replacement model, rejecting definitions inconsistent with its study question."""
    store = ArtifactStore(workspace_id)
    state = model_edit_state(store, request.input.parent_ref)
    question = read_question(store, state.current["question"].revision)
    if (reason := question_edit_reason(request.input.model, question)) is not None:
        return Rejected(reason="scientific_inputs", detail=reason)
    info = write_model_revision(
        store,
        request.input.model,
        derived_from={kind: record.revision for kind, record in state.current.items()},
        produced_by="edit_model",
    )
    return Applied(
        result=None,
        effects=ActionEffects(produced=(info,)),
    )
