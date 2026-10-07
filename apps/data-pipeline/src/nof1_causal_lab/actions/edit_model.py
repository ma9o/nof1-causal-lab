"""Stage one explicit model edit before action-owned checks and atomic publication."""

from functools import cache

from pydantic import ValidationError

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model_spec import ModelEditResult, ModelSpec, apply_model_edit
from nof1_causal_lab.models.model_checks import question_edit_reason
from nof1_causal_lab.study.inputs import model_edit_state
from nof1_causal_lab.study.records import Applied, Rejected
from nof1_causal_lab.study.store import ArtifactStore, read_model, read_question
from nof1_causal_lab.study.writes import write_model_revision


def edit_model(
    workspace_id: str, request: EditModelRequest[GitOid]
) -> Applied[ModelEditResult] | Rejected:
    """Merge into the selected parent, prune unrelated structure, and validate before writing."""
    store = ArtifactStore(workspace_id)
    state = model_edit_state(store, request.input.parent_ref)
    question = read_question(store, state.current["question"].revision)
    parent = state.get("model")
    base = read_model(store, parent.revision) if parent is not None else ModelSpec()
    try:
        edited = apply_model_edit(
            base,
            request.input.model,
            question.outcome,
            context={"distribution_array_loader": cache(store.read_array)},
        )
    except ValidationError as exc:
        return Rejected(reason="scientific_inputs", detail=str(exc))
    if (reason := question_edit_reason(edited.model, question)) is not None:
        return Rejected(reason="scientific_inputs", detail=reason)
    info = write_model_revision(
        store,
        edited.model,
        derived_from={kind: record.revision for kind, record in state.current.items()},
        produced_by="edit_model",
    )
    return Applied(
        result=edited.pruning,
        effects=ActionEffects(produced=(info,)),
    )
