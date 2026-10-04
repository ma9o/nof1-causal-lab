"""Stage one explicit model edit before action-owned checks and atomic publication."""

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.construct import Role
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.models.model_structure import StructuralSelection, StructuralSelectionError
from nof1_causal_lab.study.records import Applied, Rejected
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.study.writes import write_model_revision


def question_edit_reason(model: ModelSpec, question: QuestionSpec) -> str | None:
    """Why an edited model doesn't fit the question: it lacks the question's nodes, its outcome
    is not endogenous, or a check the outcome scopes (anchors, fitted joint laws) fails."""
    outcome = () if question.outcome is None else (question.outcome,)
    if missing := sorted(
        identity for identity in {*outcome, *question.targets} if identity not in model._constructs
    ):
        return "The model must define the question's constructs: " + ", ".join(missing)
    if (
        question.outcome is not None
        and model.get_construct(question.outcome).role != Role.ENDOGENOUS
    ):
        return "The question's outcome must reference an endogenous construct"
    try:
        StructuralSelection(model, question.outcome)
    except StructuralSelectionError as exc:
        return str(exc)
    return None


def edit_model(workspace_id: str, request: EditModelRequest, state: StudyState) -> Applied[None] | Rejected:
    store = ArtifactStore(workspace_id)
    from nof1_causal_lab.study.store import read_question

    question = read_question(store, state.current["question"].revision)
    if (reason := question_edit_reason(request.model, question)) is not None:
        return Rejected(reason="scientific_inputs", detail=reason)
    info = write_model_revision(
        store,
        request.model,
        derived_from={"question": state.current["question"].revision, **({"model": request.expected_revision} if request.expected_revision is not None else {})},
        produced_by="edit_model",
    )
    return Applied(
        result=None,
        effects=ActionEffects(produced=(info,)),
    )
