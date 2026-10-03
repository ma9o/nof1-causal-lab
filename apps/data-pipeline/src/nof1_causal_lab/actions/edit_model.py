"""Stage one explicit model edit before action-owned checks and atomic publication."""

from nof1_causal_lab.actions.contracts import EditModelRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.construct import Role
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.models.model_structure import StructuralSelection, StructuralSelectionError
from nof1_causal_lab.study.records import Applied
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


def edit_model(workspace_id: str, request: EditModelRequest, state: StudyState) -> Applied[None]:
    store = ArtifactStore(workspace_id)
    info = write_model_revision(
        store,
        state,
        request.model,
        expected_model_revision=request.expected_revision,
        derived_from={"model": request.expected_revision} if request.expected_revision else {},
        produced_by="edit_model",
    )
    return Applied(
        result=None,
        effects=ActionEffects(produced=(info,)),
    )
