"""Execute model staging and action-owned checks in local contract tests."""

from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.model_checks import evaluate_model_checks


def edit_and_check(workspace_id, request, state):
    staged = edit_model(workspace_id, request, state)
    evaluate_model_checks(workspace_id, state, staged, action="edit_model")
    return staged


def question_root(workspace_id, question=None):
    """Publish the study question as the lineage root, as set_question does."""
    from nof1_causal_lab.actions.contracts import SetQuestionRequest
    from nof1_causal_lab.actions.set_question import set_question
    from nof1_causal_lab.artifacts.question import QuestionSpec
    from nof1_causal_lab.study.history import StudyRepository

    request = SetQuestionRequest(
        question=question if question is not None else QuestionSpec(text="What does it answer?")
    )
    journal = StudyRepository(workspace_id)
    return journal.append(
        applied_record(
            set_question(workspace_id, request), request=request, seq=journal.latest_seq() + 1
        )
    )


def applied_record(result, *, seq, request=None, ts="2026-01-01T00:00:00Z", **metadata):
    """A successful test publication composes the actual owned action payload."""
    from nof1_causal_lab.study.records import (
        AttemptRecord,
        DataPreparationResult,
        EditAttempt,
        FitAttempt,
        ModelFitResult,
        ModelSimulationResult,
        PrepareAttempt,
        SetQuestionAttempt,
        SimulateAttempt,
    )

    if request is not None:
        from nof1_causal_lab.study.records import applied_attempt
        return AttemptRecord(seq=seq, ts=ts, attempt=applied_attempt(request, result), **metadata)
    match result.result:
        case None:
            attempt = (
                EditAttempt(action="edit_model", request=request, outcome=result)
                if any(artifact.artifact_id == "model" for artifact in result.effects.produced)
                else SetQuestionAttempt(action="set_question", request=request, outcome=result)
            )
        case DataPreparationResult():
            attempt = PrepareAttempt(action="prepare_data", request=request, outcome=result)
        case ModelFitResult():
            attempt = FitAttempt(action="fit", request=request, outcome=result)
        case ModelSimulationResult():
            attempt = SimulateAttempt(action="simulate", request=request, outcome=result)
        case _:
            raise TypeError("A publication fixture needs an owned action result")
    return AttemptRecord(seq=seq, ts=ts, attempt=attempt, **metadata)
