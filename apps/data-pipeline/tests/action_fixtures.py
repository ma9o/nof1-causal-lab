"""Execute model staging and action-owned checks in local contract tests."""

from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.io import EditQuestionInput
from nof1_causal_lab.actions.model_checks import evaluate_model_checks
from tests.helpers import fixture_entity_id


def edit_and_check(workspace_id, request, state):
    from nof1_causal_lab.study.records import Applied
    from nof1_causal_lab.study.store import ArtifactStore

    staged = edit_model(workspace_id, request)
    assert isinstance(staged, Applied)
    checks, identification, validation = evaluate_model_checks(
        workspace_id, state, staged, action="edit_model"
    )
    store = ArtifactStore(workspace_id)
    reports = {"checks": checks, "identification": identification}
    if validation is not None:
        reports["validation"] = validation
    return staged.revised(
        effects=staged.effects.revised(
            reports={name: store.write_report(report) for name, report in reports.items()}
        )
    )


def question_root(workspace_id, question=None):
    """Publish the study question as the lineage root, as edit_question does."""
    from nof1_causal_lab.actions.contracts import EditQuestionRequest
    from nof1_causal_lab.actions.edit_question import edit_question
    from nof1_causal_lab.artifacts.question import QuestionSpec
    from nof1_causal_lab.study.history import StudyRepository

    request = EditQuestionRequest(
        input=EditQuestionInput(
            question=question
            if question is not None
            else QuestionSpec(
                text="What does it answer?",
                outcome=fixture_entity_id("construct", "unmeasured_outcome"),
            )
        )
    )
    journal = StudyRepository(workspace_id)
    return journal.append(
        applied_record(
            edit_question(workspace_id, request), request=request, seq=journal.latest_seq() + 1
        )
    )


def applied_record(result, *, seq, request=None, ts="2026-01-01T00:00:00Z", **metadata):
    """A successful test publication composes the actual owned action payload."""
    from nof1_causal_lab.artifacts.posterior import ModelFitResult
    from nof1_causal_lab.artifacts.simulation import ModelSimulationResult
    from nof1_causal_lab.study.records import (
        AttemptRecord,
        DataPreparationResult,
        EditAttempt,
        EditQuestionAttempt,
        FitAttempt,
        PrepareAttempt,
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
                else EditQuestionAttempt(action="edit_question", request=request, outcome=result)
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
