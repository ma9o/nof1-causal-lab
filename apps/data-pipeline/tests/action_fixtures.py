"""Execute model staging and action-owned checks in local contract tests."""

from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.model_checks import evaluate_model_checks


def edit_and_check(workspace_id, request, state):
    staged = edit_model(workspace_id, request, state)
    return evaluate_model_checks(workspace_id, state, staged, action="edit_model")


def applied_record(result, *, seq, request=None, ts="2026-01-01T00:00:00Z", **metadata):
    """A successful test publication composes the actual owned action payload."""
    from nof1_causal_lab.study.records import (
        Applied,
        AttemptRecord,
        DataComparisonResult,
        DataDiffAttempt,
        DataPreparationResult,
        EditAttempt,
        FitAttempt,
        ModelEditResult,
        ModelFitResult,
        ModelSimulationResult,
        PrepareAttempt,
        SimulateAttempt,
    )

    match result:
        case ModelEditResult():
            attempt = EditAttempt(request=request, outcome=Applied(result=result))
        case DataPreparationResult():
            attempt = PrepareAttempt(request=request, outcome=Applied(result=result))
        case ModelFitResult():
            attempt = FitAttempt(request=request, outcome=Applied(result=result))
        case ModelSimulationResult():
            attempt = SimulateAttempt(request=request, outcome=Applied(result=result))
        case DataComparisonResult():
            attempt = DataDiffAttempt(request=request, outcome=Applied(result=result))
        case _:
            raise TypeError("A publication fixture needs an owned action result")
    return AttemptRecord(seq=seq, ts=ts, attempt=attempt, **metadata)
