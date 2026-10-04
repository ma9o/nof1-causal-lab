"""The study's execution and publication I/O, outside the workflow sandbox."""

from __future__ import annotations

import asyncio

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.actions.contracts import PrepareDataRequest
from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.errors import ActionExecutionError
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.actions.runners import run_action
from nof1_causal_lab.actions.set_question import set_question
from nof1_causal_lab.actions.temporal.ingestion_activities import INGESTION_ACTIVITIES
from nof1_causal_lab.actions.temporal.llm_subroutine_activities import LLM_SUBROUTINE_ACTIVITIES
from nof1_causal_lab.actions.temporal.measurement_activities import MEASUREMENT_ACTIVITIES
from nof1_causal_lab.actions.temporal.messages import (
    ActionInput,
    AttemptPublication,
    EditModelInput,
    EvaluateChecksInput,
    ReadInputsInput,
    SetQuestionInput,
)
from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import (
    ActionAttempt,
    Applied,
    ActionBase,
    DataPreparationResult,
    EditAttempt,
    ModelFitResult,
    ModelSimulationResult,
    Rejected,
    SetQuestionAttempt,
    StudyRevision,
    applied_attempt,
    failed_attempt,
)
from nof1_causal_lab.study.store import ArtifactStore, collect_run_traces
from nof1_causal_lab.study.sweep import collect_completed_runs
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact, ValidationReportArtifact


@activity.defn
async def run_action_activity(activity_input: ActionInput) -> ActionAttempt:
    try:
        result = await run_action(
            activity_input.workspace_id, activity_input.request, activity_input.state
        )
    except StudyLookupError as exc:
        return failed_attempt(
            activity_input.request, Rejected(reason="input_unavailable", detail=str(exc))
        )
    except (IncompleteModelError, AggregatedCompileError) as exc:
        return failed_attempt(
            activity_input.request, Rejected(reason="scientific_inputs", detail=str(exc))
        )
    except ActionExecutionError as exc:
        raise ApplicationError(
            str(exc), exc.diagnostics, type=type(exc).__name__, non_retryable=True
        ) from exc
    return applied_attempt(
        activity_input.request, Applied(result=result.result, effects=result.effects)
    )


@activity.defn
async def edit_model_activity(activity_input: EditModelInput) -> EditAttempt:
    result = edit_model(activity_input.workspace_id, activity_input.request, activity_input.state)
    return EditAttempt(request=activity_input.request, outcome=result, action="edit_model")


@activity.defn
async def set_question_activity(activity_input: SetQuestionInput) -> SetQuestionAttempt:
    result = set_question(activity_input.workspace_id, activity_input.request)
    return SetQuestionAttempt(request=activity_input.request, outcome=result, action="set_question")


@activity.defn
async def evaluate_model_checks_activity(
    activity_input: EvaluateChecksInput[ModelFitResult | None],
) -> tuple[ModelCheckReport, IdentificationReport, ValidationReportArtifact | None]:
    from nof1_causal_lab.actions.model_checks import evaluate_model_checks

    if isinstance(activity_input.request, PrepareDataRequest):
        raise TypeError("Model checks require an edit or fit request")
    action = activity_input.request.action
    return await asyncio.to_thread(
        lambda: evaluate_model_checks(
            activity_input.workspace_id,
            activity_input.state,
            activity_input.applied,
            action=action,
        )
    )


@activity.defn
async def evaluate_data_checks_activity(
    activity_input: EvaluateChecksInput[DataPreparationResult],
) -> DataProfileArtifact:
    from nof1_causal_lab.actions.data_checks import evaluate_data_checks

    return await asyncio.to_thread(
        evaluate_data_checks,
        activity_input.workspace_id,
        activity_input.state,
        activity_input.applied,
    )


@activity.defn
async def read_inputs_activity(activity_input: ReadInputsInput) -> ActionBase:
    repository = StudyRepository(activity_input.workspace_id)
    saved = repository.saved_call(activity_input.request)
    return ActionBase(
        commit_id=repository.head(),
        state=StudyState() if saved is not None else repository.input_state(activity_input.request),
        saved=saved,
    )


@activity.defn
async def journal_activity(activity_input: AttemptPublication) -> StudyRevision:
    """Publish the owned record and traces; replay retries use its persisted identity."""
    journal = StudyRepository(activity_input.workspace_id)
    existing = journal.read_attempt(activity_input.record.seq)
    if existing is not None:
        return journal.append(existing.record, parent_id=activity_input.parent_id)
    record = activity_input.record
    messages = record.messages
    if isinstance(record.attempt.outcome, Applied) and messages:
        store = ArtifactStore(activity_input.workspace_id)
        inference, simulation = None, None
        result = record.attempt.outcome.result
        if record.attempt.action == "fit" and isinstance(result, ModelFitResult):
            from nof1_causal_lab.actions.fit import read_inference_report
            produced = next(info for info in record.attempt.outcome.effects.produced if info.artifact_id == "model")
            inference = read_inference_report(store, produced.revision, result.evidence)
        if record.attempt.action == "simulate" and isinstance(result, ModelSimulationResult):
            from nof1_causal_lab.actions.simulate import read_simulation_report
            simulation = read_simulation_report(store, result.evidence, journal.question().revision)
        if inference is not None or simulation is not None:
            messages = messages[:-1] + completion_messages(record.attempt.outcome, messages[-1].timestamp, inference=inference, simulation=simulation) + messages[-1:]
    logs = collect_run_traces(activity_input.workspace_id, record.seq)
    record = record.with_logs(
        messages=messages,
        trace_ids=tuple(path.removeprefix("traces/").removesuffix(".json") for path in logs),
    )
    return journal.append(record, parent_id=activity_input.parent_id, logs=logs)


@activity.defn
async def collect_completed_runs_activity(workspace_id: str) -> None:
    collect_completed_runs(workspace_id)


ALL_ACTIVITIES = [
    run_action_activity,
    set_question_activity,
    edit_model_activity,
    journal_activity,
    read_inputs_activity,
    collect_completed_runs_activity,
    *INGESTION_ACTIVITIES,
    *MEASUREMENT_ACTIVITIES,
    *LLM_SUBROUTINE_ACTIVITIES,
]
