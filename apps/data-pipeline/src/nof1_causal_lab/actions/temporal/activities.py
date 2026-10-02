"""The study's execution and publication I/O, outside the workflow sandbox."""

from __future__ import annotations

import asyncio

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.errors import ActionExecutionError
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.actions.runners import run_action
from nof1_causal_lab.actions.temporal.ingestion_activities import INGESTION_ACTIVITIES
from nof1_causal_lab.actions.temporal.llm_subroutine_activities import LLM_SUBROUTINE_ACTIVITIES
from nof1_causal_lab.actions.temporal.measurement_activities import MEASUREMENT_ACTIVITIES
from nof1_causal_lab.actions.temporal.messages import (
    ActionInput,
    AttemptPublication,
    EditModelInput,
    EvaluateChecksInput,
    ReadBranchInput,
)
from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError
from nof1_causal_lab.study.errors import ArtifactWriteRejected, StudyLookupError
from nof1_causal_lab.study.history import BranchConflict, StudyRepository
from nof1_causal_lab.study.records import (
    ActionAttempt,
    Applied,
    BranchBase,
    DataPreparationResult,
    EditAttempt,
    ModelEditResult,
    ModelFitResult,
    Rejected,
    StudyRevision,
    applied_attempt,
    failed_attempt,
)
from nof1_causal_lab.study.store import ArtifactStore, collect_run_traces
from nof1_causal_lab.study.sweep import collect_completed_runs


@activity.defn
async def run_action_activity(input: ActionInput) -> ActionAttempt:
    try:
        result = await run_action(input.workspace_id, input.request, input.state)
    except StudyLookupError as exc:
        return failed_attempt(input.request, Rejected(reason="input_unavailable", detail=str(exc)))
    except (IncompleteModelError, AggregatedCompileError) as exc:
        return failed_attempt(input.request, Rejected(reason="scientific_inputs", detail=str(exc)))
    except ActionExecutionError as exc:
        raise ApplicationError(
            str(exc), exc.diagnostics, type=type(exc).__name__, non_retryable=True
        ) from exc
    return applied_attempt(input.request, result)


@activity.defn
async def edit_model_activity(input: EditModelInput) -> EditAttempt:
    try:
        result = edit_model(input.workspace_id, input.request, input.state)
    except ArtifactWriteRejected as exc:
        return EditAttempt(
            request=input.request, outcome=Rejected(reason="revision_conflict", detail=str(exc))
        )
    return EditAttempt(request=input.request, outcome=Applied(result=result))


@activity.defn
async def evaluate_model_checks_activity(
    input: EvaluateChecksInput[ModelEditResult | ModelFitResult],
) -> ModelEditResult | ModelFitResult:
    from nof1_causal_lab.actions.model_checks import evaluate_model_checks

    return await asyncio.to_thread(
        lambda: evaluate_model_checks(
            input.workspace_id, input.state, input.effects, action=input.effects.action
        )
    )


@activity.defn
async def evaluate_data_checks_activity(
    input: EvaluateChecksInput[DataPreparationResult],
) -> DataPreparationResult:
    from nof1_causal_lab.actions.data_checks import evaluate_data_checks

    return await asyncio.to_thread(
        evaluate_data_checks, input.workspace_id, input.state, input.effects
    )


@activity.defn
async def read_branch_activity(input: ReadBranchInput) -> BranchBase:
    repository = StudyRepository(input.workspace_id)
    try:
        commit_id = repository.head(input.branch)
    except StudyLookupError as exc:
        raise ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True) from exc
    return BranchBase(commit_id=commit_id, state=repository.state(commit_id))


@activity.defn
async def journal_activity(input: AttemptPublication) -> StudyRevision:
    """Publish the owned record and traces; replay retries use its persisted identity."""
    journal = StudyRepository(input.workspace_id)
    existing = journal.read_attempt(input.record.seq)
    if existing is not None:
        return journal.append(existing.record, expected_head=input.expected_head)
    record = input.record
    messages = record.messages
    if isinstance(record.attempt.outcome, Applied) and messages:
        messages = (
            messages[:-1]
            + completion_messages(
                record.attempt.outcome.result,
                messages[-1].timestamp,
                ArtifactStore(input.workspace_id).completion_reports(
                    record.attempt.outcome.result.produced
                ),
            )
            + messages[-1:]
        )
    logs = collect_run_traces(input.workspace_id, record.seq)
    record = record.with_logs(
        messages=messages,
        trace_ids=tuple(path.removeprefix("traces/").removesuffix(".json") for path in logs),
    )
    try:
        return journal.append(record, expected_head=input.expected_head, logs=logs)
    except BranchConflict as exc:
        raise ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True) from exc


@activity.defn
async def collect_completed_runs_activity(workspace_id: str) -> None:
    collect_completed_runs(workspace_id)


ALL_ACTIVITIES = [
    run_action_activity,
    edit_model_activity,
    journal_activity,
    read_branch_activity,
    collect_completed_runs_activity,
    *INGESTION_ACTIVITIES,
    *MEASUREMENT_ACTIVITIES,
    *LLM_SUBROUTINE_ACTIVITIES,
]
