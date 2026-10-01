"""Activities: the study workflow's I/O, executed outside the workflow sandbox.

Typed action failures map to non-retryable ``ApplicationError``s whose details
carry the diagnostics dict — the workflow journals them as the attempt's outcome.
Anything else (network, OOM, Modal preemption) is transient infra: the retry
policy re-runs it and the caller never sees it.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.actions.edit_model import edit_model

# Runtime imports (not TYPE_CHECKING): temporalio resolves activity type
# hints at registration time to drive payload conversion.
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.actions.errors import ActionExecutionError
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.actions.runners import run_action
from nof1_causal_lab.actions.temporal.ingestion_activities import INGESTION_ACTIVITIES
from nof1_causal_lab.actions.temporal.llm_subroutine_activities import LLM_SUBROUTINE_ACTIVITIES
from nof1_causal_lab.actions.temporal.measurement_activities import MEASUREMENT_ACTIVITIES
from nof1_causal_lab.actions.temporal.messages import (
    ActionInput,
    EditModelInput,
    EvaluateChecksInput,
    JournalInput,
    ReadBranchInput,
)
from nof1_causal_lab.study.errors import ArtifactWriteRejected
from nof1_causal_lab.study.history import BranchConflict, StudyRepository
from nof1_causal_lab.study.records import AttemptRecord, BranchBase
from nof1_causal_lab.study.store import collect_run_traces, utc_now_iso
from nof1_causal_lab.study.sweep import collect_completed_runs


@activity.defn
async def run_action_activity(input: ActionInput) -> ActionEffects:
    try:
        return await run_action(input.workspace_id, input.request, input.state)
    except ActionExecutionError as exc:
        raise ApplicationError(
            str(exc),
            exc.diagnostics,
            type=type(exc).__name__,
            non_retryable=True,
        ) from exc


@activity.defn
async def edit_model_activity(input: EditModelInput) -> ActionEffects:
    try:
        return edit_model(input.workspace_id, input.request, input.state)
    except ArtifactWriteRejected as exc:
        raise ApplicationError(
            str(exc),
            type=type(exc).__name__,
            non_retryable=True,
        ) from exc


@activity.defn
async def evaluate_model_checks_activity(
    input: EvaluateChecksInput[Literal["edit_model", "fit"]],
) -> ActionEffects:
    from nof1_causal_lab.actions.model_checks import evaluate_model_checks

    return await asyncio.to_thread(
        evaluate_model_checks, input.workspace_id, input.state, input.effects, action=input.action
    )


@activity.defn
async def evaluate_data_checks_activity(
    input: EvaluateChecksInput[Literal["prepare_data"]],
) -> ActionEffects:
    from nof1_causal_lab.actions.data_checks import evaluate_data_checks

    return await asyncio.to_thread(
        evaluate_data_checks, input.workspace_id, input.state, input.effects
    )


@activity.defn
async def read_branch_activity(input: ReadBranchInput) -> BranchBase:
    repository = StudyRepository(input.workspace_id)
    try:
        commit_id = repository.head(input.branch)
    except ValueError as exc:
        raise ApplicationError(str(exc), non_retryable=True) from exc
    return BranchBase(commit_id=commit_id, state=repository.state(commit_id))


@activity.defn
async def journal_activity(input: JournalInput) -> str:
    """Commit the resulting snapshot and this attempt's logs together."""
    journal = StudyRepository(input.workspace_id)
    existing = journal.read_attempt(input.seq)
    messages = existing.messages if existing is not None else input.messages
    if existing is None and input.status == "applied" and messages:
        messages = (
            messages[:-1]
            + completion_messages(
                input.workspace_id,
                input.action,
                input.produced,
                input.diagnostics,
                messages[-1].timestamp,
                checks=input.checks,
            )
            + messages[-1:]
        )
    logs = collect_run_traces(input.workspace_id, input.seq) if existing is None else {}
    trace_ids = (
        existing.trace_ids
        if existing is not None
        else [path.removeprefix("traces/").removesuffix(".json") for path in logs]
    )
    if existing is None and input.action == "data_diff" and input.status == "applied":
        from nof1_causal_lab.actions.data_diff import DataDiffRequest, read_data_diff

        try:
            report = await asyncio.to_thread(
                read_data_diff, input.workspace_id, DataDiffRequest.model_validate(input.inputs)
            )
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True) from exc
        logs["data-diff.json"] = report.model_dump_json().encode()
    try:
        return journal.append(
            AttemptRecord(
                seq=input.seq,
                attempt_id=input.attempt_id,
                branch=input.branch,
                ts=existing.ts if existing is not None else utc_now_iso(),
                action=input.action,
                inputs=input.inputs,
                status=input.status,
                reason=input.reason,
                error_type=input.error_type,
                error_message=input.error_message,
                diagnostics=input.diagnostics,
                checks=input.checks,
                messages=messages,
                produced=input.produced,
                retracted=input.retracted,
                trace_ids=trace_ids,
            ),
            expected_head=input.expected_head,
            logs=logs,
        )
    except (BranchConflict, FileExistsError) as exc:
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
