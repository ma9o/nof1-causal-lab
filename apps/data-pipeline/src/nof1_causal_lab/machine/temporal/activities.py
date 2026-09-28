"""Activities: the machine's I/O, executed outside the workflow sandbox.

Typed transition failures map to non-retryable ``ApplicationError``s whose
details carry the diagnostics dict — the workflow journals them and hands
them to the navigator as the action's outcome. Anything else (network,
OOM, Modal preemption) is transient infra: the retry policy re-runs it
and the navigator never sees it.
"""

from __future__ import annotations

import asyncio
import json

from temporalio import activity
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.flows.runtime_events import (
    ActionMessageEvent,
    emit_action_message,
    read_events,
)
from nof1_causal_lab.machine.errors import ArtifactWriteRejected, TransitionExecutionError

# Runtime imports (not TYPE_CHECKING): temporalio resolves activity type
# hints at registration time to drive payload conversion.
from nof1_causal_lab.machine.execution import TransitionEffects  # noqa: TC001
from nof1_causal_lab.machine.history import BranchConflict, StudyRepository
from nof1_causal_lab.machine.history_models import BranchBase
from nof1_causal_lab.machine.runners import execute_transition
from nof1_causal_lab.machine.store import TransitionRecord, collect_run_traces, utc_now_iso
from nof1_causal_lab.machine.sweep import collect_completed_runs
from nof1_causal_lab.machine.temporal.llm_subroutine_activities import LLM_SUBROUTINE_ACTIVITIES
from nof1_causal_lab.machine.temporal.measurement_activities import MEASUREMENT_ACTIVITIES
from nof1_causal_lab.machine.temporal.messages import (  # noqa: TC001
    EditModelInput,
    EmitActionMessageInput,
    EvaluateChecksInput,
    JournalInput,
    OperationInput,
    ReadBranchInput,
)
from nof1_causal_lab.machine.temporal.raw_data_activities import RAW_DATA_ACTIVITIES


@activity.defn
async def emit_action_message_activity(input: EmitActionMessageInput) -> None:
    emit_action_message(
        ActionMessageEvent(
            attempt_id=input.attempt_id,
            action=input.action,
            index=input.index,
            message=input.message,
        ),
        input.workspace_id,
    )


@activity.defn
async def run_transition_activity(input: OperationInput) -> TransitionEffects:
    try:
        return await execute_transition(
            input.workspace_id,
            input.operation_id,
            input.state,
            input.options,
            input.input_revisions,
        )
    except TransitionExecutionError as exc:
        raise ApplicationError(
            str(exc),
            exc.diagnostics,
            type=type(exc).__name__,
            non_retryable=True,
        ) from exc


@activity.defn
async def edit_model_activity(input: EditModelInput) -> TransitionEffects:
    try:
        return edit_model(input.workspace_id, input.request, input.state)
    except ArtifactWriteRejected as exc:
        raise ApplicationError(
            str(exc),
            type=type(exc).__name__,
            non_retryable=True,
        ) from exc


@activity.defn
async def evaluate_model_checks_activity(input: EvaluateChecksInput) -> TransitionEffects:
    from nof1_causal_lab.actions.model_checks import evaluate_model_checks

    return await asyncio.to_thread(
        evaluate_model_checks, input.workspace_id, input.state, input.effects, action=input.action
    )


@activity.defn
async def evaluate_data_checks_activity(input: EvaluateChecksInput) -> TransitionEffects:
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
    events = read_events(input.workspace_id)
    return BranchBase(
        commit_id=commit_id,
        state=repository.state(commit_id),
        event_cursor=events[-1].cursor if events else None,
    )


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
    if existing is None:
        logs["events.json"] = json.dumps(
            [
                event.model_dump(mode="json")
                for event in read_events(input.workspace_id, after=input.event_cursor)
            ]
        ).encode()
    try:
        commit_id = journal.append(
            TransitionRecord(
                seq=input.seq,
                attempt_id=input.attempt_id,
                branch=input.branch,
                ts=existing.ts if existing is not None else utc_now_iso(),
                action=input.action,
                inputs=input.inputs,
                operation_id=input.operation_id,
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
                resume=input.resume,
            ),
            expected_head=input.expected_head,
            logs=logs,
        )
        if input.attempt_id is not None:
            for index, message in enumerate(messages):
                emit_action_message(
                    ActionMessageEvent(
                        attempt_id=input.attempt_id,
                        action=input.action,
                        index=index,
                        message=message,
                    ),
                    input.workspace_id,
                )
        return commit_id
    except (BranchConflict, FileExistsError) as exc:
        raise ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True) from exc


@activity.defn
async def collect_completed_runs_activity(workspace_id: str) -> None:
    collect_completed_runs(workspace_id)


ALL_ACTIVITIES = [
    emit_action_message_activity,
    run_transition_activity,
    edit_model_activity,
    journal_activity,
    read_branch_activity,
    collect_completed_runs_activity,
    *RAW_DATA_ACTIVITIES,
    *MEASUREMENT_ACTIVITIES,
    *LLM_SUBROUTINE_ACTIVITIES,
]
