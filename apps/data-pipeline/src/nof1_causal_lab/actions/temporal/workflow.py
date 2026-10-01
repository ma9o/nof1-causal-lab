"""Serialize the four scientific actions and commit every attempted outcome."""

from __future__ import annotations

import asyncio
import re
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import TypeAdapter
from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, ChildWorkflowError

from nof1_causal_lab.json_types import JsonObject

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_checks import ModelCheckReport

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.actions.contracts import EditModelRequest, PrepareDataRequest
    from nof1_causal_lab.actions.data_diff import DataDiffRequest
    from nof1_causal_lab.actions.effects import ActionEffects
    from nof1_causal_lab.actions.results import ActionPoll, RunningAction
    from nof1_causal_lab.actions.temporal.client import (
        MODEL_CHECKS_TASK_QUEUE,
        RUNNING_ACTION_MEMO,
    )
    from nof1_causal_lab.actions.temporal.messages import (
        ActionInput,
        ActionRequest,
        EditModelInput,
        EvaluateChecksInput,
        IngestionWorkflowInput,
        JournalInput,
        JournalStatus,
        MeasurementsWorkflowInput,
        ReadBranchInput,
        StudyInit,
    )
    from nof1_causal_lab.artifacts.data_preparation import FilePreparationSpec
    from nof1_causal_lab.study.records import ActionMessage, BranchBase
    from nof1_causal_lab.study.state import (
        ArtifactRecord,
        RetractedArtifact,
        validate_model_base,
    )

_RUN_ACTION_TIMEOUT = timedelta(hours=4)
_WRITE_TIMEOUT = timedelta(minutes=5)
_CHECK_TIMEOUT = timedelta(hours=1)
_JOURNAL_TIMEOUT = timedelta(minutes=1)
_RUN_COLLECTION_TIMEOUT = timedelta(minutes=10)
_NON_RETRYABLE_ERRORS = [
    "ActionExecutionError",
    "ModelCompileError",
    "IncompleteModelError",
    "ModelFitError",
    "ArtifactWriteRejected",
    "ValueError",
]

_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
    non_retryable_error_types=_NON_RETRYABLE_ERRORS,
)
_JOURNAL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    maximum_attempts=5,
)


@workflow.defn
class StudyWorkflow:
    @workflow.init
    def __init__(self, init: StudyInit) -> None:
        # workflow.init: updates can be dispatched before the run method's
        # first line executes, and every handler needs workspace_id.
        self._workspace_id = init.workspace_id
        self._seq = init.initial_seq
        self._closed = False
        self._lock = asyncio.Lock()
        self._attempts: dict[UUID, ActionPoll] = {}
        self._messages: tuple[ActionMessage, ...] = ()

    @workflow.run
    async def run(self, init: StudyInit) -> None:
        del init  # consumed by @workflow.init
        await workflow.wait_condition(lambda: self._closed)

    # -- actions -----------------------------------------------------------

    @workflow.update
    async def execute_action(self, request: ActionRequest) -> None:
        # Serialize actions: one attempt at a time per study. Validation happens
        # inside the accepted update so rejections reach history and the journal.
        self._attempts[request.attempt_id] = ActionPoll(done=False)
        async with self._lock:
            try:
                await self._execute_action(request)
            except (ActivityError, ChildWorkflowError) as exc:
                # Infrastructure can fail before a branch is captured or while
                # committing logs. The accepted update must still terminate.
                error_type, _, _ = _unwrap_temporal_failure(exc)
                progress = self._attempts[request.attempt_id]
                self._attempts[request.attempt_id] = ActionPoll(
                    done=True,
                    messages=(
                        *progress.messages,
                        ActionMessage(
                            timestamp=workflow.now(), level="error", label=_error_label(error_type)
                        ),
                    ),
                )
            finally:
                workflow.upsert_memo({RUNNING_ACTION_MEMO: None})

    async def _execute_action(self, request: ActionRequest) -> None:
        action = request.request
        self._messages = (
            ActionMessage(
                timestamp=workflow.now(),
                level="info",
                label=f"{action.action.upper()}_STARTED",
            ),
        )
        self._report_progress(request)
        self._seq += 1
        seq = self._seq

        base = await workflow.execute_activity(
            "read_branch_activity",
            ReadBranchInput(workspace_id=self._workspace_id, branch=request.branch),
            result_type=BranchBase,
            start_to_close_timeout=_JOURNAL_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
        )
        state = base.state
        reason = (
            "Branch conflict: head changed; reload the selected branch before retrying"
            if not isinstance(action, DataDiffRequest)
            and request.expected_head is not None
            and request.expected_head != base.commit_id
            else validate_model_base(state, action.expected_revision)
            if isinstance(action, EditModelRequest)
            else None
        )
        if reason is not None:
            await self._journal(seq, request, base, status="rejected", reason=reason)
            return

        try:
            if isinstance(action, DataDiffRequest):
                effects = ActionEffects()
            elif isinstance(action, EditModelRequest):
                effects = await workflow.execute_activity(
                    "edit_model_activity",
                    EditModelInput(workspace_id=self._workspace_id, request=action, state=state),
                    result_type=ActionEffects,
                    start_to_close_timeout=_WRITE_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            elif isinstance(action, PrepareDataRequest) and isinstance(
                action.input, FilePreparationSpec
            ):
                effects = await self._prepare_files(seq, request, action.input)
            else:
                effects = await workflow.execute_activity(
                    "run_action_activity",
                    ActionInput(workspace_id=self._workspace_id, request=action, state=state),
                    result_type=ActionEffects,
                    start_to_close_timeout=_RUN_ACTION_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            if action.action in {"edit_model", "prepare_data", "fit"}:
                self._messages = (
                    *self._messages,
                    ActionMessage(
                        timestamp=workflow.now(),
                        level="info",
                        label="DATA_CHECKS_STARTED"
                        if action.action == "prepare_data"
                        else "MODEL_CHECKS_STARTED",
                    ),
                )
                self._report_progress(request)
                effects = await workflow.execute_activity(
                    "evaluate_data_checks_activity"
                    if action.action == "prepare_data"
                    else "evaluate_model_checks_activity",
                    EvaluateChecksInput(
                        workspace_id=self._workspace_id,
                        action=action.action,
                        state=state,
                        effects=effects,
                    ),
                    result_type=ActionEffects,
                    task_queue=MODEL_CHECKS_TASK_QUEUE,
                    start_to_close_timeout=_CHECK_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
        except (ActivityError, ChildWorkflowError) as exc:
            error_type, error_message, diagnostics = _unwrap_temporal_failure(exc)
            await self._journal(
                seq,
                request,
                base,
                status="raised",
                error_type=error_type,
                error_message=error_message,
                diagnostics=diagnostics,
            )
            return

        try:
            await self._journal(
                seq,
                request,
                base,
                status="applied",
                diagnostics=effects.diagnostics,
                produced=effects.produced,
                retracted=effects.retracted,
                checks=effects.checks,
            )
        except ActivityError as exc:
            error_type, message, _ = _unwrap_temporal_failure(exc)
            if error_type != "BranchConflict" and not isinstance(action, DataDiffRequest):
                raise
            await self._journal(
                seq,
                request,
                base,
                status="raised",
                error_type=error_type,
                error_message=message,
                diagnostics=effects.diagnostics,
            )

    async def _prepare_files(
        self, seq: int, request: ActionRequest, preparation: FilePreparationSpec
    ) -> ActionEffects:
        """Ingest the uploaded files, then extract measurements from that raw-data revision."""
        memo = {"workspace_id": self._workspace_id, "seq": seq, "action": "prepare_data"}
        ingested = await workflow.execute_child_workflow(
            "IngestionWorkflow",
            IngestionWorkflowInput(
                workspace_id=self._workspace_id,
                seq=seq,
                attempt_id=request.attempt_id,
                source=preparation.source,
            ),
            id=f"raw-data-{self._workspace_id}-{seq:06d}",
            result_type=ActionEffects,
            execution_timeout=_RUN_ACTION_TIMEOUT,
            static_summary="Prepare source data",
            memo=memo,
        )
        (raw_data,) = ingested.produced
        extracted = await workflow.execute_child_workflow(
            "MeasurementsWorkflow",
            MeasurementsWorkflowInput(
                workspace_id=self._workspace_id,
                seq=seq,
                attempt_id=request.attempt_id,
                raw_data_revision=raw_data.revision,
                preparation=preparation,
            ),
            id=f"measurements-{self._workspace_id}-{seq:06d}",
            result_type=ActionEffects,
            execution_timeout=_RUN_ACTION_TIMEOUT,
            static_summary="Prepare observations",
            memo=memo,
        )
        return ActionEffects(
            produced=[*ingested.produced, *extracted.produced],
            diagnostics={**ingested.diagnostics, **extracted.diagnostics},
        )

    @workflow.signal
    def close(self) -> None:
        self._closed = True

    # -- queries ---------------------------------------------------------

    @workflow.query
    def action_progress(self, attempt_id: UUID) -> ActionPoll | None:
        """Accepted updates remain queryable while they wait or execute."""
        return self._attempts.get(attempt_id)

    # -- internals -------------------------------------------------------

    def _report_progress(self, request: ActionRequest) -> None:
        """Expose the executing attempt's labels to its poll query and to the memo."""
        self._attempts[request.attempt_id] = ActionPoll(done=False, messages=self._messages)
        workflow.upsert_memo(
            {
                RUNNING_ACTION_MEMO: RunningAction(
                    attempt_id=request.attempt_id,
                    action=request.request.action,
                    branch=request.branch,
                    messages=self._messages,
                )
            }
        )

    async def _journal(
        self,
        seq: int,
        request: ActionRequest,
        base: BranchBase,
        *,
        status: JournalStatus,
        reason: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        diagnostics: JsonObject | None = None,
        produced: list[ArtifactRecord] | None = None,
        retracted: list[RetractedArtifact] | None = None,
        checks: ModelCheckReport | None = None,
    ) -> None:
        label = (
            "ACTION_COMPLETED"
            if status == "applied"
            else "REVISION_CONFLICT"
            if status == "rejected" or error_type == "BranchConflict"
            else _error_label(error_type or "ActionError")
        )
        messages = (
            *self._messages,
            ActionMessage(
                timestamp=workflow.now(),
                level="info" if status == "applied" else "error",
                label=label,
            ),
        )
        await workflow.execute_activity(
            "journal_activity",
            JournalInput(
                workspace_id=self._workspace_id,
                branch=request.branch,
                expected_head=request.expected_head
                if isinstance(request.request, DataDiffRequest)
                else base.commit_id,
                seq=seq,
                action=request.request.action,
                inputs=request.request.model_dump(mode="json", exclude={"action"}),
                status=status,
                reason=reason,
                error_type=error_type,
                error_message=error_message,
                diagnostics=diagnostics or {},
                checks=checks,
                produced=produced or [],
                retracted=retracted or [],
                attempt_id=request.attempt_id,
                messages=messages,
            ),
            result_type=str,
            start_to_close_timeout=_CHECK_TIMEOUT
            if isinstance(request.request, DataDiffRequest)
            else _JOURNAL_TIMEOUT,
            retry_policy=_JOURNAL_RETRY,
        )
        # Run collection is lifecycle hygiene, never part of the action commit.
        try:
            await workflow.execute_activity(
                "collect_completed_runs_activity",
                self._workspace_id,
                start_to_close_timeout=_RUN_COLLECTION_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
            )
        except ActivityError as exc:
            workflow.logger.warning("run scratch collection failed after seq %d: %s", seq, exc)


def _error_label(error_type: str) -> str:
    words = re.sub(r"(?<!^)(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])", "_", error_type)
    return "ACTION_FAILED_" + re.sub(r"[^A-Za-z0-9]+", "_", words).strip("_").upper()


def _unwrap_temporal_failure(
    exc: ActivityError | ChildWorkflowError,
) -> tuple[str, str, JsonObject]:
    cause = exc.cause
    while isinstance(cause, (ActivityError, ChildWorkflowError)):
        cause = cause.cause
    if isinstance(cause, ApplicationError):
        diagnostics: JsonObject = {}
        if cause.details:
            first = cause.details[0]
            if isinstance(first, dict):
                diagnostics = TypeAdapter(JsonObject).validate_python(first)
        return cause.type or "ApplicationError", cause.message, diagnostics
    return type(cause).__name__ if cause else "ActivityError", str(cause or exc), {}
