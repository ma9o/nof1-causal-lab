"""Serialize the four scientific actions and commit every attempted outcome."""

from __future__ import annotations

import asyncio
import re
from datetime import timedelta
from typing import Any
from uuid import UUID  # noqa: TC003

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError, ChildWorkflowError

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.actions.contracts import EditModelRequest, ScientificActionRequest
    from nof1_causal_lab.actions.execution import plan_execution
    from nof1_causal_lab.actions.results import ActionMessage, ActionPoll, RunningAction
    from nof1_causal_lab.artifacts.identity import SCIENTIFIC_ACTION_IDS, OperationId
    from nof1_causal_lab.artifacts.model_checks import ModelCheckReport  # noqa: TC001
    from nof1_causal_lab.machine.artifacts import (
        ArtifactRecord,
        EpisodeState,
    )
    from nof1_causal_lab.machine.execution import (
        RetractedArtifact,
        TransitionEffects,
        apply_transition,
        freshness_report,
        validate_model_base,
    )
    from nof1_causal_lab.machine.history_models import BranchBase
    from nof1_causal_lab.machine.status import ActionOutcome, EpisodeStatus
    from nof1_causal_lab.machine.store import ResumeRef
    from nof1_causal_lab.machine.temporal.client import (
        MODEL_CHECKS_TASK_QUEUE,
        RUNNING_ACTION_MEMO,
    )
    from nof1_causal_lab.machine.temporal.messages import (
        ActionRequest,
        EditModelInput,
        EmitActionMessageInput,
        EpisodeInit,
        EvaluateChecksInput,
        JournalInput,
        JournalStatus,
        MeasurementsWorkflowInput,
        OperationInput,
        ReadBranchInput,
        SingleLLMTransitionWorkflowInput,
    )

_RUN_TRANSITION_TIMEOUT = timedelta(hours=4)
_WRITE_TIMEOUT = timedelta(minutes=5)
_CHECK_TIMEOUT = timedelta(hours=1)
_JOURNAL_TIMEOUT = timedelta(minutes=1)
_RUN_COLLECTION_TIMEOUT = timedelta(minutes=10)
_NON_RETRYABLE_ERRORS = [
    "TransitionExecutionError",
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
class EpisodeWorkflow:
    @workflow.init
    def __init__(self, init: EpisodeInit) -> None:
        # workflow.init: updates can be dispatched before the run method's
        # first line executes, and every handler needs workspace_id.
        self._workspace_id = init.workspace_id
        # Queries start at the persisted snapshot; actions capture their own branch
        # through an activity so storage I/O remains outside deterministic replay.
        self._state = init.initial_state if init.initial_state is not None else EpisodeState()
        self._seq = init.initial_seq
        self._branch = "main"
        self._base: BranchBase | None = None
        self._commit_id: str | None = None
        self._head_id: str | None = None
        self._closed = False
        self._lock = asyncio.Lock()
        self._attempts: dict[UUID, ActionPoll] = {}
        self._active_attempt_id: UUID | None = None
        self._messages: tuple[ActionMessage, ...] = ()
        self._running: RunningAction | None = None

    @workflow.run
    async def run(self, init: EpisodeInit) -> EpisodeState:
        del init  # consumed by @workflow.init
        await workflow.wait_condition(lambda: self._closed)
        return self._state

    # -- actions -----------------------------------------------------------

    @workflow.update
    async def execute_action(self, request: ActionRequest) -> ActionOutcome:
        # Serialize actions: one transition at a time per episode. Validation
        # happens inside the accepted update so rejections reach history
        # and the journal (scrubber requirement).
        self._attempts[request.attempt_id] = ActionPoll(done=False)
        async with self._lock:
            try:
                return await self._execute_action(request)
            except (ActivityError, ChildWorkflowError) as exc:
                # Infrastructure can fail before a branch is captured or while
                # committing logs. The accepted update must still terminate.
                error_type, message, diagnostics, _ = _unwrap_temporal_failure(exc)
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
                return self._outcome(
                    self._seq,
                    status="raised",
                    error_type=error_type,
                    error_message=message,
                    diagnostics=diagnostics,
                )
            finally:
                self._running = None
                workflow.upsert_memo({RUNNING_ACTION_MEMO: None})

    async def _execute_action(self, request: ActionRequest) -> ActionOutcome:
        self._active_attempt_id = request.attempt_id
        self._messages = (
            ActionMessage(
                timestamp=workflow.now(),
                level="info",
                label=f"{request.request.action.upper()}_STARTED",
            ),
        )
        self._report_progress(request)
        await workflow.execute_activity(
            "emit_action_message_activity",
            EmitActionMessageInput(
                workspace_id=self._workspace_id,
                attempt_id=request.attempt_id,
                action=request.request.action,
                index=0,
                message=self._messages[0],
            ),
            start_to_close_timeout=_JOURNAL_TIMEOUT,
            retry_policy=_JOURNAL_RETRY,
        )
        self._seq += 1
        seq = self._seq
        action = request.request
        plan = None if isinstance(action, EditModelRequest) else plan_execution(action)
        operation_id = plan.operation.operation_id if plan is not None else None

        self._branch = request.branch
        self._base = await workflow.execute_activity(
            "read_branch_activity",
            ReadBranchInput(workspace_id=self._workspace_id, branch=request.branch),
            result_type=BranchBase,
            start_to_close_timeout=_JOURNAL_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
        )
        self._state = self._base.state
        self._commit_id = self._base.commit_id
        self._head_id = self._base.commit_id
        reason = (
            "Branch conflict: head changed; reload the selected branch before retrying"
            if request.expected_head is not None and request.expected_head != self._base.commit_id
            else validate_model_base(self._state, action.expected_revision)
            if isinstance(action, EditModelRequest)
            else None
        )
        if reason is not None:
            await self._journal(seq, action, operation_id, status="rejected", reason=reason)
            return self._outcome(seq, status="rejected", reason=reason)

        operation = plan.operation if plan is not None else None
        try:
            if isinstance(action, EditModelRequest):
                effects = await workflow.execute_activity(
                    "edit_model_activity",
                    EditModelInput(
                        workspace_id=self._workspace_id, request=action, state=self._state
                    ),
                    result_type=TransitionEffects,
                    start_to_close_timeout=_WRITE_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            elif operation is not None and operation.operation_id == "measurements":
                raw_effects = await workflow.execute_child_workflow(
                    "SingleLLMTransitionWorkflow",
                    SingleLLMTransitionWorkflowInput(
                        workspace_id=self._workspace_id,
                        seq=seq,
                        transition_id="raw_data",
                        state=self._state,
                        source=operation.preparation.source,
                    ),
                    id=f"raw-data-{self._workspace_id}-{seq:06d}",
                    result_type=TransitionEffects,
                    execution_timeout=_RUN_TRANSITION_TIMEOUT,
                    static_summary="Prepare source data",
                    memo={
                        "workspace_id": self._workspace_id,
                        "seq": seq,
                        "action": action.action,
                    },
                )
                effects = await workflow.execute_child_workflow(
                    "MeasurementsWorkflow",
                    MeasurementsWorkflowInput(
                        workspace_id=self._workspace_id,
                        seq=seq,
                        state=apply_transition(self._state, raw_effects.produced),
                        preparation=operation.preparation,
                    ),
                    id=f"measurements-{self._workspace_id}-{seq:06d}",
                    result_type=TransitionEffects,
                    execution_timeout=_RUN_TRANSITION_TIMEOUT,
                    static_summary="Prepare observations",
                    memo={
                        "workspace_id": self._workspace_id,
                        "seq": seq,
                        "action": action.action,
                    },
                )
                effects = effects.model_copy(
                    update={
                        "produced": [*raw_effects.produced, *effects.produced],
                        "diagnostics": {**raw_effects.diagnostics, **effects.diagnostics},
                    }
                )
            else:
                assert plan is not None
                assert operation is not None
                effects = await workflow.execute_activity(
                    "run_transition_activity",
                    OperationInput(
                        workspace_id=self._workspace_id,
                        operation=operation,
                        state=self._state,
                        input_revisions=plan.input_revisions,
                    ),
                    result_type=TransitionEffects,
                    start_to_close_timeout=_RUN_TRANSITION_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            if action.action in {"edit_model", "prepare_data", "fit"}:
                message = ActionMessage(
                    timestamp=workflow.now(),
                    level="info",
                    label="DATA_CHECKS_STARTED"
                    if action.action == "prepare_data"
                    else "MODEL_CHECKS_STARTED",
                )
                self._messages = (*self._messages, message)
                self._report_progress(request)
                await workflow.execute_activity(
                    "emit_action_message_activity",
                    EmitActionMessageInput(
                        workspace_id=self._workspace_id,
                        attempt_id=request.attempt_id,
                        action=action.action,
                        index=len(self._messages) - 1,
                        message=message,
                    ),
                    start_to_close_timeout=_JOURNAL_TIMEOUT,
                    retry_policy=_JOURNAL_RETRY,
                )
                effects = await workflow.execute_activity(
                    "evaluate_data_checks_activity"
                    if action.action == "prepare_data"
                    else "evaluate_model_checks_activity",
                    EvaluateChecksInput(
                        workspace_id=self._workspace_id,
                        action=action.action,
                        state=self._state,
                        effects=effects,
                    ),
                    result_type=TransitionEffects,
                    task_queue=MODEL_CHECKS_TASK_QUEUE,
                    start_to_close_timeout=_CHECK_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            produced = effects.produced
            retracted = effects.retracted
        except (ActivityError, ChildWorkflowError) as exc:
            error_type, error_message, diagnostics, resume = _unwrap_temporal_failure(exc)
            await self._journal(
                seq,
                action,
                operation_id,
                status="raised",
                error_type=error_type,
                error_message=error_message,
                diagnostics=diagnostics,
                resume=resume,
            )
            return self._outcome(
                seq,
                status="raised",
                error_type=error_type,
                error_message=error_message,
                diagnostics=diagnostics,
            )

        try:
            await self._journal(
                seq,
                action,
                operation_id,
                status="applied",
                diagnostics=effects.diagnostics,
                produced=produced,
                retracted=retracted,
                checks=effects.checks,
            )
        except ActivityError as exc:
            error_type, message, _, _ = _unwrap_temporal_failure(exc)
            if error_type != "BranchConflict":
                raise
            await self._journal(
                seq,
                action,
                operation_id,
                status="raised",
                error_type=error_type,
                error_message=message,
                diagnostics=effects.diagnostics,
            )
            return self._outcome(
                seq,
                status="raised",
                error_type=error_type,
                error_message=message,
                diagnostics=effects.diagnostics,
            )
        self._state = apply_transition(self._state, produced, retracted, effects.checks)
        return self._outcome(
            seq,
            status="applied",
            produced=produced,
            retracted=retracted,
            diagnostics=effects.diagnostics,
        )

    @workflow.signal
    def close(self) -> None:
        self._closed = True

    # -- queries ---------------------------------------------------------

    @workflow.query
    def action_progress(self, attempt_id: UUID) -> ActionPoll | None:
        """Accepted updates remain queryable while they wait or execute."""
        return self._attempts.get(attempt_id)

    @workflow.query
    def get_state(self) -> EpisodeState:
        return self._state

    @workflow.query
    def get_status(self) -> EpisodeStatus:
        return EpisodeStatus(
            workspace_id=self._workspace_id,
            branch=self._branch,
            commit_id=self._head_id,
            seq=self._seq,
            state=self._state,
            artifacts=freshness_report(self._state),
            actions=list(SCIENTIFIC_ACTION_IDS),
            running=self._running,
        )

    # -- internals -------------------------------------------------------

    def _report_progress(self, request: ActionRequest) -> None:
        """Expose the executing attempt's labels to its poll query and to the memo."""
        self._attempts[request.attempt_id] = ActionPoll(done=False, messages=self._messages)
        self._running = RunningAction(
            attempt_id=request.attempt_id,
            action=request.request.action,
            branch=request.branch,
            messages=self._messages,
        )
        workflow.upsert_memo({RUNNING_ACTION_MEMO: self._running})

    def _outcome(self, seq: int, **kwargs: Any) -> ActionOutcome:
        return ActionOutcome(
            seq=seq, state=self._state, branch=self._branch, commit_id=self._commit_id, **kwargs
        )

    async def _journal(
        self,
        seq: int,
        request: ScientificActionRequest,
        operation_id: OperationId | None,
        *,
        status: JournalStatus,
        reason: str | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        diagnostics: UncheckedJsonObject | None = None,
        produced: list[ArtifactRecord] | None = None,
        retracted: list[RetractedArtifact] | None = None,
        checks: ModelCheckReport | None = None,
        resume: ResumeRef | None = None,
    ) -> None:
        assert self._base is not None
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
        commit_id = await workflow.execute_activity(
            "journal_activity",
            JournalInput(
                workspace_id=self._workspace_id,
                branch=self._branch,
                expected_head=self._base.commit_id,
                event_cursor=self._base.event_cursor,
                seq=seq,
                action=request.action,
                inputs=request.model_dump(mode="json", exclude={"action"}),
                operation_id=operation_id,
                status=status,
                reason=reason,
                error_type=error_type,
                error_message=error_message,
                diagnostics=diagnostics or {},
                checks=checks,
                produced=produced or [],
                retracted=retracted or [],
                resume=resume,
                attempt_id=self._active_attempt_id,
                messages=messages,
            ),
            result_type=str,
            start_to_close_timeout=_JOURNAL_TIMEOUT,
            retry_policy=_JOURNAL_RETRY,
        )
        self._commit_id = commit_id
        if status == "applied":
            self._head_id = commit_id
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
) -> tuple[str, str, UncheckedJsonObject, ResumeRef | None]:
    cause = exc.cause
    while isinstance(cause, (ActivityError, ChildWorkflowError)):
        cause = cause.cause
    if isinstance(cause, ApplicationError):
        diagnostics: UncheckedJsonObject = {}
        if cause.details:
            first = cause.details[0]
            if isinstance(first, dict):
                diagnostics = dict(first)
        resume_payload = diagnostics.pop("resume", None)
        resume = ResumeRef.model_validate(resume_payload) if resume_payload is not None else None
        return cause.type or "ApplicationError", cause.message, diagnostics, resume
    return type(cause).__name__ if cause else "ActivityError", str(cause or exc), {}, None
