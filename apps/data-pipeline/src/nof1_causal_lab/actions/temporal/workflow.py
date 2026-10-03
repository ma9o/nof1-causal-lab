"""Serialize scientific attempts and publish their correlated outcomes atomically."""

from __future__ import annotations

import asyncio
import re
from datetime import timedelta
from uuid import UUID

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ChildWorkflowError

from nof1_causal_lab.actions.effects import ActionEffects

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.actions.contracts import (
        EditModelRequest,
        FitRequest,
        PrepareDataRequest,
        SetQuestionRequest,
    )
    from nof1_causal_lab.actions.results import (
        ActionPoll,
        CompletedPoll,
        RunningAction,
        RunningPoll,
    )
    from nof1_causal_lab.actions.temporal.activities import (
        evaluate_model_checks_activity,
        run_action_activity,
    )
    from nof1_causal_lab.actions.temporal.client import MODEL_CHECKS_TASK_QUEUE, RUNNING_ACTION_MEMO
    from nof1_causal_lab.actions.temporal.messages import (
        ActionInput,
        ActionRequest,
        AttemptPublication,
        EditModelInput,
        EvaluateChecksInput,
        IngestionWorkflowInput,
        MeasurementsWorkflowInput,
        ReadBranchInput,
        SetQuestionInput,
        StudyInit,
    )
    from nof1_causal_lab.actions.temporal.workflow_support import temporal_failure
    from nof1_causal_lab.artifacts.data_preparation import FilePreparationSpec
    from nof1_causal_lab.study.records import (
        ActionAttempt,
        ActionMessage,
        Applied,
        AttemptRecord,
        BranchBase,
        DataPreparationResult,
        EditAttempt,
        ModelFitResult,
        PrepareAttempt,
        Rejected,
        SetQuestionAttempt,
        StudyRevision,
        applied_attempt,
        failed_attempt,
    )
    from nof1_causal_lab.study.state import validate_lineage, validate_model_base
    from nof1_causal_lab.study.view_models import DataDiffRequest
_RUN_ACTION_TIMEOUT = timedelta(hours=4)
_WRITE_TIMEOUT = timedelta(minutes=5)
_CHECK_TIMEOUT = timedelta(hours=1)
_JOURNAL_TIMEOUT = timedelta(minutes=1)
_RUN_COLLECTION_TIMEOUT = timedelta(minutes=10)
_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
    non_retryable_error_types=[
        "ActionExecutionError",
        "ModelCompileError",
        "IncompleteModelError",
        "ModelFitError",
        "ArtifactWriteRejected",
        "StudyLookupError",
    ],
)
_JOURNAL_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), maximum_attempts=5)


@workflow.defn
class StudyWorkflow:
    @workflow.init
    def __init__(self, init: StudyInit) -> None:
        self._workspace_id = init.workspace_id
        self._seq = init.initial_seq
        self._closed = False
        self._lock = asyncio.Lock()
        self._attempts: dict[UUID, ActionPoll] = {}
        self._messages: tuple[ActionMessage, ...] = ()

    @workflow.run
    async def run(self, init: StudyInit) -> None:
        del init
        await workflow.wait_condition(lambda: self._closed)

    @workflow.update
    async def execute_action(self, request: ActionRequest) -> None:
        self._attempts[request.attempt_id] = RunningPoll()
        async with self._lock:
            try:
                await self._execute_action(request)
            except (ActivityError, ChildWorkflowError) as exc:
                failure = temporal_failure(exc)
                progress = self._attempts[request.attempt_id]
                self._attempts[request.attempt_id] = CompletedPoll(
                    commit_id=None,
                    attempt=failed_attempt(request.request, failure),
                    messages=(
                        *progress.messages,
                        ActionMessage(
                            timestamp=workflow.now(),
                            level="error",
                            label=_error_label(failure.error_type),
                        ),
                    ),
                )
            finally:
                workflow.upsert_memo({RUNNING_ACTION_MEMO: None})

    async def _execute_action(self, request: ActionRequest) -> None:
        action = request.request
        self._messages = (
            ActionMessage(
                timestamp=workflow.now(), level="info", label=f"{action.action.upper()}_STARTED"
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
        reason = (
            "Branch conflict: head changed; reload the selected branch before retrying"
            if not isinstance(action, DataDiffRequest)
            and request.expected_head is not None
            and request.expected_head != base.commit_id
            else validate_model_base(base.state, action.expected_revision)
            if isinstance(action, EditModelRequest)
            else None
        )
        lineage = (
            None
            if isinstance(action, DataDiffRequest)
            else validate_lineage(base.state, action.action)
        )
        rejection = (
            Rejected(reason="revision_conflict", detail=reason)
            if reason is not None
            else Rejected(
                reason="revision_conflict"
                if isinstance(action, SetQuestionRequest)
                else "input_unavailable",
                detail=lineage,
            )
            if lineage is not None
            else None
        )
        if rejection is not None:
            await self._journal(seq, request, base, failed_attempt(action, rejection))
            return
        try:
            if isinstance(action, SetQuestionRequest):
                attempt = await workflow.execute_activity(
                    "set_question_activity",
                    SetQuestionInput(workspace_id=self._workspace_id, request=action),
                    result_type=SetQuestionAttempt,
                    start_to_close_timeout=_WRITE_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            elif isinstance(action, EditModelRequest):
                attempt = await workflow.execute_activity(
                    "edit_model_activity",
                    EditModelInput(
                        workspace_id=self._workspace_id, request=action, state=base.state
                    ),
                    result_type=EditAttempt,
                    start_to_close_timeout=_WRITE_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            elif isinstance(action, PrepareDataRequest) and isinstance(
                action.input, FilePreparationSpec
            ):
                attempt = await self._prepare_files(seq, request, action.input)
            else:
                attempt = await workflow.execute_activity(
                    run_action_activity,
                    ActionInput(workspace_id=self._workspace_id, request=action, state=base.state),
                    start_to_close_timeout=_RUN_ACTION_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                )
            if isinstance(
                action, (EditModelRequest, FitRequest, PrepareDataRequest)
            ) and isinstance(attempt.outcome, Applied):
                result = attempt.outcome
                self._messages = (
                    *self._messages,
                    ActionMessage(
                        timestamp=workflow.now(),
                        level="info",
                        label="DATA_CHECKS_STARTED"
                        if attempt.action == "prepare_data"
                        else "MODEL_CHECKS_STARTED",
                    ),
                )
                self._report_progress(request)
                if isinstance(action, PrepareDataRequest):
                    assert isinstance(result.result, DataPreparationResult)
                    result = await workflow.execute_activity(
                        "evaluate_data_checks_activity",
                        EvaluateChecksInput[DataPreparationResult](
                            workspace_id=self._workspace_id,
                            state=base.state,
                            applied=Applied(result=result.result, effects=result.effects),
                            request=action,
                        ),
                        result_type=Applied[DataPreparationResult],
                        task_queue=MODEL_CHECKS_TASK_QUEUE,
                        start_to_close_timeout=_CHECK_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                else:
                    assert result.result is None or isinstance(result.result, ModelFitResult)
                    result = await workflow.execute_activity(
                        evaluate_model_checks_activity,
                        EvaluateChecksInput[ModelFitResult | None](
                            workspace_id=self._workspace_id,
                            state=base.state,
                            applied=Applied(result=result.result, effects=result.effects),
                            request=action,
                        ),
                        task_queue=MODEL_CHECKS_TASK_QUEUE,
                        start_to_close_timeout=_CHECK_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                attempt = applied_attempt(action, result)
        except (ActivityError, ChildWorkflowError) as exc:
            attempt = failed_attempt(action, temporal_failure(exc))
        try:
            await self._journal(seq, request, base, attempt)
        except ActivityError as exc:
            failure = temporal_failure(exc)
            if failure.error_type != "BranchConflict":
                raise
            await self._journal(
                seq,
                request,
                base,
                failed_attempt(
                    action, Rejected(reason="revision_conflict", detail=failure.error_message)
                ),
            )

    async def _prepare_files(
        self, seq: int, request: ActionRequest, preparation: FilePreparationSpec
    ) -> PrepareAttempt:
        assert isinstance(request.request, PrepareDataRequest)
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
            result_type=Applied[DataPreparationResult],
            execution_timeout=_RUN_ACTION_TIMEOUT,
            static_summary="Prepare source data",
            memo=memo,
        )
        (raw_data,) = ingested.effects.produced
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
            result_type=Applied[DataPreparationResult],
            execution_timeout=_RUN_ACTION_TIMEOUT,
            static_summary="Prepare observations",
            memo=memo,
        )
        return PrepareAttempt(
            action="prepare_data",
            request=request.request,
            outcome=Applied(
                result=DataPreparationResult(
                    raw_data=extracted.result.raw_data,
                    workers=extracted.result.workers,
                    n_observations=extracted.result.n_observations,
                    ingestion_reused=ingested.result.ingestion_reused,
                    extraction_reused=extracted.result.extraction_reused,
                ),
                effects=ActionEffects(
                    produced=(*ingested.effects.produced, *extracted.effects.produced)
                ),
            ),
        )

    @workflow.signal
    def close(self) -> None:
        self._closed = True

    @workflow.query
    def action_progress(self, attempt_id: UUID) -> ActionPoll | None:
        return self._attempts.get(attempt_id)

    def _report_progress(self, request: ActionRequest) -> None:
        self._attempts[request.attempt_id] = RunningPoll(messages=self._messages)
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
        self, seq: int, request: ActionRequest, base: BranchBase, attempt: ActionAttempt
    ) -> None:
        outcome = attempt.outcome
        label = (
            "ACTION_COMPLETED"
            if isinstance(outcome, Applied)
            else "ACTION_REJECTED"
            if isinstance(outcome, Rejected)
            else _error_label(outcome.error_type)
        )
        messages = (
            *self._messages,
            ActionMessage(
                timestamp=workflow.now(),
                level="info" if isinstance(outcome, Applied) else "error",
                label=label,
            ),
        )
        revision = await workflow.execute_activity(
            "journal_activity",
            AttemptPublication(
                workspace_id=self._workspace_id,
                expected_head=request.expected_head
                if isinstance(request.request, DataDiffRequest)
                else base.commit_id,
                record=AttemptRecord(
                    seq=seq,
                    attempt_id=request.attempt_id,
                    branch=request.branch,
                    ts=workflow.now().isoformat(),
                    attempt=attempt,
                    messages=messages,
                ),
            ),
            result_type=StudyRevision,
            start_to_close_timeout=_JOURNAL_TIMEOUT,
            retry_policy=_JOURNAL_RETRY,
        )
        self._attempts[request.attempt_id] = CompletedPoll(
            commit_id=revision.commit_id,
            attempt=revision.record.attempt,
            messages=revision.record.messages,
        )
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
