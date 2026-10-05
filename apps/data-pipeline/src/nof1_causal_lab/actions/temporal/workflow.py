"""Serialize scientific attempts and publish their correlated outcomes atomically."""

from __future__ import annotations

import asyncio
import re
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ChildWorkflowError

from nof1_causal_lab.actions.effects import ActionEffects

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identification import IdentificationReport
    from nof1_causal_lab.artifacts.model_checks import ModelCheckReport

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.actions.contracts import (
        EditModelRequest,
        FitRequest,
        PrepareDataRequest,
        SetQuestionRequest,
        call_identity,
    )
    from nof1_causal_lab.actions.messages import completion_messages
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
        ReadInputsInput,
        SetQuestionInput,
        StudyInit,
    )
    from nof1_causal_lab.actions.temporal.workflow_support import temporal_failure
    from nof1_causal_lab.artifacts.data_preparation import FilePreparationSpec
    from nof1_causal_lab.artifacts.validation_report import (
        DataProfileArtifact,
        ValidationReportArtifact,
    )
    from nof1_causal_lab.study.records import (
        ActionAttempt,
        ActionBase,
        ActionMessage,
        Applied,
        AttemptRecord,
        DataPreparationResult,
        EditAttempt,
        ModelFitResult,
        PrepareAttempt,
        Rejected,
        SetQuestionAttempt,
        StudyRevision,
        failed_attempt,
    )
    from nof1_causal_lab.study.state import validate_lineage
    from nof1_causal_lab.study.view_models import DataDiffRequest, ModelDiffRequest
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
        self._calls: dict[str, UUID] = {}
        self._messages: tuple[ActionMessage, ...] = ()

    @workflow.run
    async def run(self, init: StudyInit) -> None:
        del init
        await workflow.wait_condition(lambda: self._closed)

    @workflow.update
    async def execute_action(self, request: ActionRequest) -> ActionPoll:
        identity = call_identity(request.request)
        existing = self.call_progress(identity)
        if isinstance(existing, RunningPoll) or (
            isinstance(existing, CompletedPoll) and isinstance(existing.attempt.outcome, Applied)
        ):
            return existing
        self._calls[identity] = request.attempt_id
        self._attempts[request.attempt_id] = RunningPoll(
            attempt_id=request.attempt_id, request=request.request
        )
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
        return self._attempts[request.attempt_id]

    async def _execute_action(self, request: ActionRequest) -> None:
        action = request.request
        self._messages = (
            ActionMessage(
                timestamp=workflow.now(), level="info", label=f"{action.action.upper()}_STARTED"
            ),
        )
        self._report_progress(request)
        try:
            base = await workflow.execute_activity(
                "read_inputs_activity",
                ReadInputsInput(workspace_id=self._workspace_id, request=action),
                result_type=ActionBase,
                start_to_close_timeout=_JOURNAL_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
            )
        except ActivityError as exc:
            self._seq += 1
            failure = temporal_failure(exc)
            outcome = (
                Rejected(reason="input_unavailable", detail=failure.error_message)
                if failure.error_type == "StudyLookupError"
                else failure
            )
            await self._journal(self._seq, request, None, failed_attempt(action, outcome))
            return
        if base.saved is not None:
            saved = base.saved
            self._attempts[request.attempt_id] = CompletedPoll(
                commit_id=saved.commit_id,
                attempt=saved.record.attempt,
                messages=saved.record.messages,
            )
            return
        self._seq += 1
        seq = self._seq
        lineage = (
            validate_lineage(base.state, action.action)
            if not isinstance(action, (DataDiffRequest, ModelDiffRequest))
            else (None if base.state.has("question") else "Set the study question first")
        )
        rejection = (
            Rejected(reason="input_unavailable", detail=lineage) if lineage is not None else None
        )
        if rejection is not None:
            await self._journal(seq, request, base, failed_attempt(action, rejection))
            return
        model_checks = None
        profile = None
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
                    profile = await workflow.execute_activity(
                        "evaluate_data_checks_activity",
                        EvaluateChecksInput[DataPreparationResult](
                            workspace_id=self._workspace_id,
                            state=base.state,
                            applied=Applied(result=result.result, effects=result.effects),
                            request=action,
                        ),
                        result_type=DataProfileArtifact,
                        task_queue=MODEL_CHECKS_TASK_QUEUE,
                        start_to_close_timeout=_CHECK_TIMEOUT,
                        retry_policy=_ACTIVITY_RETRY,
                    )
                    self._messages = (
                        *self._messages,
                        *completion_messages(result.result, workflow.now(), (profile,)),
                    )
                else:
                    assert result.result is None or isinstance(result.result, ModelFitResult)
                    model_checks = await workflow.execute_activity(
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
                    checks, identification, validation = model_checks
                    reports = (
                        (identification, validation)
                        if validation is not None
                        else (identification,)
                    )
                    self._messages = (
                        *self._messages,
                        *completion_messages(result.result, workflow.now(), reports, checks=checks),
                    )
        except (ActivityError, ChildWorkflowError) as exc:
            attempt = failed_attempt(action, temporal_failure(exc))
        await self._journal(
            seq, request, base, attempt, model_checks=model_checks, data_profile=profile
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
                    workers=extracted.result.workers,
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

    @workflow.query
    def call_progress(self, identity: str) -> ActionPoll | None:
        attempt_id = self._calls.get(identity)
        return self._attempts.get(attempt_id) if attempt_id is not None else None

    def _report_progress(self, request: ActionRequest) -> None:
        self._attempts[request.attempt_id] = RunningPoll(
            attempt_id=request.attempt_id, request=request.request, messages=self._messages
        )
        workflow.upsert_memo(
            {
                RUNNING_ACTION_MEMO: RunningAction(
                    attempt_id=request.attempt_id,
                    action=request.request.action,
                    request=request.request,
                    messages=self._messages,
                )
            }
        )

    async def _journal(
        self,
        seq: int,
        request: ActionRequest,
        base: ActionBase | None,
        attempt: ActionAttempt,
        *,
        model_checks: tuple[ModelCheckReport, IdentificationReport, ValidationReportArtifact | None]
        | None = None,
        data_profile: DataProfileArtifact | None = None,
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
                parent_id=base.commit_id if base is not None else None,
                record=AttemptRecord(
                    seq=seq,
                    attempt_id=request.attempt_id,
                    ts=workflow.now().isoformat(),
                    attempt=attempt,
                    messages=messages,
                ),
                model_checks=model_checks,
                data_profile=data_profile,
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
