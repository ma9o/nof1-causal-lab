"""Temporal workflow that ingests uploaded files into a raw-data table."""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    # Temporal resolves workflow result annotations when registering the class.
    from nof1_causal_lab.actions.effects import ActionEffects
    from nof1_causal_lab.actions.progress import StepError, StepEvent, StepStatus
    from nof1_causal_lab.actions.temporal.ingestion_activities import (
        finalize_ingestion_activity,
        plan_ingestion_activity,
    )
    from nof1_causal_lab.actions.temporal.llm_subroutine_workflow import LLMSubroutineWorkflow
    from nof1_causal_lab.actions.temporal.messages import (
        IngestionFinalizeInput,
        IngestionWorkflowInput,
        LLMSubroutineInput,
    )
    from nof1_causal_lab.actions.temporal.workflow_support import (
        emit_progress,
        temporal_failure_details,
    )

_FINALIZE_TIMEOUT = timedelta(minutes=5)

_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
)


@workflow.defn
class IngestionWorkflow:
    @workflow.run
    async def run(self, input: IngestionWorkflowInput) -> ActionEffects:
        def step(status: StepStatus, error: StepError | None = None) -> StepEvent:
            return StepEvent(
                attempt_id=input.attempt_id, step="ingestion", status=status, error=error
            )

        await emit_progress(input.workspace_id, step("running"))
        try:
            context_kind = "raw_data_ingestion"
            subroutine_id = "raw-data"
            plan = await workflow.execute_activity(
                plan_ingestion_activity,
                input,
                start_to_close_timeout=timedelta(minutes=30),
                retry_policy=_ACTIVITY_RETRY,
                summary="Plan raw-data ingestion",
            )
            result_ref = plan.cached_result_ref
            if result_ref is None:
                subroutine = await workflow.execute_child_workflow(
                    LLMSubroutineWorkflow.run,
                    LLMSubroutineInput(
                        workspace_id=input.workspace_id,
                        run_id=plan.run_id,
                        subroutine_id=subroutine_id,
                        context_kind=context_kind,
                        context_ref=plan.context_ref,
                        llm=plan.llm,
                        max_tool_turns=plan.max_tool_turns,
                    ),
                    id=f"llm-raw-data-{input.workspace_id}-{input.seq:06d}",
                    task_queue=workflow.info().task_queue,
                    static_summary="LLM raw-data ingestion subroutine",
                    static_details=(
                        f"workspace={input.workspace_id}; subroutine={subroutine_id}; "
                        f"context={context_kind}"
                    ),
                    memo={
                        "workspace_id": input.workspace_id,
                        "subroutine_id": subroutine_id,
                        "context_kind": context_kind,
                        "run_id": plan.run_id,
                    },
                )
                result_ref = subroutine.result_ref
            effects = await workflow.execute_activity(
                finalize_ingestion_activity,
                IngestionFinalizeInput(
                    workspace_id=input.workspace_id,
                    context_ref=plan.context_ref,
                    result_ref=result_ref,
                ),
                start_to_close_timeout=_FINALIZE_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                summary="Finalize raw-data ingestion",
            )
        except Exception as exc:
            failure_type, failure_message, _ = temporal_failure_details(exc)
            await emit_progress(
                input.workspace_id,
                step("failed", StepError(type=failure_type, message=failure_message)),
            )
            raise
        await emit_progress(input.workspace_id, step("completed"))
        return effects
