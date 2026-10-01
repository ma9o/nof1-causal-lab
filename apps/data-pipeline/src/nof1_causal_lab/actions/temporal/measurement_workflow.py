"""Temporal workflows that extract measurements from ingested raw data."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ChildWorkflowError

with workflow.unsafe.imports_passed_through():
    from nof1_causal_lab.actions.effects import ActionEffects
    from nof1_causal_lab.actions.progress import (
        ExtractionPlanEvent,
        ExtractionSnapshotEvent,
        ExtractionWorkerEvent,
        ProgressEvent,
        StepError,
        StepEvent,
    )
    from nof1_causal_lab.actions.temporal.messages import (
        ExtractionChunkFinalizeInput,
        ExtractionChunkResult,
        ExtractionChunkWorkflowInput,
        LLMSubroutineInput,
        LLMSubroutineResult,
        MeasurementChunkRef,
        MeasurementsFinalizeInput,
        MeasurementsPlan,
        MeasurementsWorkflowInput,
    )
    from nof1_causal_lab.actions.temporal.workflow_support import (
        emit_progress,
        temporal_failure_details,
    )

_PLAN_TIMEOUT = timedelta(minutes=30)
_FINALIZE_CHUNK_TIMEOUT = timedelta(minutes=5)
_FINALIZE_MEASUREMENTS_TIMEOUT = timedelta(minutes=30)

_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
)
_CHUNK_WORKFLOW_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=1.0,
    maximum_attempts=3,
)


@workflow.defn
class ExtractionChunkWorkflow:
    @workflow.run
    async def run(self, input: ExtractionChunkWorkflowInput) -> ExtractionChunkResult:
        attempt = workflow.info().attempt
        subroutine_id = f"measurement-chunk-{input.worker_id:06d}-attempt-{attempt:03d}"
        result_ref = input.cached_result_ref
        conversation_ref = ""
        n_llm_calls = 0
        if result_ref is None:
            subroutine = await workflow.execute_child_workflow(
                "LLMSubroutineWorkflow",
                LLMSubroutineInput(
                    workspace_id=input.workspace_id,
                    run_id=input.run_id,
                    subroutine_id=subroutine_id,
                    context_kind="measurement_extraction",
                    context_ref=input.spec_ref,
                    llm=input.llm,
                    max_tool_turns=input.max_tool_turns,
                ),
                id=(
                    f"llm-measurement-{input.workspace_id}-{input.run_id}-"
                    f"chunk-{input.worker_id:06d}-attempt-{attempt:03d}"
                ),
                task_queue=workflow.info().task_queue,
                result_type=LLMSubroutineResult,
                static_summary=f"LLM extraction subroutine chunk {input.worker_id}",
                static_details=(
                    f"workspace={input.workspace_id}; run={input.run_id}; "
                    f"chunk={input.worker_id}; attempt={attempt}; "
                    "context=measurement_extraction"
                ),
                memo={
                    "workspace_id": input.workspace_id,
                    "run_id": input.run_id,
                    "worker_id": input.worker_id,
                    "attempt": attempt,
                    "context_kind": "measurement_extraction",
                    "subroutine_id": subroutine_id,
                },
            )
            result_ref = subroutine.result_ref
            conversation_ref = subroutine.conversation_ref
            n_llm_calls = subroutine.n_llm_calls
        return await workflow.execute_activity(
            "finalize_extraction_chunk_activity",
            ExtractionChunkFinalizeInput(
                workspace_id=input.workspace_id,
                run_id=input.run_id,
                worker_id=input.worker_id,
                attempt=attempt,
                n_windows=input.n_windows,
                result_ref=result_ref,
                conversation_ref=conversation_ref,
                n_llm_calls=n_llm_calls,
                spec_ref=input.spec_ref,
                reused=input.cached_result_ref is not None,
            ),
            result_type=ExtractionChunkResult,
            start_to_close_timeout=_FINALIZE_CHUNK_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
            summary=f"Finalize extraction chunk {input.worker_id}",
        )


@workflow.defn
class MeasurementsWorkflow:
    @workflow.run
    async def run(self, input: MeasurementsWorkflowInput) -> ActionEffects:
        chunk_results: list[ExtractionChunkResult] = []
        attempt_id = input.attempt_id

        async def emit(event: ProgressEvent) -> None:
            await emit_progress(input.workspace_id, event)

        await emit(StepEvent(attempt_id=attempt_id, step="extraction", status="running"))
        try:
            plan = await workflow.execute_activity(
                "plan_measurements_activity",
                input,
                result_type=MeasurementsPlan,
                start_to_close_timeout=_PLAN_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                summary="Plan measurements extraction",
            )
            await emit(
                ExtractionPlanEvent(
                    attempt_id=attempt_id,
                    total_workers=len(plan.chunks),
                    max_concurrent_workers=plan.max_concurrent_workers,
                )
            )

            total_workers = len(plan.chunks)
            pending_workers = total_workers
            running_workers = 0
            completed_workers = 0
            failed_workers = 0

            async def emit_snapshot() -> None:
                await emit(
                    ExtractionSnapshotEvent(
                        attempt_id=attempt_id,
                        total_workers=total_workers,
                        pending_workers=pending_workers,
                        running_workers=running_workers,
                        completed_workers=completed_workers,
                        failed_workers=failed_workers,
                    )
                )

            await emit_snapshot()

            semaphore = asyncio.Semaphore(max(1, plan.max_concurrent_workers))

            async def run_chunk(chunk: MeasurementChunkRef) -> ExtractionChunkResult:
                nonlocal pending_workers, running_workers, completed_workers, failed_workers
                async with semaphore:
                    pending_workers -= 1
                    running_workers += 1
                    await emit(
                        ExtractionWorkerEvent(
                            attempt_id=attempt_id,
                            worker_id=chunk.worker_id,
                            state="running",
                            n_windows=chunk.n_windows,
                        )
                    )
                    await emit_snapshot()

                    try:
                        result = await workflow.execute_child_workflow(
                            "ExtractionChunkWorkflow",
                            ExtractionChunkWorkflowInput(
                                workspace_id=input.workspace_id,
                                run_id=plan.run_id,
                                worker_id=chunk.worker_id,
                                n_windows=chunk.n_windows,
                                spec_ref=chunk.spec_ref,
                                cached_result_ref=chunk.cached_result_ref,
                                attempt=1,
                                llm=plan.llm,
                                max_tool_turns=plan.max_tool_turns,
                            ),
                            id=(
                                f"measurements-{input.workspace_id}-"
                                f"{input.seq:06d}-chunk-{chunk.worker_id:06d}"
                            ),
                            task_queue=workflow.info().task_queue,
                            result_type=ExtractionChunkResult,
                            retry_policy=_CHUNK_WORKFLOW_RETRY,
                            static_summary=f"Extract measurements chunk {chunk.worker_id}",
                            static_details=(
                                f"workspace={input.workspace_id}; run={plan.run_id}; "
                                f"chunk={chunk.worker_id}; windows={chunk.n_windows}"
                            ),
                            memo={
                                "workspace_id": input.workspace_id,
                                "run_id": plan.run_id,
                                "worker_id": chunk.worker_id,
                                "n_windows": chunk.n_windows,
                                "workflow_kind": "measurement_chunk",
                            },
                        )
                    except ChildWorkflowError as exc:
                        _, failure_message, _ = temporal_failure_details(exc)
                        result = ExtractionChunkResult(
                            worker_id=chunk.worker_id,
                            status="failed",
                            n_extractions=0,
                            n_windows=chunk.n_windows,
                            error=failure_message,
                        )

                    running_workers -= 1
                    if result.status == "completed":
                        completed_workers += 1
                    else:
                        failed_workers += 1
                    await emit(
                        ExtractionWorkerEvent(
                            attempt_id=attempt_id,
                            worker_id=result.worker_id,
                            state=result.status,
                            n_windows=result.n_windows,
                            n_extractions=result.n_extractions,
                            n_llm_calls=result.n_llm_calls or None,
                            error=result.error,
                        )
                    )
                    await emit_snapshot()
                    return result

            tasks = [asyncio.create_task(run_chunk(chunk)) for chunk in plan.chunks]
            chunk_results = [await task for task in workflow.as_completed(tasks)]
            chunk_results.sort(key=lambda result: result.worker_id)

            effects = await workflow.execute_activity(
                "finalize_measurements_activity",
                MeasurementsFinalizeInput(
                    workspace_id=input.workspace_id,
                    run_id=plan.run_id,
                    plan_ref=plan.plan_ref,
                    pins=plan.pins,
                    chunk_results=chunk_results,
                ),
                result_type=ActionEffects,
                start_to_close_timeout=_FINALIZE_MEASUREMENTS_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                summary="Finalize measurements artifacts",
            )
        except Exception as exc:
            failure_type, failure_message, _ = temporal_failure_details(exc)
            await emit(
                StepEvent(
                    attempt_id=attempt_id,
                    step="extraction",
                    status="failed",
                    error=StepError(type=failure_type, message=failure_message),
                )
            )
            raise
        await emit(StepEvent(attempt_id=attempt_id, step="extraction", status="completed"))
        return effects
