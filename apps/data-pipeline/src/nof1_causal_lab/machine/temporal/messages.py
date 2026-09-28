"""Message types crossing the workflow/activity/facade boundaries.

Kept free of heavy imports (only pydantic + the pure machine modules) so
the workflow sandbox can import this module without dragging in storage,
polars, or jax.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.actions.contracts import (  # noqa: TC001
    EditModelRequest,
    ScientificActionRequest,
)
from nof1_causal_lab.actions.results import ActionMessage  # noqa: TC001
from nof1_causal_lab.artifacts.data_preparation import (  # noqa: TC001
    FilePreparationSpec,
    FileSourceRef,
)
from nof1_causal_lab.artifacts.identity import (
    ArtifactId,
    GitOid,
    OperationId,
    ScientificActionId,
)
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport  # noqa: TC001
from nof1_causal_lab.json_types import JsonObject  # noqa: TC001
from nof1_causal_lab.llm_specs import (  # noqa: TC001
    EmbeddedLLMSpec,
    HarnessLLMSpec,
    LLMProfileSpec,
)
from nof1_causal_lab.machine.artifacts import (  # noqa: TC001
    ArtifactRecord,
    EpisodeState,
)
from nof1_causal_lab.machine.execution import (  # noqa: TC001
    LocalOperation,
    RetractedArtifact,
    TransitionEffects,
)
from nof1_causal_lab.machine.store import (
    JournalStatus,  # noqa: TC001
    ResumeRef,  # noqa: TC001
)

LLMSubroutineContextKind = Literal[
    "measurement_extraction",
    "raw_data_ingestion",
]

SingleLLMTransitionId = Literal["raw_data",]
TransitionRuntimeStatus = Literal["running", "completed", "failed"]


class EpisodeInit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    # Initial query state only. Each action captures its own branch head in an activity.
    initial_state: EpisodeState | None = None
    initial_seq: int = 0


class ReadBranchInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    branch: str = "main"


class ActionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    branch: str = "main"
    expected_head: GitOid | None = None
    request: ScientificActionRequest
    attempt_id: UUID = Field(default_factory=uuid4)


class EmitActionMessageInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    attempt_id: UUID
    action: ScientificActionId
    index: int
    message: ActionMessage


class OperationInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    operation: LocalOperation
    state: EpisodeState
    input_revisions: dict[ArtifactId, GitOid] = Field(default_factory=dict)


class MeasurementsWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    state: EpisodeState
    input_revisions: dict[ArtifactId, GitOid] = Field(default_factory=dict)
    preparation: FilePreparationSpec


class LLMToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    parameters: JsonObject
    kind: Literal["read_only", "checkpoint", "terminal"] = "terminal"
    executor: Literal[
        "measurement_validation",
        "raw_data_list_files",
        "raw_data_read_file_sample",
        "raw_data_execute_python",
        "raw_data_submit_table",
    ] = "measurement_validation"
    success_output: str | None = "VALID"


class LLMSubroutineInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str
    llm: LLMProfileSpec
    max_tool_turns: int


class LLMSubroutineStartInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str


class LLMSubroutineStart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    conversation_ref_base: str
    user_message_count: int
    tools: list[LLMToolSpec] = Field(default_factory=list)
    call_ref_base: str
    assistant_ref_base: str
    tool_execution_ref_base: str
    harness_state_ref: str
    harness_tool_ref_base: str
    result_ref_base: str


class AppendLLMUserMessageInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str
    conversation_ref: str
    user_message_index: int


class AppendLLMUserMessageResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str


class AppendLLMRepairMessageInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    conversation_ref: str
    next_conversation_ref: str
    error_text: str
    tools: list[LLMToolSpec] = Field(default_factory=list)


class AppendLLMRepairMessageResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str


class LLMToolExecutionInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str
    conversation_ref: str
    assistant_ref: str
    execution_ref: str
    result_ref: str
    tools: list[LLMToolSpec] = Field(default_factory=list)
    max_tool_output: int | None = None


class LLMToolExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    terminal_success: bool
    result_ref: str | None = None
    feedback_preview: str
    tool_calls_fired: list[str] = Field(default_factory=list)


class HarnessTurnInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    workflow_run_id: str
    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str
    harness_state_ref: str
    harness_tool_ref_base: str
    result_ref: str
    llm: HarnessLLMSpec
    tools: list[LLMToolSpec] = Field(default_factory=list)
    user_message_index: int
    log_label: str


class HarnessToolRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str
    result_ref: str
    tool: LLMToolSpec
    tool_name: str
    arguments: JsonObject
    request_ref: str
    response_ref: str


class HarnessToolExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    tool_name: str
    output: str
    result_ref: str | None = None
    success: bool = False


class HarnessTurnResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    harness_state_ref: str
    trace_ref: str
    completion_preview: str
    result_ref: str | None = None
    terminal_tool_name: str | None = None
    tool_calls_fired: list[str] = Field(default_factory=list)


class LLMSubroutineResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    result_ref: str
    conversation_ref: str
    trace_ref: str
    n_llm_calls: int = 0
    n_harness_turns: int = 0


class LLMSubroutineTraceInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    conversation_ref: str
    call_ref_base: str
    harness_trace_refs: list[str] = Field(default_factory=list)


class LLMSubroutineTraceResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_ref: str


class SingleLLMTransitionWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    transition_id: SingleLLMTransitionId
    state: EpisodeState
    source: FileSourceRef


class SingleLLMTransitionPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    context_ref: str
    pins: dict[ArtifactId, GitOid]
    llm: LLMProfileSpec
    max_tool_turns: int


class SingleLLMTransitionFinalizeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    transition_id: SingleLLMTransitionId
    state: EpisodeState
    pins: dict[ArtifactId, GitOid]
    context_ref: str
    result_ref: str


class MeasurementChunkRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: int
    n_windows: int
    spec_ref: str


class MeasurementsPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    plan_ref: str
    pins: dict[ArtifactId, GitOid]
    chunks: list[MeasurementChunkRef] = Field(default_factory=list)
    max_concurrent_workers: int
    max_rpm: int
    max_tool_turns: int
    llm: EmbeddedLLMSpec


class ExtractionProgressSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_workers: int
    pending_workers: int
    running_workers: int
    completed_workers: int
    failed_workers: int
    llm_requests_last_60s: int = 0


class ExtractionPlanEventInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    kind: Literal["plan"] = "plan"
    total_workers: int
    max_concurrent_workers: int | None = None
    max_rpm: int | None = None


class ExtractionWorkerEventInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    kind: Literal["worker"] = "worker"
    worker_id: int
    state: Literal["pending", "running", "completed", "failed"]
    n_windows: int
    n_extractions: int | None = None
    n_llm_calls: int | None = None
    error: str | None = None


class ExtractionSnapshotEventInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    kind: Literal["snapshot"] = "snapshot"
    snapshot: ExtractionProgressSnapshot


type ExtractionProgressEventInput = Annotated[
    ExtractionPlanEventInput | ExtractionWorkerEventInput | ExtractionSnapshotEventInput,
    Field(discriminator="kind"),
]


class TransitionRuntimeError(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    type: str
    message: str


class TransitionRuntimeEventInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    transition_id: str
    status: TransitionRuntimeStatus
    error: TransitionRuntimeError | None = None


class ExtractionChunkWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    worker_id: int
    n_windows: int
    spec_ref: str
    attempt: int
    llm: EmbeddedLLMSpec
    max_tool_turns: int


class OpenRouterCallInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    next_conversation_ref: str
    call_ref: str
    assistant_ref: str
    llm: EmbeddedLLMSpec
    tools: list[LLMToolSpec] = Field(default_factory=list)
    log_label: str


class ToolCallSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    id: str
    name: str


class OpenRouterCallResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    assistant_ref: str
    model: str
    stop_reason: str | None = None
    time: float
    usage: dict[str, int | None] | None = None
    completion_preview: str
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)


class ExtractionChunkFinalizeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    worker_id: int
    attempt: int
    n_windows: int
    result_ref: str
    conversation_ref: str
    n_llm_calls: int


class ExtractionChunkResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: int
    status: Literal["completed", "failed"]
    n_extractions: int
    n_windows: int
    n_llm_calls: int = 0
    result_ref: str | None = None
    error: str | None = None


class MeasurementsFinalizeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    state: EpisodeState
    run_id: str
    plan_ref: str
    pins: dict[ArtifactId, GitOid]
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)


class EditModelInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: EditModelRequest
    state: EpisodeState


class EvaluateChecksInput[ActionT: ScientificActionId](BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    action: ActionT
    state: EpisodeState
    effects: TransitionEffects


class JournalInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    branch: str = "main"
    expected_head: GitOid | None = None
    event_cursor: str | None = None
    seq: int
    action: ScientificActionId
    inputs: JsonObject
    operation_id: OperationId | None = None
    status: JournalStatus
    reason: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    diagnostics: JsonObject = Field(default_factory=dict)
    checks: ModelCheckReport | None = None
    produced: list[ArtifactRecord] = Field(default_factory=list)
    retracted: list[RetractedArtifact] = Field(default_factory=list)
    resume: ResumeRef | None
    attempt_id: UUID | None = None
    messages: tuple[ActionMessage, ...] = ()
