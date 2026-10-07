"""Message types crossing the workflow/activity/facade boundaries.

Kept free of heavy imports (only pydantic + the pure study modules) so
the workflow sandbox can import this module without dragging in storage,
polars, or jax.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.actions.contracts import (
    DataDiffRequest,
    EditModelRequest,
    EditQuestionRequest,
    FitRequest,
    ModelDiffRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SimulateRequest,
)
from nof1_causal_lab.actions.effects import ActionReportName
from nof1_causal_lab.actions.io import PrepareDataInput
from nof1_causal_lab.actions.progress_contracts import ProgressEvent
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_preparation import (
    CompletedExtractionWorker,
    FailedExtractionChunk,
    FilePreparationSpec,
    FileSourceRef,
)
from nof1_causal_lab.artifacts.identity import (
    ArtifactId,
    GitOid,
)
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.llm_specs import (
    EmbeddedLLMSpec,
    HarnessLLMSpec,
    LLMProfileSpec,
)
from nof1_causal_lab.study.records import ActionMessage, Applied, AttemptRecord, StagedActionAttempt
from nof1_causal_lab.study.state import (
    StudyState,
)
from nof1_causal_lab.workers.context import MeasurementContext
from nof1_causal_lab.workers.schemas import ExtractionRow, WorkerOutput


class StudyInit(BaseModel):
    """Workspace identity and starting sequence number for a study workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    # Attempt numbering resumes after the journal; each call names its scientific inputs.
    initial_seq: int = 0


class ReadInputsInput(BaseModel):
    """Pinned action request whose input state or prior saved result must be read."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]


class ActionRequest(BaseModel):
    """Submitted call arguments paired with the attempt identity used for progress tracking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]
    attempt_id: UUID = Field(default_factory=uuid4)


class ActionInput(BaseModel):
    """Resolved scientific request and input state handed to an execution activity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: Annotated[
        FitRequest[GitOid]
        | SimulateRequest[GitOid]
        | DataDiffRequest[GitOid]
        | ModelDiffRequest[GitOid],
        Field(discriminator="action"),
    ]
    state: StudyState


class MeasurementsWorkflowInput(BaseModel):
    """Preparation workflow inputs pinned to a raw table revision and study attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    attempt_id: UUID
    raw_data_revision: GitOid
    preparation: PrepareDataInput[GitOid, FileSourceRef]


class ProgressEventInput(BaseModel):
    """Progress event paired with the workspace in which it must be persisted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    event: ProgressEvent


class LLMToolSpec(BaseModel):
    """An LLM-visible tool schema and its role in completing a subroutine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    parameters: JsonObject
    kind: Literal["read_only", "checkpoint", "terminal"] = "terminal"


class LLMSubroutineRef(BaseModel):
    """Execution identity and stored context shared by a subroutine's messages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_ref: str


class LLMSubroutineInput(BaseModel):
    """Subroutine context, model profile, and tool-turn budget for an LLM workflow."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    llm: LLMProfileSpec
    max_tool_turns: int


class LLMSubroutineStart(BaseModel):
    """Initial conversation and storage locations allocated for one LLM subroutine."""

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
    """A context prompt index and the conversation to which it should be appended."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    conversation_ref: str
    user_message_index: int


class AppendLLMUserMessageResult(BaseModel):
    """Reference to the persisted conversation after adding a user prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str


class AppendLLMRepairMessageInput(BaseModel):
    """Tool failure feedback and the destination for a repaired conversation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    conversation_ref: str
    next_conversation_ref: str
    error_text: str
    tools: list[LLMToolSpec] = Field(default_factory=list)


class AppendLLMRepairMessageResult(BaseModel):
    """Reference to the persisted conversation containing repair feedback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str


class LLMToolExecutionInput(BaseModel):
    """Stored assistant response, allowed tools, and destinations for execution results."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    conversation_ref: str
    assistant_ref: str
    execution_ref: str
    result_ref: str
    tools: list[LLMToolSpec] = Field(default_factory=list)


class LLMToolExecutionResult(BaseModel):
    """Conversation after tool execution and any result accepted by a terminal tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    terminal_success: bool
    result_ref: str | None = None
    tool_calls_fired: list[str] = Field(default_factory=list)


class HarnessTurnInput(BaseModel):
    """Harness session inputs and Temporal routing identities for one model turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workflow_id: str
    workflow_run_id: str
    subroutine: LLMSubroutineRef
    harness_state_ref: str
    harness_tool_ref_base: str
    result_ref: str
    llm: HarnessLLMSpec
    tools: list[LLMToolSpec] = Field(default_factory=list)
    user_message_index: int
    log_label: str


class HarnessToolRequest(BaseModel):
    """Harness tool invocation with the request and response paths used for rendezvous."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    subroutine: LLMSubroutineRef
    result_ref: str
    tool: LLMToolSpec
    tool_name: str
    arguments: JsonObject
    request_ref: str
    response_ref: str


class HarnessToolExecutionResult(BaseModel):
    """Correlated harness tool response and any successfully retained scientific output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    tool_name: str
    output: str
    result_ref: str | None = None
    success: bool = False


class HarnessTurnResult(BaseModel):
    """Saved harness session and trace, with any terminal output produced by the turn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    harness_state_ref: str
    trace_ref: str
    result_ref: str | None = None
    terminal_tool_name: str | None = None
    tool_calls_fired: list[str] = Field(default_factory=list)


class LLMSubroutineResult(BaseModel):
    """Completed extraction output and conversation provenance, including model call counts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    result_ref: str
    conversation_ref: str
    trace_ref: str
    n_llm_calls: int = 0
    n_harness_turns: int = 0


class LLMSubroutineTraceInput(BaseModel):
    """Conversation and call-trace references to assemble into a subroutine trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    conversation_ref: str
    call_ref_base: str
    harness_trace_refs: list[str] = Field(default_factory=list)


class LLMSubroutineTraceResult(BaseModel):
    """Reference to the assembled and persisted subroutine trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_ref: str


class ReadSourceDataInput(BaseModel):
    """Captured source files to read into a workspace's raw table artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    source: FileSourceRef


class MeasurementChunkRef(BaseModel):
    """One extraction chunk's specification and optional reusable worker output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: int
    n_windows: int
    spec_ref: str
    cached_result_ref: str | None = None


class MeasurementsPlan(BaseModel):
    """Pinned measurement plan with extraction chunks and their execution limits."""

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


class ExtractionChunkWorkflowInput(BaseModel):
    """One planned extraction chunk and the LLM settings used to execute it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    chunk: MeasurementChunkRef
    llm: EmbeddedLLMSpec
    max_tool_turns: int


class OpenRouterCallInput(BaseModel):
    """Persisted conversation, model settings, and output paths for one OpenRouter call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    next_conversation_ref: str
    call_ref: str
    assistant_ref: str
    llm: EmbeddedLLMSpec
    tools: list[LLMToolSpec] = Field(default_factory=list)
    log_label: str


class ToolCallSummary(BaseModel):
    """Tool call identity and position within an assistant response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index: int
    id: str
    name: str


class OpenRouterCallResult(BaseModel):
    """Stored provider response references, token usage, duration, and requested tools."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    assistant_ref: str
    model: str
    stop_reason: str | None = None
    time: float
    usage: dict[str, int | None] | None = None
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)


class ExtractionChunkFinalizeInput(BaseModel):
    """Chunk output and execution provenance to validate and retain as measurements."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    worker_id: int
    attempt: int
    n_windows: int
    result_ref: str
    conversation_ref: str
    n_llm_calls: int
    spec_ref: str
    reused: bool | None = False


class CompletedExtractionChunk(CompletedExtractionWorker):
    """The worker's execution envelope adds only its transient result path."""

    n_llm_calls: int | None = 0
    reused: bool | None = False
    result_ref: str


type ExtractionChunkResult = Annotated[
    CompletedExtractionChunk | FailedExtractionChunk, Field(discriminator="status")
]


class MeasurementsFinalizeInput(BaseModel):
    """Pinned plan and completed or failed chunks to assemble into a prepared panel."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    plan_ref: str
    pins: dict[ArtifactId, GitOid]
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)


class EditQuestionActivityInput(BaseModel):
    """Question edit and destination workspace passed across the activity boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: EditQuestionRequest


class EditModelActivityInput(BaseModel):
    """Pinned model edit and destination workspace passed across the activity boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: EditModelRequest[GitOid]


class EvaluateChecksInput[ResultT](Value):
    """Successful scientific result and its input state supplied to post-action checks."""

    workspace_id: str
    state: StudyState
    applied: Applied[ResultT]
    request: (
        EditModelRequest[GitOid] | FitRequest[GitOid] | PrepareDataRequest[GitOid, FileSourceRef]
    )


class ChecksResult(Value):
    """Checks stay in the artifact store; orchestration carries references and labels."""

    reports: Mapping[ActionReportName, GitOid]
    messages: tuple[ActionMessage, ...]


class CompleteResultInput(Value):
    """The completed execution whose full scientific result must be saved before publication."""

    workspace_id: str
    attempt: StagedActionAttempt


class AttemptPublication(Value):
    """The writer receives the same record; Git publication identity is computed outside it."""

    workspace_id: str
    parent_id: GitOid | None
    record: AttemptRecord


class MeasurementChunkContext(BaseModel):
    """Stored input of one foreign extraction subroutine, composing its measurement owner."""

    question: str
    window_text: str
    window_starts: list[str]
    measurement_structure: MeasurementContext
    cache_ref: str | None = None


class MeasurementsFile(BaseModel):
    """Materialization inputs retained by the existing plan's scratch transport."""

    measurement_structure: MeasurementContext
    preparation: FilePreparationSpec
    computed_dicts: list[ExtractionRow]
    empty_output: WorkerOutput
    chunks: list[MeasurementChunkRef]


class StoredConversation(TypedDict):
    """Ordered provider messages persisted between LLM activity calls."""

    messages: list[JsonObject]
