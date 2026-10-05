"""Message types crossing the workflow/activity/facade boundaries.

Kept free of heavy imports (only pydantic + the pure study modules) so
the workflow sandbox can import this module without dragging in storage,
polars, or jax.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.actions.contracts import (
    EditModelRequest,
    FitRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SetQuestionRequest,
    SimulateRequest,
)
from nof1_causal_lab.actions.progress_contracts import ProgressEvent
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_preparation import (
    FilePreparationSpec,
    FileSourceRef,
)
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import (
    ArtifactId,
    GitOid,
)
from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
    ValidationReportArtifact,
)
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.llm_specs import (
    EmbeddedLLMSpec,
    HarnessLLMSpec,
    LLMProfileSpec,
)
from nof1_causal_lab.study.records import (
    Applied,
    AttemptRecord,
    CompletedExtractionWorker,
    FailedExtractionChunk,
)
from nof1_causal_lab.study.state import (
    StudyState,
)
from nof1_causal_lab.study.view_models import DataDiffRequest, ModelDiffRequest
from nof1_causal_lab.workers.context import MeasurementContext
from nof1_causal_lab.workers.schemas import ExtractionRow, WorkerOutput

LLMSubroutineContextKind = Literal[
    "measurement_extraction",
    "raw_data_ingestion",
]


class StudyInit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    # Attempt numbering resumes after the journal; each call names its scientific inputs.
    initial_seq: int = 0


class ReadInputsInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: ScientificActionRequest | DataDiffRequest | ModelDiffRequest


class ActionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request: ScientificActionRequest | DataDiffRequest | ModelDiffRequest
    attempt_id: UUID = Field(default_factory=uuid4)


class ActionInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: Annotated[
        FitRequest | SimulateRequest | PrepareDataRequest | DataDiffRequest | ModelDiffRequest,
        Field(discriminator="action"),
    ]
    state: StudyState


class MeasurementsWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    attempt_id: UUID
    raw_data_revision: GitOid
    preparation: FilePreparationSpec


class ProgressEventInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    event: ProgressEvent


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


class LLMSubroutineRef(BaseModel):
    """Execution identity and stored context shared by a subroutine's messages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    subroutine_id: str
    context_kind: LLMSubroutineContextKind
    context_ref: str


class LLMSubroutineInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    llm: LLMProfileSpec
    max_tool_turns: int


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

    subroutine: LLMSubroutineRef
    conversation_ref: str
    user_message_index: int


class AppendLLMUserMessageResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str


class AppendLLMRepairMessageInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    conversation_ref: str
    next_conversation_ref: str
    error_text: str
    tools: list[LLMToolSpec] = Field(default_factory=list)


class AppendLLMRepairMessageResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str


class LLMToolExecutionInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    subroutine: LLMSubroutineRef
    conversation_ref: str
    assistant_ref: str
    execution_ref: str
    result_ref: str
    tools: list[LLMToolSpec] = Field(default_factory=list)


class LLMToolExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_ref: str
    terminal_success: bool
    result_ref: str | None = None
    tool_calls_fired: list[str] = Field(default_factory=list)


class HarnessTurnInput(BaseModel):
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

    subroutine: LLMSubroutineRef
    conversation_ref: str
    call_ref_base: str
    harness_trace_refs: list[str] = Field(default_factory=list)


class LLMSubroutineTraceResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    trace_ref: str


class IngestionWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    seq: int
    attempt_id: UUID
    source: FileSourceRef


class IngestionPlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    context_ref: str
    llm: LLMProfileSpec
    max_tool_turns: int
    cached_result_ref: str | None = None


class IngestionFinalizeInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    context_ref: str
    result_ref: str


class MeasurementChunkRef(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    worker_id: int
    n_windows: int
    spec_ref: str
    cached_result_ref: str | None = None


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


class ExtractionChunkWorkflowInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    chunk: MeasurementChunkRef
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
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    run_id: str
    plan_ref: str
    pins: dict[ArtifactId, GitOid]
    chunk_results: list[ExtractionChunkResult] = Field(default_factory=list)


class SetQuestionInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: SetQuestionRequest


class EditModelInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_id: str
    request: EditModelRequest
    state: StudyState


class EvaluateChecksInput[ResultT](Value):
    workspace_id: str
    state: StudyState
    applied: Applied[ResultT]
    request: EditModelRequest | FitRequest | PrepareDataRequest


class AttemptPublication(Value):
    """The writer receives the same record; Git publication identity is computed outside it."""

    workspace_id: str
    parent_id: GitOid | None
    record: AttemptRecord
    model_checks: (
        tuple[ModelCheckReport, IdentificationReport, ValidationReportArtifact | None] | None
    ) = None
    data_profile: DataProfileArtifact | None = None


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
    messages: list[JsonObject]
