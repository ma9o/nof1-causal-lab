"""Correlated scientific attempts; Git publication and attempt metadata have one owner."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field

from nof1_causal_lab.actions.contracts import (
    EditModelRequest,
    FitRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SetQuestionRequest,
    SimulateRequest,
)
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
from nof1_causal_lab.artifacts.identity import GitOid, GitRef
from nof1_causal_lab.artifacts.posterior import InferenceEvidence
from nof1_causal_lab.artifacts.simulation import SimulationEvidence
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.view_models import (
    DataDiffRequest,
    ModelDiffRequest,
    PanelRef,
    SimulationRef,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class ActionMessage(Value):
    """A label emitted by an attempt; measurements belong in its scientific result."""

    timestamp: AwareDatetime
    level: Literal["debug", "info", "warn", "error"]
    label: str = Field(pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")


class _ExtractionWorkerMeasurements(Value):
    """Retained extraction measurements, independent of a transient result-file path."""

    worker_id: int
    n_extractions: int
    n_windows: int


class CompletedExtractionWorker(_ExtractionWorkerMeasurements):
    """Retained measurements from a completed worker; no failure field exists."""

    status: Literal["completed"] = "completed"


class FailedExtractionChunk(_ExtractionWorkerMeasurements):
    """An extraction failure with its error and no usable result-file reference."""

    status: Literal["failed"] = "failed"
    error: str
    n_llm_calls: int | None = 0
    reused: bool | None = False


type ExtractionWorkerResult = Annotated[
    CompletedExtractionWorker | FailedExtractionChunk, Field(discriminator="status")
]


class DataPreparationResult(Value):
    """Preparation artifacts and the measurements actually retained by extraction."""

    workers: tuple[ExtractionWorkerResult, ...] = ()
    ingestion_reused: bool | None = None
    extraction_reused: int | None = None


class ModelFitResult(Value):
    """Fit inputs and native telemetry; current reports are derived from its atoms."""

    model: GitRef
    panel: GitRef
    evidence: InferenceEvidence


class ModelSimulationResult(Value):
    """The exact generated histories retained by one simulation."""

    evidence: SimulationEvidence


class Applied[ResultT](Value):
    status: Literal["applied"] = "applied"
    result: ResultT
    effects: ActionEffects


type RejectionReason = Literal[
    "revision_conflict", "input_unavailable", "scientific_inputs", "recorded_rejection"
]


class Rejected(Value):
    status: Literal["rejected"] = "rejected"
    reason: RejectionReason
    detail: str


class Raised(Value):
    status: Literal["raised"] = "raised"
    error_type: str
    error_message: str
    # Foreign failure payloads are retained as opaque text, never scientific inputs.
    details: tuple[str, ...] = ()


type FailedOutcome = Rejected | Raised


class Attempt[ActionT: str, RequestT: Value, ResultT](Value):
    """One action's request and successful result share the same attempt owner."""

    action: ActionT
    request: RequestT | None = Field(
        description="Parsed arguments, or null for a historical attempt whose arguments were not retained"
    )
    outcome: Annotated[Applied[ResultT] | Rejected | Raised, Field(discriminator="status")]


SetQuestionAttempt = Attempt[Literal["set_question"], SetQuestionRequest, None]
EditAttempt = Attempt[Literal["edit_model"], EditModelRequest, None]
PrepareAttempt = Attempt[Literal["prepare_data"], PrepareDataRequest, DataPreparationResult]
FitAttempt = Attempt[Literal["fit"], FitRequest, ModelFitResult | None]
SimulateAttempt = Attempt[Literal["simulate"], SimulateRequest, ModelSimulationResult]
DataDiffAttempt = Attempt[Literal["data_diff"], DataDiffRequest, None]
ModelDiffAttempt = Attempt[Literal["model_diff"], ModelDiffRequest, None]


type ActionAttempt = Annotated[
    SetQuestionAttempt
    | EditAttempt
    | PrepareAttempt
    | FitAttempt
    | SimulateAttempt
    | DataDiffAttempt
    | ModelDiffAttempt,
    Field(discriminator="action"),
]


class AttemptMetadata(Value):
    seq: int
    attempt_id: UUID | None = None
    ts: str
    messages: tuple[ActionMessage, ...] = ()
    trace_ids: tuple[str, ...] = ()


class AttemptRecord(AttemptMetadata):
    """Stored inside the Git object, with no self-referential publication ID."""

    attempt: ActionAttempt

    def with_logs(self, *, messages: tuple[ActionMessage, ...], trace_ids: tuple[str, ...]) -> Self:
        return self.revised(messages=messages, trace_ids=trace_ids)


class StudyRevision(Value):
    """Git publication wraps its already-owned record, without copying its fields."""

    commit_id: GitOid
    parent_ids: tuple[GitOid, ...]
    record: AttemptRecord


class RecordDependency(Value):
    seq: int
    source_seq: int
    argument: str
    check: bool = Field(
        description="Only the action's checks read the output; its request did not name it."
    )


def failed_attempt(
    request: ScientificActionRequest | DataDiffRequest | ModelDiffRequest, outcome: FailedOutcome
) -> ActionAttempt:
    """Close the action/request relation for a rejection or execution failure."""
    match request:
        case SetQuestionRequest():
            return SetQuestionAttempt(action="set_question", request=request, outcome=outcome)
        case EditModelRequest():
            return EditAttempt(action="edit_model", request=request, outcome=outcome)
        case PrepareDataRequest():
            return PrepareAttempt(action="prepare_data", request=request, outcome=outcome)
        case FitRequest():
            return FitAttempt(action="fit", request=request, outcome=outcome)
        case SimulateRequest():
            return SimulateAttempt(action="simulate", request=request, outcome=outcome)
        case DataDiffRequest():
            return DataDiffAttempt(action="data_diff", request=request, outcome=outcome)
        case ModelDiffRequest():
            return ModelDiffAttempt(action="model_diff", request=request, outcome=outcome)
    raise TypeError("Unknown action request")


def applied_attempt[ResultT](
    request: ScientificActionRequest | DataDiffRequest | ModelDiffRequest, applied: Applied[ResultT]
) -> ActionAttempt:
    """The execution transport is closed again before publication; mismatches are bugs."""
    result = applied.result
    match request, result:
        case SetQuestionRequest(), None:
            return SetQuestionAttempt(
                action="set_question",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case EditModelRequest(), None:
            return EditAttempt(
                action="edit_model",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case PrepareDataRequest(), DataPreparationResult():
            return PrepareAttempt(
                action="prepare_data",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case FitRequest(), ModelFitResult() | None:
            return FitAttempt(
                action="fit",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case SimulateRequest(), ModelSimulationResult():
            return SimulateAttempt(
                action="simulate",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case DataDiffRequest(), None:
            return DataDiffAttempt(
                action="data_diff",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case ModelDiffRequest(), None:
            return ModelDiffAttempt(
                action="model_diff",
                request=request,
                outcome=Applied(result=result, effects=applied.effects),
            )
        case _:
            raise TypeError("Action result does not match its request")


def argument_revisions(attempt: ActionAttempt) -> tuple[tuple[str, GitOid], ...]:
    """Explicit request fields and retained input refs replace suffix scanning."""
    request = attempt.request
    match request:
        case EditModelRequest(expected_revision=revision, panel_revision=panel):
            return tuple(
                (name, oid)
                for name, oid in (("model", revision), ("panel", panel))
                if oid is not None
            )
        case FitRequest(model_revision=model, panel_revision=panel):
            return (("model", model), ("panel", panel))
        case SimulateRequest(model_revision=model, panel_revision=panel):
            return (("model", model),) + ((("panel", panel),) if panel is not None else ())
        case PrepareDataRequest(input=SimulationReplicateRef(revision=revision)):
            return (("simulation", revision),)
        case DataDiffRequest():
            return tuple(
                (name, ref.revision)
                for name, selection in (("left", request.left), ("right", request.right))
                for ref in (
                    (selection,) if isinstance(selection, (PanelRef, SimulationRef)) else selection
                )
            )
        case ModelDiffRequest(before=before, after=after):
            return (("before", before), ("after", after))
        case SetQuestionRequest() | PrepareDataRequest() | None:
            return ()


def record_dependencies(revisions: Sequence[StudyRevision]) -> list[RecordDependency]:
    """Link explicit inputs and derived artifact pins to their first producer."""
    producers: dict[GitOid, int] = {}
    for revision in revisions:
        record = revision.record
        outcome = record.attempt.outcome
        producers[revision.commit_id] = record.seq
        if not isinstance(outcome, Applied):
            continue
        for artifact in outcome.effects.produced:
            producers.setdefault(artifact.revision, record.seq)
    dependencies: list[RecordDependency] = []
    for revision in revisions:
        record = revision.record
        arguments: dict[int, str] = {}
        for name, identity in argument_revisions(record.attempt):
            source = producers.get(identity)
            if source is not None and source < record.seq:
                arguments.setdefault(source, name)
        outcome = record.attempt.outcome
        pins = (
            [
                (artifact.artifact_id, artifact_id, source)
                for artifact in outcome.effects.produced
                if producers.get(artifact.revision) == record.seq
                for artifact_id, identity in artifact.derived_from.items()
                if (source := producers.get(identity)) is not None and source < record.seq
            ]
            if isinstance(outcome, Applied)
            else []
        )
        for produced, artifact_id, source in pins:
            if artifact_id == produced:
                arguments.setdefault(source, artifact_id)
        checks: dict[int, str] = {}
        for _, artifact_id, source in pins:
            if source not in arguments:
                checks.setdefault(source, artifact_id)
        dependencies.extend(
            RecordDependency(seq=record.seq, source_seq=source, argument=name, check=False)
            for source, name in arguments.items()
        )
        dependencies.extend(
            RecordDependency(seq=record.seq, source_seq=source, argument=name, check=True)
            for source, name in checks.items()
        )
    return dependencies


class ActionBase(Value):
    """Immutable execution base, captured before an action starts."""

    commit_id: GitOid
    state: StudyState
    saved: StudyRevision | None = None


def inference_record[T: StudyRevision](records: Iterable[T], model_revision: GitOid) -> T | None:
    """Find the committed inference operation that produced this exact model value."""
    return next(
        (
            record
            for record in reversed(list(records))
            if record.record.attempt.action == "fit"
            and isinstance(record.record.attempt.outcome, Applied)
            and isinstance(record.record.attempt.outcome.result, ModelFitResult)
            and any(
                info.artifact_id == "model" and info.revision == model_revision
                for info in record.record.attempt.outcome.effects.produced
            )
        ),
        None,
    )
