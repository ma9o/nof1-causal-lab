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
    SimulateRequest,
)
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
from nof1_causal_lab.artifacts.identity import GitOid, GitRef
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.view_models import (
    DataDiffReport,
    DataDiffRequest,
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
    n_llm_calls: int | None = None
    reused: bool | None = None


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


class ModelEditResult(ActionEffects):
    """The model/check artifacts are the result; the selected base remains explicit."""

    action: Literal["edit_model"] = "edit_model"
    base: GitRef | None = None


class DataPreparationResult(ActionEffects):
    """Preparation artifacts and the measurements actually retained by extraction."""

    action: Literal["prepare_data"] = "prepare_data"
    raw_data: GitRef | None = None
    model: GitRef | None = None
    simulation_source: SimulationReplicateRef | None = None
    n_observations: int | None = None
    workers: tuple[ExtractionWorkerResult, ...] = ()
    ingestion_reused: bool | None = None
    extraction_reused: int | None = None


class ModelFitResult(ActionEffects):
    """One retained fit report, with the exact inputs and truthful retention state."""

    action: Literal["fit"] = "fit"
    model: GitRef
    panel: GitRef
    report: InferenceReport
    retention: Literal["joint", "report_only"] = "joint"

    @staticmethod
    def from_remote(payload: object) -> ModelFitResult:
        """Decode the foreign compute transport once in the target's owner."""
        return ModelFitResult.model_validate(payload)


class ModelSimulationResult(ActionEffects):
    """The report owns its model reference; the selected panel is separately pinned."""

    action: Literal["simulate"] = "simulate"
    panel: GitRef | None = None
    report: SimulationReport


class DataComparisonResult(ActionEffects):
    """A retained data comparison; it never installs scientific artifacts."""

    action: Literal["data_diff"] = "data_diff"
    report: DataDiffReport


type ActionBody = Annotated[
    ModelEditResult
    | DataPreparationResult
    | ModelFitResult
    | ModelSimulationResult
    | DataComparisonResult,
    Field(discriminator="action"),
]


class Applied[ResultT: ActionEffects](Value):
    status: Literal["applied"] = "applied"
    result: ResultT


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


class Attempt[ActionT: str, RequestT: Value, ResultT: ActionEffects](Value):
    """One action's request and successful result share the same attempt owner."""

    action: ActionT
    request: RequestT | None
    outcome: Annotated[Applied[ResultT] | Rejected | Raised, Field(discriminator="status")]


class EditAttempt(Attempt[Literal["edit_model"], EditModelRequest, ModelEditResult]):
    action: Literal["edit_model"] = "edit_model"


class PrepareAttempt(Attempt[Literal["prepare_data"], PrepareDataRequest, DataPreparationResult]):
    action: Literal["prepare_data"] = "prepare_data"


class FitAttempt(Attempt[Literal["fit"], FitRequest, ModelFitResult]):
    action: Literal["fit"] = "fit"


class SimulateAttempt(Attempt[Literal["simulate"], SimulateRequest, ModelSimulationResult]):
    action: Literal["simulate"] = "simulate"


class DataDiffAttempt(Attempt[Literal["data_diff"], DataDiffRequest, DataComparisonResult]):
    action: Literal["data_diff"] = "data_diff"


type ActionAttempt = Annotated[
    EditAttempt | PrepareAttempt | FitAttempt | SimulateAttempt | DataDiffAttempt,
    Field(discriminator="action"),
]


class AttemptMetadata(Value):
    seq: int
    attempt_id: UUID | None = None
    branch: str = "main"
    ts: str
    messages: tuple[ActionMessage, ...] = ()
    trace_ids: tuple[str, ...] = ()


class AttemptRecord(AttemptMetadata):
    """Stored inside the Git object, with no self-referential publication ID."""

    attempt: ActionAttempt

    def with_logs(self, *, messages: tuple[ActionMessage, ...], trace_ids: tuple[str, ...]) -> Self:
        return self.model_copy(update={"messages": messages, "trace_ids": trace_ids})


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
    request: ScientificActionRequest | DataDiffRequest, outcome: FailedOutcome
) -> ActionAttempt:
    """Close the action/request relation for a rejection or execution failure."""
    match request:
        case EditModelRequest():
            return EditAttempt(request=request, outcome=outcome)
        case PrepareDataRequest():
            return PrepareAttempt(request=request, outcome=outcome)
        case FitRequest():
            return FitAttempt(request=request, outcome=outcome)
        case SimulateRequest():
            return SimulateAttempt(request=request, outcome=outcome)
        case DataDiffRequest():
            return DataDiffAttempt(request=request, outcome=outcome)
    raise TypeError("Unknown action request")


def applied_attempt(
    request: ScientificActionRequest | DataDiffRequest, result: ActionBody
) -> ActionAttempt:
    """The execution transport is closed again before publication; mismatches are bugs."""
    match request, result:
        case EditModelRequest(), ModelEditResult():
            return EditAttempt(request=request, outcome=Applied(result=result))
        case PrepareDataRequest(), DataPreparationResult():
            return PrepareAttempt(request=request, outcome=Applied(result=result))
        case FitRequest(), ModelFitResult():
            return FitAttempt(request=request, outcome=Applied(result=result))
        case SimulateRequest(), ModelSimulationResult():
            return SimulateAttempt(request=request, outcome=Applied(result=result))
        case DataDiffRequest(), DataComparisonResult():
            return DataDiffAttempt(request=request, outcome=Applied(result=result))
        case _:
            raise TypeError("Action result does not match its request")


def argument_revisions(attempt: ActionAttempt) -> tuple[tuple[str, GitOid], ...]:
    """Explicit request fields and retained input refs replace suffix scanning."""
    request, outcome = attempt.request, attempt.outcome
    match request:
        case EditModelRequest(expected_revision=revision):
            return (("model", revision),) if revision is not None else ()
        case FitRequest(model_revision=model, panel_revision=panel):
            return (("model", model), ("panel", panel))
        case SimulateRequest(model_revision=model):
            return (("model", model),)
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
        case PrepareDataRequest() | None:
            pass
    if isinstance(outcome, Applied):
        match outcome.result:
            case ModelEditResult(base=base):
                return (("model", base.revision),) if base is not None else ()
            case ModelFitResult(model=model, panel=panel):
                return (("model", model.revision), ("panel", panel.revision))
            case DataPreparationResult(simulation_source=SimulationReplicateRef(revision=revision)):
                return (("simulation", revision),)
            case DataPreparationResult(raw_data=raw, model=model):
                return tuple(
                    (name, ref.revision)
                    for name, ref in (("raw_data", raw), ("model", model))
                    if ref is not None
                )
            case ModelSimulationResult(report=report):
                return (("model", report.model.revision),)
            case DataComparisonResult(report=report):
                return tuple(
                    (name, ref.revision)
                    for name, refs in (("left", report.left), ("right", report.right))
                    for ref in refs
                )
    return ()


def record_dependencies(revisions: Sequence[StudyRevision]) -> list[RecordDependency]:
    """Link explicit inputs and derived artifact pins to their first producer."""
    producers: dict[GitOid, int] = {}
    for revision in revisions:
        record = revision.record
        outcome = record.attempt.outcome
        if not isinstance(outcome, Applied):
            continue
        for artifact in outcome.result.produced:
            producers.setdefault(artifact.revision, record.seq)
        if isinstance(record.attempt, SimulateAttempt):
            producers[revision.commit_id] = record.seq
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
                for artifact in outcome.result.produced
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


class BranchBase(Value):
    """Immutable execution base, captured before an action starts."""

    commit_id: GitOid
    state: StudyState


def inference_record[T: StudyRevision](records: Iterable[T], model_revision: GitOid) -> T | None:
    """Find the committed inference operation that produced this exact model value."""
    return next(
        (
            record
            for record in reversed(list(records))
            if isinstance(record.record.attempt, FitAttempt)
            and isinstance(record.record.attempt.outcome, Applied)
            and any(
                info.artifact_id == "model" and info.revision == model_revision
                for info in record.record.attempt.outcome.result.produced
            )
        ),
        None,
    )
