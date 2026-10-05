"""Content-named scientific calls and their slim attempt journal over HTTP."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Annotated, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Response, UploadFile
from pydantic import TypeAdapter

from nof1_causal_lab.actions.contracts import (
    EditModelRequest,
    FitRequest,
    PrepareDataRequest,
    ScientificActionRequest,
    SetQuestionRequest,
    SimulateRequest,
    call_identity,
)
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.actions.progress import read_events
from nof1_causal_lab.actions.results import ActionPoll, CompletedPoll, RunningAction, RunningPoll
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_preparation import FilePreparationSpec
from nof1_causal_lab.artifacts.identity import ActionId, GitOid
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import (
    ActionMessage,
    Applied,
    Attempt,
    RecordDependency,
    StudyRevision,
    record_dependencies,
)
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.store import (
    ArtifactStore,
    cached_read,
    read_attempt_trace,
    read_question,
)
from nof1_causal_lab.study.view_models import DataDiffRequest, ModelDiffReport, ModelDiffRequest
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.llm import LLMTrace

if TYPE_CHECKING:
    from temporalio.client import Client, WorkflowHandle

    from nof1_causal_lab.actions.temporal.workflow import StudyWorkflow
    from nof1_causal_lab.json_types import JsonObject, JsonValue

router = APIRouter(prefix="/api/studies")
workspaces_router = APIRouter(prefix="/api")
uploads_router = APIRouter(prefix="/api")

_COMPLETED_JSON = TypeAdapter(CompletedPoll)
_MODEL_DIFF_JSON = TypeAdapter(ModelDiffReport)
_ACTION_POLL_TYPE = cast("type[ActionPoll]", ActionPoll)
_CALL_PROGRESS_TYPE = cast("type[ActionPoll | None]", ActionPoll | None)


def _cached_read[T](
    workspace_id: str, key: tuple[str, ...], adapter: TypeAdapter[T], render: Callable[[], T]
) -> Response:
    body, _ = cached_read(workspace_id, key, adapter, render)
    return Response(content=body, media_type="application/json")


def actions_enabled() -> bool:
    return os.environ.get("READ_ONLY_FACADE") != "1"


def _require_actions_enabled() -> None:
    if not actions_enabled():
        raise HTTPException(403, "This read-only facade answers saved calls only")


def _safe_workspace_id(value: str) -> str:
    workspace_id = value.strip()
    if (
        not workspace_id
        or len(workspace_id) > 200
        or re.fullmatch(r"[A-Za-z0-9_-]+", workspace_id) is None
    ):
        raise HTTPException(400, "Invalid workspaceId format")
    return workspace_id


@workspaces_router.get("/workspaces", response_model=Mapping[str, str | None])
def list_workspaces(response: Response) -> Mapping[str, str | None]:
    """Available workspaces and their immutable study questions.

    X-Actions-Enabled preserves the landing page's deployment capability without a separate endpoint.
    """
    response.headers["X-Actions-Enabled"] = "true" if actions_enabled() else "false"
    workspaces = {}
    for entry in sorted(storage.listdir(data_module.data_root())):
        workspace_id = entry.rstrip("/").rsplit("/", 1)[-1]
        if not workspace_id or workspace_id.startswith("."):
            continue
        repository = StudyRepository(workspace_id)
        records = repository.attempts()
        question = next(
            (
                item
                for revision in records
                if revision.record.attempt.action == "set_question"
                and isinstance(revision.record.attempt.outcome, Applied)
                for item in revision.record.attempt.outcome.effects.produced
                if item.artifact_id == "question"
            ),
            None,
        )
        workspaces[workspace_id] = (
            read_question(ArtifactStore(workspace_id), question.revision).text
            if question is not None
            else None
        )
    return workspaces


@uploads_router.post("/upload", response_model=str)
async def upload_file(
    file: Annotated[UploadFile, File()], workspace_id: Annotated[str, Form(alias="workspaceId")]
) -> str:
    """Stage one named raw input file for prepare_data; that call captures its SHA-256."""
    _require_actions_enabled()
    workspace_id = _safe_workspace_id(workspace_id)
    filename = (file.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if not filename or filename in {".", ".."}:
        raise HTTPException(400, "Invalid file name")
    directory = data_module.input_dir(workspace_id)
    storage.makedirs(directory)
    with storage.open_file(storage.join(directory, filename), "wb") as handle:
        handle.write(await file.read())
    return f"{workspace_id}/input/{filename}"


class TemporalClientProvider:
    """One lazily connected Temporal client owned by the facade."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._client: Client | None = None

    async def get(self) -> Client:
        async with self._lock:
            if self._client is None:
                from nof1_causal_lab.actions.temporal.client import connect_client

                self._client = await connect_client()
            return self._client


def study_clients(request: Request) -> TemporalClientProvider:
    return cast("TemporalClientProvider", request.app.state.study_clients)


async def _study_handle(
    workspace_id: str, clients: TemporalClientProvider
) -> WorkflowHandle[StudyWorkflow, None]:
    from temporalio.common import WorkflowIDConflictPolicy

    from nof1_causal_lab.actions.temporal.client import STUDY_TASK_QUEUE, study_workflow_id
    from nof1_causal_lab.actions.temporal.messages import StudyInit
    from nof1_causal_lab.actions.temporal.workflow import StudyWorkflow

    return await (await clients.get()).start_workflow(
        StudyWorkflow.run,
        StudyInit(
            workspace_id=workspace_id, initial_seq=StudyRepository(workspace_id).latest_seq()
        ),
        id=study_workflow_id(workspace_id),
        task_queue=STUDY_TASK_QUEUE,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
    )


async def _running_action(
    workspace_id: str, clients: TemporalClientProvider
) -> RunningAction | None:
    from temporalio.client import WorkflowExecutionStatus
    from temporalio.service import RPCError, RPCStatusCode

    from nof1_causal_lab.actions.temporal.client import RUNNING_ACTION_MEMO, study_workflow_id

    try:
        description = (
            await (await clients.get())
            .get_workflow_handle(study_workflow_id(workspace_id))
            .describe()
        )
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            return None
        raise
    if description.status != WorkflowExecutionStatus.RUNNING:
        return None
    running = await description.memo_value(RUNNING_ACTION_MEMO, None, type_hint=RunningAction)
    if running is None:
        return None
    return running.revised(
        events=tuple(await asyncio.to_thread(read_events, workspace_id, running.attempt_id))
    )


def _file_arguments(
    workspace_id: str, request: ScientificActionRequest | DataDiffRequest | ModelDiffRequest
) -> tuple[ScientificActionRequest | DataDiffRequest | ModelDiffRequest, Mapping[str, bytes]]:
    contents: dict[str, bytes] = {}
    if (
        not isinstance(request, PrepareDataRequest)
        or not isinstance(request.input, FilePreparationSpec)
        or request.input.source.hashes
    ):
        return request, contents
    for filename in request.input.source.files:
        path = storage.join(data_module.input_dir(workspace_id), filename)
        if not storage.exists(path):
            _require_actions_enabled()
            raise HTTPException(422, f"Upload the named file before prepare_data: {filename}")
        with storage.open_file(path, "rb") as handle:
            contents[filename] = handle.read()
    hashes = {name: hashlib.sha256(body).hexdigest() for name, body in contents.items()}
    return request.revised(
        input=request.input.revised(source=request.input.source.revised(hashes=hashes))
    ), contents


def _stage_sources(
    workspace_id: str,
    request: ScientificActionRequest | DataDiffRequest | ModelDiffRequest,
    contents: Mapping[str, bytes],
) -> None:
    if not isinstance(request, PrepareDataRequest) or not isinstance(
        request.input, FilePreparationSpec
    ):
        return
    for filename, identity in request.input.source.hashes.items():
        path = (
            pathlib.Path(data_module.scratch_dir(workspace_id))
            / "source-files"
            / identity
            / filename
        )
        if path.exists():
            continue
        if filename in contents:
            body = contents[filename]
        else:
            upload = storage.join(data_module.input_dir(workspace_id), filename)
            if not storage.exists(upload):
                raise HTTPException(422, f"No retained or uploaded bytes for {filename}")
            with storage.open_file(upload, "rb") as handle:
                body = handle.read()
        if hashlib.sha256(body).hexdigest() != identity:
            raise HTTPException(422, f"Uploaded file differs from the call's SHA-256: {filename}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{filename}.{uuid4().hex}.partial")
        partial.write_bytes(body)
        partial.replace(path)


def _array_references(value: JsonValue) -> frozenset[str]:
    if isinstance(value, dict):
        own = (
            frozenset((cast("str", value["array_ref"]),))
            if "array_ref" in value
            else frozenset[str]()
        )
        return own.union(*(_array_references(item) for item in value.values()))
    if isinstance(value, list):
        return frozenset[str]().union(*(_array_references(item) for item in value))
    return frozenset[str]()


def _completed_call(workspace_id: str, revision: StudyRevision) -> CompletedPoll:
    """Materialize every retained result at the storage edge, never in a Temporal payload."""
    import numpy as np

    attempt = revision.record.attempt
    traces = {
        identity: LLMTrace.model_validate(
            read_attempt_trace(workspace_id, revision.commit_id, identity)
        )
        for identity in revision.record.trace_ids
    }
    if not isinstance(attempt.outcome, Applied):
        return CompletedPoll(
            commit_id=revision.commit_id,
            attempt=attempt,
            messages=revision.record.messages,
            traces=traces,
        )
    reader = ModelReader(workspace_id, at=revision.commit_id)
    snapshot = reader.snapshot()
    metadata = reader.data_metadata
    observation_histories = (
        {
            variable.id: history
            for variable in metadata.value.variables
            if (history := reader.observation_history(variable.id)) is not None
        }
        if metadata is not None
        else {}
    )
    predictive_overlays = {
        indicator.observation.id: overlay
        for indicator in reader.indicators()
        if (overlay := reader.predictive_history(indicator.observation.id)) is not None
    }
    simulation = reader.simulation()
    paths = (
        reader.simulation_paths(start=0, count=simulation.value.evidence.draws)
        if simulation is not None
        else None
    )
    artifacts: dict[str, JsonObject] = {}
    for info in reader.state.current.values():
        payload: dict[str, JsonValue] = {}
        for name in reader.store.filenames(info.artifact_id, info.revision):
            payload[name] = (
                reader.store.read_json_file(info.artifact_id, info.revision, name)
                if name.endswith(".json")
                else json.loads(
                    reader.store.read_parquet_file(
                        info.artifact_id, info.revision, name
                    ).write_json()
                )
            )
        artifacts[info.artifact_id] = payload
    references = _array_references(artifacts)
    if simulation is not None:
        report = simulation.value.evidence
        references = references.union(
            report.parameter_draws.values(),
            (
                report.latent_paths,
                report.observations,
                report.observation_layout.mask,
                report.observation_layout.support_start_times,
                report.observation_layout.support_end_times,
            ),
            (report.reference_latent_paths,) if report.reference_latent_paths is not None else (),
            (report.reference_observations,) if report.reference_observations is not None else (),
        )
    for record in reader.records:
        if (
            record.record.attempt.action != "fit"
            or not isinstance(record.record.attempt.outcome, Applied)
            or record.record.attempt.outcome.result is None
        ):
            continue
        evidence = record.record.attempt.outcome.result.evidence
        references = references.union(evidence.array_references)
    comparison = None
    if attempt.action == "data_diff" and attempt.request is not None:
        from nof1_causal_lab.actions.data_diff import read_data_diff

        comparison = read_data_diff(workspace_id, attempt.request)
    model_comparison = None
    if attempt.action == "model_diff" and attempt.request is not None:
        from nof1_causal_lab.actions.revisions import read_model_diff

        model_comparison = read_model_diff(
            workspace_id, attempt.request.before, attempt.request.after
        )
    arrays: dict[str, JsonValue] = {}
    for identity in sorted(references):
        values = reader.store.read_array(identity)
        arrays[identity] = np.where(
            np.isfinite(values), values, np.asarray(None, dtype=object)
        ).tolist()
    return CompletedPoll(
        commit_id=revision.commit_id,
        attempt=attempt,
        messages=revision.record.messages,
        snapshot=snapshot,
        inference_report=reader.inference_report,
        data_comparison=comparison,
        model_comparison=model_comparison,
        checks=reader.checks[0] if reader.checks is not None else None,
        observation_histories=observation_histories,
        predictive_overlays=predictive_overlays,
        simulation_paths=paths,
        parameter_draws=reader.parameter_draws(),
        traces=traces,
        artifacts=artifacts,
        arrays=arrays,
    )


def _saved_response(workspace_id: str, revision: StudyRevision) -> Response:
    return _cached_read(
        workspace_id,
        ("call-result", revision.commit_id),
        _COMPLETED_JSON,
        lambda: _completed_call(workspace_id, revision),
    )


async def _dispatch_action(
    workspace_id: str,
    body: ScientificActionRequest | DataDiffRequest | ModelDiffRequest,
    clients: TemporalClientProvider,
) -> ActionPoll | Response:
    """Applied calls reuse saved results; the serialized workflow deduplicates running calls."""
    from temporalio.client import WorkflowUpdateStage

    from nof1_causal_lab.actions.temporal.messages import ActionRequest

    workspace_id = _safe_workspace_id(workspace_id)
    if not actions_enabled() and not storage.exists(
        storage.join(data_module.study_dir(workspace_id), "history.git")
    ):
        raise HTTPException(403, "This read-only facade answers saved calls only")
    body, contents = await asyncio.to_thread(_file_arguments, workspace_id, body)
    repository = StudyRepository(workspace_id)
    saved = await asyncio.to_thread(repository.saved_call, body)
    if saved is not None:
        return await asyncio.to_thread(_saved_response, workspace_id, saved)
    _require_actions_enabled()
    identity = call_identity(body)
    handle = await _study_handle(workspace_id, clients)
    progress = cast(
        "ActionPoll | None",
        await handle.query("call_progress", identity, result_type=_CALL_PROGRESS_TYPE),
    )
    if not isinstance(progress, RunningPoll) and not (
        isinstance(progress, CompletedPoll) and isinstance(progress.attempt.outcome, Applied)
    ):
        await asyncio.to_thread(_stage_sources, workspace_id, body, contents)
        attempt_id = uuid4()
        await handle.start_update(
            "execute_action",
            ActionRequest(request=body, attempt_id=attempt_id),
            id=str(attempt_id),
            result_type=_ACTION_POLL_TYPE,
            wait_for_stage=WorkflowUpdateStage.ACCEPTED,
        )
        progress = cast(
            "ActionPoll | None",
            await handle.query("call_progress", identity, result_type=_CALL_PROGRESS_TYPE),
        )
    if progress is None:
        raise RuntimeError("Accepted call has no workflow progress")
    if isinstance(progress, RunningPoll):
        return progress.revised(
            events=tuple(await asyncio.to_thread(read_events, workspace_id, progress.attempt_id))
        )
    if progress.commit_id is not None:
        return await asyncio.to_thread(
            _saved_response, workspace_id, repository.record(progress.commit_id)
        )
    return progress


@router.post("/{workspace_id}/set_question", response_model=ActionPoll, operation_id="set_question")
async def set_question(
    workspace_id: str,
    body: SetQuestionRequest,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Set the immutable study question first. Repeat parsed arguments to read saved results or running progress; failed calls may be retried."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/edit_model", response_model=ActionPoll, operation_id="edit_model")
async def edit_model(
    workspace_id: str,
    body: EditModelRequest,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Save a model naming its base expected_revision and optional panel_revision. No head conflict check. The complete result includes findings, draws, histories, artifacts and traces; use model_diff separately for changes."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/prepare_data", response_model=ActionPoll, operation_id="prepare_data")
async def prepare_data(
    workspace_id: str,
    body: PrepareDataRequest,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Prepare named uploaded files or a saved simulation replicate. Capture each named file's SHA-256 at call time. Repeat the retained source.hashes to read saved results without uploaded bytes. Running results include step/extraction events; completed results include all observations, profiles and traces."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/fit", response_model=ActionPoll, operation_id="fit")
async def fit(
    workspace_id: str,
    body: FitRequest,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Condition the named model_revision on panel_revision. Returns the saved complete inference result, including joint posterior arrays, diagnostics and every observation history. Repeat the same arguments to read progress or the saved completion."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/simulate", response_model=ActionPoll, operation_id="simulate")
async def simulate(
    workspace_id: str,
    body: SimulateRequest,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Simulate the named model_revision and optional panel_revision. Fitted laws retain their fit origin. Returns all paths and arrays without paging, their summaries, causal evidence and traces. Repeat the same arguments to read progress or saved completion."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/data_diff", response_model=ActionPoll, operation_id="data_diff")
async def data_diff(
    workspace_id: str,
    body: DataDiffRequest,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Compare immutable left/right data selections and retain a comparison leaf. Identical applied calls reuse the complete comparison without another attempt; identical running calls return that attempt's progress. Failures stay in the timeline and can be retried."""
    return await _dispatch_action(workspace_id, body, clients)


@router.post("/{workspace_id}/model_diff", response_model=ActionPoll, operation_id="model_diff")
async def model_diff(
    workspace_id: str,
    body: ModelDiffRequest,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Compare named before/after model trees or checkpoints and retain a comparison leaf, including definitions and evidence in model_comparison. Identical applied calls reuse the comparison; running calls return progress. Failures stay in the timeline and can be retried."""
    return await _dispatch_action(workspace_id, body, clients)


@router.get("/{workspace_id}/model-comparison", response_model=ModelDiffReport)
def preview_model_comparison(workspace_id: str, before: GitOid, after: GitOid) -> Response:
    """Read the immutable comparison projection for viewer previews without submitting an action."""
    from nof1_causal_lab.actions.revisions import model_diff as compare

    workspace_id = _safe_workspace_id(workspace_id)
    if not storage.exists(storage.join(data_module.study_dir(workspace_id), "history.git")):
        raise HTTPException(404, "Unknown study workspace")
    try:
        return _cached_read(
            workspace_id,
            ("model-diff", before, after),
            _MODEL_DIFF_JSON,
            lambda: compare(workspace_id, before, after),
        )
    except StudyLookupError as exc:
        raise HTTPException(404, str(exc)) from exc


class TimelineRecord(Value):
    """A replayable call log, excluding every scientific result and check payload."""

    seq: int
    ts: str
    messages: tuple[ActionMessage, ...]
    trace_ids: tuple[str, ...]
    attempt: Attempt[ActionId, ScientificActionRequest | DataDiffRequest | ModelDiffRequest, None]


class TimelineRevision(Value):
    commit_id: GitOid
    parent_ids: tuple[GitOid, ...]
    record: TimelineRecord


class TimelineResponse(Value):
    attempts: tuple[TimelineRevision, ...]
    dependencies: tuple[RecordDependency, ...]
    running: RunningAction | None


@router.get("/{workspace_id}/timeline", response_model=TimelineResponse)
async def get_timeline(
    workspace_id: str, clients: Annotated[TemporalClientProvider, Depends(study_clients)]
) -> TimelineResponse:
    """Slim call log: replayable arguments, status, messages, errors, trace ids and dependencies. No inline results and no branches. Selecting an applied node repeats its action; never repeat failed or unknown nodes from the viewer."""
    workspace_id = _safe_workspace_id(workspace_id)
    records = await asyncio.to_thread(StudyRepository(workspace_id).attempts)
    attempts = []
    for revision in records:
        record = revision.record
        outcome = record.attempt.outcome
        summary = (
            Applied[None](
                result=None,
                effects=ActionEffects(
                    produced=outcome.effects.produced, retracted=outcome.effects.retracted
                ),
            )
            if isinstance(outcome, Applied)
            else outcome
        )
        attempts.append(
            TimelineRevision(
                commit_id=revision.commit_id,
                parent_ids=revision.parent_ids,
                record=TimelineRecord(
                    seq=record.seq,
                    ts=record.ts,
                    messages=record.messages,
                    trace_ids=record.trace_ids,
                    attempt=Attempt[
                        ActionId, ScientificActionRequest | DataDiffRequest | ModelDiffRequest, None
                    ](
                        action=record.attempt.action,
                        request=record.attempt.request,
                        outcome=summary,
                    ),
                ),
            )
        )
    return TimelineResponse(
        attempts=tuple(attempts),
        dependencies=tuple(record_dependencies(records)),
        running=await _running_action(workspace_id, clients) if actions_enabled() else None,
    )
