"""Scientific action execution and versioned study reads over HTTP."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pathlib
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, cast
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    UploadFile,
)
from pydantic import Field, TypeAdapter

from nof1_causal_lab.actions.contracts import EditModelRequest, ScientificActionRequest
from nof1_causal_lab.actions.progress import ProgressEvent, read_events
from nof1_causal_lab.actions.results import ActionPoll, ActionReceipt, CompletedPoll, RunningAction
from nof1_causal_lab.actions.status import StudyStatus
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
from nof1_causal_lab.artifacts.identity import (
    SCIENTIFIC_ACTION_IDS,
    ArtifactId,
    GitOid,
    IndicatorId,
)
from nof1_causal_lab.artifacts.indicator import IndicatorSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.posterior_diagnostics import PPCOverlay
from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import (
    Applied,
    RecordDependency,
    StudyRevision,
    record_dependencies,
)
from nof1_causal_lab.study.snapshot_models import ModelSnapshot, Sourced
from nof1_causal_lab.study.snapshots import (
    ModelReader,
)
from nof1_causal_lab.study.state import ArtifactRecord, freshness_report
from nof1_causal_lab.study.store import (
    ArtifactStore,
    read_attempt_trace,
    read_current_state,
)
from nof1_causal_lab.study.view_models import (
    DataDiffReport,
    DataDiffRequest,
    ModelDiffReport,
)
from nof1_causal_lab.study.visual_models import (
    MechanismCurves,
    MechanismViewRequest,
    ObservationHistory,
    ParameterDraws,
    SimulationPaths,
)
from nof1_causal_lab.utils.data import cache_dir
from nof1_causal_lab.utils.llm import LLMTrace

if TYPE_CHECKING:
    from collections.abc import Callable

    from temporalio.client import Client, WorkflowHandle

    from nof1_causal_lab.actions.temporal.workflow import StudyWorkflow

logger = logging.getLogger(__name__)


# A cached read is a function of immutable Git objects and of the code that renders it.
_CODE_DIGEST = hashlib.sha256(
    b"".join(path.read_bytes() for path in sorted(pathlib.Path(__file__).parent.rglob("*.py")))
).hexdigest()


def _cached_read[T](
    workspace_id: str, key: tuple[str, ...], adapter: TypeAdapter[T], render: Callable[[], T]
) -> Response:
    """Serve a read pinned to immutable Git objects from the workspace cache tier.

    Keys name exact commits, revisions or finished attempts, and the code digest pins the
    renderer, so an entry never goes stale. All server processes share the files; a hit
    skips re-validating and re-serializing the stored artifacts.
    """
    digest = hashlib.sha256("\0".join((_CODE_DIGEST, *key)).encode()).hexdigest()
    path = pathlib.Path(cache_dir(workspace_id)) / "reads" / f"{digest}.json"
    if path.exists():
        return Response(content=path.read_bytes(), media_type="application/json")
    body = adapter.dump_json(render(), by_alias=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Processes may render one entry concurrently; the rename publishes only whole files.
    partial = path.with_name(f"{digest}.{os.getpid()}.partial")
    partial.write_bytes(body)
    partial.replace(path)
    return Response(content=body, media_type="application/json")


_SNAPSHOT_JSON = TypeAdapter(ModelSnapshot)
_MODEL_JSON = TypeAdapter(ModelSpec)
_MODEL_DIFF_JSON = TypeAdapter(ModelDiffReport)
_ACTION_POLL_JSON: TypeAdapter[ActionPoll] = TypeAdapter(ActionPoll)
_INFERENCE_REPORT_JSON = TypeAdapter(Sourced[InferenceReport] | None)
_OBSERVATION_HISTORY_JSON = TypeAdapter(ObservationHistory | None)
_PREDICTIVE_HISTORY_JSON = TypeAdapter(PPCOverlay | None)
_SIMULATION_PATHS_JSON = TypeAdapter(SimulationPaths | None)
_PARAMETER_DRAWS_JSON = TypeAdapter(ParameterDraws)
_MECHANISM_CURVES_JSON = TypeAdapter(MechanismCurves)

router = APIRouter(prefix="/api/studies")

capabilities_router = APIRouter(prefix="/api")
workspaces_router = APIRouter(prefix="/api")
uploads_router = APIRouter(prefix="/api")


class ArtifactEnvelope(Value):
    """An artifact envelope delivers a stored payload with its revision and file
    list.
    """

    workspace_id: str
    meta: ArtifactRecord
    payload: JsonObject
    binary_files: tuple[str, ...]


def actions_enabled() -> bool:
    """Whether this facade deployment serves scientific actions.

    A read-only facade (the hosted viewer's backend) runs the same read
    endpoints against a published store with no Temporal attached; the
    viewer reads this capability instead of being built as a fork.
    """
    return os.environ.get("READ_ONLY_FACADE") != "1"


def _require_actions_enabled() -> None:
    if not actions_enabled():
        raise HTTPException(
            403, "This facade is read-only: scientific actions is not deployed here"
        )


def _safe_workspace_id(value: str) -> str:
    workspace_id = value.strip()
    if (
        not workspace_id
        or len(workspace_id) > 200
        or re.fullmatch(r"[A-Za-z0-9_-]+", workspace_id) is None
    ):
        raise HTTPException(400, "Invalid workspaceId format")
    return workspace_id


@capabilities_router.get("/actions-enabled", response_model=bool)
def get_actions_enabled() -> bool:
    """Whether this deployment serves scientific actions.

    `actions_enabled` is `false` on the hosted read-only viewer backend, where
    every `POST` (scientific actions and study management) returns 403 and only the read
    endpoints are live.
    """
    return actions_enabled()


def _workspace_question(workspace_id: str) -> str | None:
    from nof1_causal_lab.study.store import ArtifactStore, read_question

    info = read_current_state(workspace_id).get("question")
    if info is None:
        return None
    return read_question(ArtifactStore(workspace_id), info.revision).text


@workspaces_router.get("/workspaces", response_model=Mapping[str, str | None])
def list_workspaces() -> Mapping[str, str | None]:
    """Published/local workspaces visible through this facade."""
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils import storage

    workspaces: dict[str, str | None] = {}
    for entry in sorted(storage.listdir(data_module.data_root())):
        workspace_id = entry.rstrip("/").rsplit("/", 1)[-1]
        if not workspace_id or workspace_id.startswith("."):
            continue
        workspaces[workspace_id] = _workspace_question(workspace_id)
    return workspaces


@uploads_router.post("/upload", response_model=str)
async def upload_file(
    file: Annotated[UploadFile, File()],
    workspace_id: Annotated[str, Form(alias="workspaceId")],
) -> str:
    """Stage one raw input file for prepare_data."""
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils import storage

    _require_actions_enabled()
    safe_workspace_id = _safe_workspace_id(workspace_id)
    filename = (file.filename or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if not filename:
        raise HTTPException(400, "Invalid file name")

    upload_dir = data_module.input_dir(safe_workspace_id)
    storage.makedirs(upload_dir)
    path = storage.join(upload_dir, filename)
    with storage.open_file(path, "wb") as handle:
        handle.write(await file.read())
    return f"{safe_workspace_id}/input/{filename}"


# ---------------------------------------------------------------------------
# Temporal client plumbing (actions and the running attempt; other reads never touch Temporal)
# ---------------------------------------------------------------------------


class TemporalClientProvider:
    """One lazily connected Temporal client owned by a facade application."""

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
    """Resolve the request application's connection owner."""
    return cast("TemporalClientProvider", request.app.state.study_clients)


async def _study_handle(
    workspace_id: str, clients: TemporalClientProvider
) -> WorkflowHandle[StudyWorkflow, None]:
    """Start-or-attach the study workflow for a workspace."""
    from temporalio.common import WorkflowIDConflictPolicy

    from nof1_causal_lab.actions.temporal.client import (
        STUDY_TASK_QUEUE,
        study_workflow_id,
    )
    from nof1_causal_lab.actions.temporal.messages import StudyInit
    from nof1_causal_lab.actions.temporal.workflow import StudyWorkflow

    client = await clients.get()
    # Each action captures its branch head inside the workflow before validation and execution.
    return await client.start_workflow(
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
    """The attempt the study workflow is executing, read from its memo without a worker."""
    from temporalio.client import WorkflowExecutionStatus
    from temporalio.service import RPCError, RPCStatusCode

    from nof1_causal_lab.actions.temporal.client import RUNNING_ACTION_MEMO, study_workflow_id

    client = await clients.get()
    try:
        description = await client.get_workflow_handle(study_workflow_id(workspace_id)).describe()
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            return None
        raise
    # A terminated or failed workflow never finishes its attempt; only a live one executes.
    if description.status != WorkflowExecutionStatus.RUNNING:
        return None
    return await description.memo_value(RUNNING_ACTION_MEMO, None, type_hint=RunningAction)


# ---------------------------------------------------------------------------
# Request/response bodies
# ---------------------------------------------------------------------------


@router.post("/{workspace_id}/actions", response_model=ActionReceipt, status_code=202)
async def execute_scientific_action(
    workspace_id: str,
    body: ScientificActionRequest,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
    branch: str = "main",
    expected_head: GitOid | None = None,
) -> ActionReceipt:
    """Accept durable work and return its receipt; retrieve results by polling the attempt."""
    return await _dispatch_action(workspace_id, body, clients, branch, expected_head)


def _require_question_fit(workspace_id: str, head: GitOid, model: ModelSpec) -> None:
    """An edit that doesn't fit the study question is an invalid request, like a bad model."""
    from nof1_causal_lab.actions.edit_model import question_edit_reason
    from nof1_causal_lab.study.store import ArtifactStore, read_question

    state = StudyRepository(workspace_id).state(head)
    # Before set_question, the workflow's lineage gate refuses every edit.
    if not state.has("question"):
        return
    question = read_question(ArtifactStore(workspace_id), state.current["question"].revision)
    if (reason := question_edit_reason(model, question)) is not None:
        raise HTTPException(422, reason)


async def _dispatch_action(
    workspace_id: str,
    body: ScientificActionRequest | DataDiffRequest,
    clients: TemporalClientProvider,
    branch: str,
    expected_head: GitOid | None = None,
) -> ActionReceipt:
    from temporalio.client import WorkflowUpdateStage

    from nof1_causal_lab.actions.temporal.messages import ActionRequest

    _require_actions_enabled()
    workspace_id = _safe_workspace_id(workspace_id)
    try:
        head = StudyRepository(workspace_id).head(branch)
    except StudyLookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    if isinstance(body, EditModelRequest):
        _require_question_fit(workspace_id, head, body.model)
    attempt_id = uuid4()
    handle = await _study_handle(workspace_id, clients)
    update = await handle.start_update(
        "execute_action",
        ActionRequest(
            branch=branch,
            expected_head=head if isinstance(body, DataDiffRequest) else expected_head,
            request=body,
            attempt_id=attempt_id,
        ),
        id=str(attempt_id),
        result_type=ActionReceipt,
        wait_for_stage=WorkflowUpdateStage.ACCEPTED,
    )
    # Temporal can return a rejected update at ACCEPTED without raising. The SDK
    # exposes no public nonblocking outcome accessor; result() would wait for an
    # accepted action to finish. Only inspect the outcome returned by this RPC.
    outcome = update._known_outcome
    if outcome is not None and outcome.HasField("failure"):
        failure = outcome.failure
        reasons = [failure.message]
        while failure.HasField("cause"):
            failure = failure.cause
            reasons.append(failure.message)
        detail = "Workflow update rejected: " + ": ".join(reasons)
        logger.error("Action %s for workspace %s: %s", attempt_id, workspace_id, detail)
        raise HTTPException(500, detail)
    return ActionReceipt(attempt_id=attempt_id)


@router.get("/{workspace_id}/actions/{attempt_id}", response_model=ActionPoll)
async def poll_scientific_action(
    workspace_id: str,
    attempt_id: UUID,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
) -> ActionPoll | Response:
    """Read progress or the completed attempt's typed outcome without dispatching work."""
    from temporalio.service import RPCError, RPCStatusCode

    from nof1_causal_lab.actions.temporal.client import study_workflow_id
    from nof1_causal_lab.actions.temporal.workflow import StudyWorkflow

    workspace_id = _safe_workspace_id(workspace_id)
    record = StudyRepository(workspace_id).dispatched_attempt(attempt_id)
    if record is not None:
        journaled = record
        # A journaled attempt never changes; render its owned outcome off the event loop once.
        return await asyncio.to_thread(
            _cached_read,
            workspace_id,
            ("action", str(attempt_id)),
            _ACTION_POLL_JSON,
            lambda: CompletedPoll(
                commit_id=journaled.commit_id,
                attempt=journaled.record.attempt,
                messages=journaled.record.messages,
            ),
        )
    if not actions_enabled():
        raise HTTPException(404, "Unknown action attempt")
    client = await clients.get()
    handle = client.get_workflow_handle(study_workflow_id(workspace_id))
    try:
        progress = await handle.query(StudyWorkflow.action_progress, attempt_id)
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            raise HTTPException(404, "Unknown action attempt") from exc
        raise
    if progress is None:
        raise HTTPException(404, "Unknown action attempt")
    return progress


async def read_action_poll(
    workspace_id: str, attempt_id: UUID, clients: TemporalClientProvider
) -> ActionPoll:
    """Read the typed action result for in-process callers, including cached completions."""
    result = await poll_scientific_action(workspace_id, attempt_id, clients)
    if isinstance(result, Response):
        return _ACTION_POLL_JSON.validate_json(bytes(result.body))
    return result


# ---------------------------------------------------------------------------
# Reads: attempt-log-backed (no Temporal dependency)
# ---------------------------------------------------------------------------


def _study_status(
    workspace_id: str, *, branch: str = "main", running: RunningAction | None = None
) -> StudyStatus:
    repository = StudyRepository(workspace_id)
    commit_id = repository.head(branch)
    state = repository.state(commit_id)
    return StudyStatus(
        workspace_id=workspace_id,
        branch=branch,
        commit_id=commit_id,
        seq=repository.latest_seq(),
        state=state,
        artifacts=tuple(freshness_report(state)),
        actions=SCIENTIFIC_ACTION_IDS,
        running=running,
    )


@router.get("/{workspace_id}", response_model=StudyStatus)
async def get_study(
    workspace_id: str,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
    branch: str = "main",
) -> StudyStatus:
    """Current study state: the single read to poll while navigating.

    Returns the five scientific action names and per-artifact existence,
    freshness and revision from the selected Git branch snapshot, and the
    attempt the study's Temporal workflow is executing on any branch, if any.
    """
    running = await _running_action(workspace_id, clients) if actions_enabled() else None
    return await asyncio.to_thread(_study_status, workspace_id, branch=branch, running=running)


def model_reader(
    workspace_id: Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$", max_length=200)],
    branch: str = "main",
    at: GitOid | None = None,
) -> ModelReader:
    try:
        return ModelReader(workspace_id, at=at, branch=branch)
    except StudyLookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{workspace_id}/model", response_model=ModelSnapshot)
def get_model_snapshot(reader: Annotated[ModelReader, Depends(model_reader)]) -> Response:
    """Batch canonical aggregates in one committed read transaction.

    Omit `at` for the selected branch head, or pass an exact Git commit ID.
    Use `commit_id` to pin subsequent reads. Failed attempts retain logs without advancing scientific state.
    """
    return _cached_read(
        reader.workspace_id,
        ("snapshot", reader.commit_id, reader.branch),
        _SNAPSHOT_JSON,
        reader.snapshot,
    )


@router.get("/{workspace_id}/revisions", response_model=tuple[ArtifactRecord, ...])
def get_revisions(workspace_id: str) -> tuple[ArtifactRecord, ...]:
    """List stored model, observation and source revisions for deliberate selection."""
    records = StudyRepository(_safe_workspace_id(workspace_id)).attempts()
    committed = {
        (info.artifact_id, info.revision): info
        for record in records
        if isinstance(record.record.attempt.outcome, Applied)
        for info in record.record.attempt.outcome.effects.produced
    }
    return tuple(
        info for (aid, _), info in committed.items() if aid in {"model", "panel", "raw_data"}
    )


@router.get("/{workspace_id}/revisions/model/{revision}", response_model=ModelSpec)
def read_model_revision(workspace_id: str, revision: GitOid) -> Response:
    """Read a historical definition, including the input to an earlier fit."""
    from nof1_causal_lab.study.store import read_model

    workspace_id = _safe_workspace_id(workspace_id)
    return _cached_read(
        workspace_id,
        ("model-revision", revision),
        _MODEL_JSON,
        lambda: read_model(ArtifactStore(workspace_id), revision),
    )


@router.get("/{workspace_id}/model-diff", response_model=ModelDiffReport, operation_id="model_diff")
def get_model_diff(
    workspace_id: str,
    before: GitOid,
    after: GitOid,
) -> Response:
    """Compare two model artifact revisions or Git checkpoints containing a model.

    Returns identity-aligned definition changes, parameter decisions and graph
    topology differences. Graph highlights exclude laws and other entity attributes.
    Checkpoint selections also include their recorded fit/simulation
    evidence; selecting a model tree alone does not infer an associated run.
    """
    from nof1_causal_lab.actions.revisions import model_diff

    workspace_id = _safe_workspace_id(workspace_id)
    try:
        return _cached_read(
            workspace_id,
            ("model-diff", before, after),
            _MODEL_DIFF_JSON,
            lambda: model_diff(workspace_id, before, after),
        )
    except StudyLookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/{workspace_id}/data-diff",
    response_model=ActionReceipt,
    status_code=202,
    operation_id="data_diff",
)
async def post_data_diff(
    workspace_id: str,
    request: DataDiffRequest,
    clients: Annotated[TemporalClientProvider, Depends(study_clients)],
    branch: str = "main",
) -> ActionReceipt:
    """Record a comparison as a read-only leaf off the branch head captured at dispatch.

    Each side accepts a data reference or a nonempty array of references. Panel
    references select artifact revisions; simulation references select applied
    simulation commits and optionally one replicate (otherwise every draw).
    Simulation calendar coordinates come from the saved report's origin.
    Exact anchors and measurement windows determine which predictive comparisons
    are available. Results preserve each history and report incompatible inputs.
    Poll the returned attempt_id for the report and its commit_id. The study's
    serialized writer saves the report beside the journal record; it never moves
    the branch or changes scientific state. Failed comparisons also leave a leaf.
    GET /data-diff/{commit_id} reads a saved report without running the comparison.
    """
    return await _dispatch_action(workspace_id, request, clients, branch)


@router.get("/{workspace_id}/data-diff/{commit_id}", response_model=DataDiffReport)
def get_data_diff(workspace_id: str, commit_id: GitOid) -> Response:
    """Read the comparison report retained by its applied outcome."""
    record = StudyRepository(_safe_workspace_id(workspace_id)).record(commit_id).record
    if record.attempt.action != "data_diff" or not isinstance(record.attempt.outcome, Applied):
        raise HTTPException(404, "No comparison report recorded at this commit")
    return Response(
        content=record.attempt.outcome.result.report.model_dump_json(),
        media_type="application/json",
    )


@router.get(
    "/{workspace_id}/revisions/data-profile/{panel_revision}", response_model=DataProfileArtifact
)
def read_data_profile(workspace_id: str, panel_revision: GitOid) -> DataProfileArtifact:
    """Read the empirical profile for an observation revision independently of the model."""
    store = ArtifactStore(_safe_workspace_id(workspace_id))
    for revision in reversed(store.list_revisions("data_profile")):
        if store.read_meta("data_profile", revision).derived_from["panel"] == panel_revision:
            return DataProfileArtifact.model_validate(
                store.read_json_file("data_profile", revision, "data_profile.json")
            )
    raise HTTPException(404, "This observation revision has no recorded data profile")


@router.get("/{workspace_id}/model/definition", response_model=Sourced[ModelSpec] | None)
def get_model_definition(
    reader: Annotated[ModelReader, Depends(model_reader)],
) -> Sourced[ModelSpec] | None:
    """The canonical scientific value selected by this journal revision."""
    return reader.fact(reader.model, "model", "") if reader.model is not None else None


@router.get(
    "/{workspace_id}/model/inference-report", response_model=Sourced[InferenceReport] | None
)
def get_model_inference_report(reader: Annotated[ModelReader, Depends(model_reader)]) -> Response:
    """Read the fit report associated with the selected model revision."""
    return _cached_read(
        reader.workspace_id,
        ("inference-report", reader.commit_id, reader.branch),
        _INFERENCE_REPORT_JSON,
        lambda: reader.inference_report,
    )


@router.get(
    "/{workspace_id}/model/visuals/observations/{indicator_id}",
    response_model=ObservationHistory | None,
)
def get_observation_history(
    indicator_id: IndicatorId, reader: Annotated[ModelReader, Depends(model_reader)]
) -> Response:
    """All prepared observations on their recorded temporal support."""
    return _cached_read(
        reader.workspace_id,
        ("observation-history", reader.commit_id, indicator_id),
        _OBSERVATION_HISTORY_JSON,
        lambda: reader.observation_history(indicator_id),
    )


@router.get(
    "/{workspace_id}/model/visuals/predictive/{indicator_id}",
    response_model=PPCOverlay | None,
)
def get_predictive_history(
    indicator_id: IndicatorId, reader: Annotated[ModelReader, Depends(model_reader)]
) -> Response:
    """Saved predictive paths on the exact schedule of their pinned inputs."""
    return _cached_read(
        reader.workspace_id,
        ("predictive-history", reader.commit_id, indicator_id),
        _PREDICTIVE_HISTORY_JSON,
        lambda: reader.predictive_history(indicator_id),
    )


@router.get("/{workspace_id}/model/visuals/simulation", response_model=SimulationPaths | None)
def get_simulation_paths(
    reader: Annotated[ModelReader, Depends(model_reader)],
    start: Annotated[int, Query(ge=0)] = 0,
    count: Annotated[int, Query(ge=1, le=128)] = 24,
) -> Response:
    """A contiguous page of original simulation draws, without time thinning."""
    try:
        return _cached_read(
            reader.workspace_id,
            ("simulation-paths", reader.commit_id, str(start), str(count)),
            _SIMULATION_PATHS_JSON,
            lambda: reader.simulation_paths(start=start, count=count),
        )
    except StudyLookupError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{workspace_id}/model/visuals/parameters", response_model=ParameterDraws)
def get_parameter_draws(reader: Annotated[ModelReader, Depends(model_reader)]) -> Response:
    """All coordinates and all draws of the retained joint posterior."""
    return _cached_read(
        reader.workspace_id,
        ("parameter-draws", reader.commit_id),
        _PARAMETER_DRAWS_JSON,
        lambda: reader.parameter_draws(),
    )


@router.post("/{workspace_id}/model/visuals/mechanism", response_model=MechanismCurves)
def get_mechanism_curves(
    request: MechanismViewRequest, reader: Annotated[ModelReader, Depends(model_reader)]
) -> Response:
    """Read conditional drift curves using the exact model equations; creates no scientific action."""
    try:
        return _cached_read(
            reader.workspace_id,
            ("mechanism-curves", reader.commit_id, request.model_dump_json()),
            _MECHANISM_CURVES_JSON,
            lambda: reader.mechanism_curves(request),
        )
    except StudyLookupError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{workspace_id}/model/constructs", response_model=tuple[ConstructSpec, ...])
def get_model_constructs(
    reader: Annotated[ModelReader, Depends(model_reader)],
) -> tuple[ConstructSpec, ...]:
    """Authored constructs, using their canonical domain type."""
    return reader.constructs()


@router.get("/{workspace_id}/model/edges", response_model=tuple[CausalEdgeSpec, ...])
def get_model_edges(
    reader: Annotated[ModelReader, Depends(model_reader)],
) -> tuple[CausalEdgeSpec, ...]:
    """Authored edges, using their canonical domain type."""
    return reader.edges()


@router.get("/{workspace_id}/model/indicators", response_model=tuple[IndicatorSpec, ...])
def get_model_indicators(
    reader: Annotated[ModelReader, Depends(model_reader)],
) -> tuple[IndicatorSpec, ...]:
    """Authored indicators whose owners survive at the selected revision."""
    return reader.indicators()


@router.get("/{workspace_id}/model/parameters", response_model=tuple[ParameterSpec, ...])
def get_model_parameters(
    reader: Annotated[ModelReader, Depends(model_reader)],
) -> tuple[ParameterSpec, ...]:
    """Scientific parameter definitions from the selected model, without inference execution."""
    return reader.parameters()


class TimelineResponse(Value):
    """Typed attempt journal returned by the study read plane."""

    workspace_id: str
    attempts: tuple[StudyRevision, ...]
    branches: Mapping[str, GitOid]
    dependencies: tuple[RecordDependency, ...]


@router.get("/{workspace_id}/timeline", response_model=TimelineResponse)
def get_timeline(workspace_id: str) -> TimelineResponse:
    """The attempt journal: every action attempt in order.

    Each record is `applied` (completed; data_diff leaves do not advance state),
    `rejected` (rejected action, state unchanged), or `raised` (the action ran but threw — the record carries the
    typed error). Re-running after a `raised`/`rejected` is just proposing the
    action again. `dependencies` links each record to the earlier records whose
    outputs its request named; `check` marks outputs only its checks read.
    """
    repository = StudyRepository(workspace_id)
    records = repository.attempts()
    return TimelineResponse(
        workspace_id=workspace_id,
        attempts=tuple(records),
        branches=repository.branches(),
        dependencies=tuple(record_dependencies(records)),
    )


class CreateBranchBody(Value):
    name: str = Field(min_length=1)
    at: GitOid


@router.get("/{workspace_id}/branches", response_model=dict[str, GitOid])
def get_branches(workspace_id: str) -> dict[str, GitOid]:
    return StudyRepository(workspace_id).branches()


@router.post("/{workspace_id}/branches", response_model=GitOid)
def create_branch(workspace_id: str, body: CreateBranchBody) -> GitOid:
    """Fork the complete study at a checkpoint; the new branch shares its ancestry."""
    import pygit2

    _require_actions_enabled()
    try:
        return StudyRepository(_safe_workspace_id(workspace_id)).create_branch(
            body.name, at=body.at
        )
    except StudyLookupError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (pygit2.AlreadyExistsError, pygit2.GitError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{workspace_id}/logs/{commit_id}", response_model=StudyRevision)
def get_attempt_log(workspace_id: str, commit_id: GitOid) -> StudyRevision:
    try:
        return StudyRepository(workspace_id).record(commit_id)
    except StudyLookupError as exc:
        raise HTTPException(404, f"No action log at {commit_id}") from exc


@router.get("/{workspace_id}/events", response_model=tuple[ProgressEvent, ...])
def get_events(
    workspace_id: str, attempt_id: UUID, after: str | None = None
) -> tuple[ProgressEvent, ...]:
    """Live progress of one attempt: data-preparation step status and extraction fan-out.

    Pass the last-seen event cursor as `after` to page forward. Progress is disposable
    and never saved with the attempt; its record and traces are authoritative.
    """
    return tuple(read_events(_safe_workspace_id(workspace_id), attempt_id, after=after))


@router.get("/{workspace_id}/artifacts/{artifact_id}", response_model=ArtifactEnvelope)
def get_artifact(
    workspace_id: str, artifact_id: ArtifactId, revision: GitOid | None = None, branch: str = "main"
) -> ArtifactEnvelope:
    """One artifact revision: meta + inline JSON payloads.

    Defaults to the selected branch's current revision. Binary payload files (parquet, pickle) are listed by name, never
    inlined.
    """
    from nof1_causal_lab.study.store import ArtifactStore

    store = ArtifactStore(workspace_id)
    if revision is None:
        info = read_current_state(workspace_id, branch=branch).get(artifact_id)
        if info is None:
            raise HTTPException(
                404, f"No current '{artifact_id}' artifact for workspace {workspace_id}"
            )
        revision = info.revision

    try:
        filenames = store.filenames(artifact_id, revision)
    except StudyLookupError as exc:
        raise HTTPException(404, f"No {artifact_id} tree at {revision}") from exc
    payload: JsonObject = {}
    binary_files: list[str] = []
    for name in filenames:
        if name.endswith(".json"):
            payload[name] = store.read_json_file(artifact_id, revision, name)
        else:
            binary_files.append(name)

    return ArtifactEnvelope(
        workspace_id=workspace_id,
        meta=store.read_meta(artifact_id, revision),
        payload=payload,
        binary_files=tuple(sorted(binary_files)),
    )


class AttemptTraceIndex(Value):
    """Promoted traces identified by their committed execution sequence."""

    commit_id: GitOid
    trace_ids: tuple[str, ...]


@router.get("/{workspace_id}/artifacts/{artifact_id}/traces", response_model=AttemptTraceIndex)
def get_artifact_traces(
    workspace_id: str, artifact_id: ArtifactId, revision: GitOid | None = None, branch: str = "main"
) -> AttemptTraceIndex:
    """Traces of the applied attempt that produced an artifact revision.

    Defaults to the study's current revision. The join runs over the
    attempt journal, so it works against a published read-only store.
    """
    if revision is None:
        info = read_current_state(workspace_id, branch=branch).get(artifact_id)
        if info is None:
            raise HTTPException(
                404, f"No current '{artifact_id}' artifact for workspace {workspace_id}"
            )
        revision = info.revision
    for record in reversed(StudyRepository(workspace_id).attempts()):
        if not isinstance(record.record.attempt.outcome, Applied):
            continue
        if any(
            item.artifact_id == artifact_id and item.revision == revision
            for item in record.record.attempt.outcome.effects.produced
        ):
            return AttemptTraceIndex(
                commit_id=record.commit_id,
                trace_ids=record.record.trace_ids,
            )
    raise HTTPException(404, f"No applied attempt produced {artifact_id} v{revision}")


@router.get("/{workspace_id}/traces/{commit_id}/{subroutine_id}", response_model=LLMTrace)
def get_trace(workspace_id: str, commit_id: GitOid, subroutine_id: str) -> LLMTrace:
    """One trace from the owning attempt's Git commit."""
    if "/" in subroutine_id or ".." in subroutine_id:
        raise HTTPException(400, f"Invalid subroutine id {subroutine_id!r}")
    try:
        return LLMTrace.model_validate(read_attempt_trace(workspace_id, commit_id, subroutine_id))
    except StudyLookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{workspace_id}/artifacts/{artifact_id}/files/{filename}")
def get_artifact_file(
    workspace_id: str,
    artifact_id: ArtifactId,
    filename: str,
    revision: GitOid | None = None,
    branch: str = "main",
) -> Response:
    """One declared payload file from an artifact revision.

    Defaults to the study's current revision. Unlike the JSON artifact
    endpoint, this serves binary files as bytes and refuses undeclared
    filenames so callers cannot browse arbitrary workspace paths.
    """
    from nof1_causal_lab.study.artifact_files import is_declared_artifact_file
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.utils import storage

    if "/" in filename or not is_declared_artifact_file(artifact_id, filename):
        raise HTTPException(404, f"{filename} is not a declared file for {artifact_id}")

    store = ArtifactStore(workspace_id)
    if revision is None:
        info = read_current_state(workspace_id, branch=branch).get(artifact_id)
        if info is None:
            raise HTTPException(
                404, f"No current '{artifact_id}' artifact for workspace {workspace_id}"
            )
        revision = info.revision

    path = store.file_path(artifact_id, revision, filename)
    if not storage.exists(path):
        raise HTTPException(404, f"{artifact_id} v{revision}/{filename} does not exist")

    with storage.open_file(path, "rb") as handle:
        data = handle.read()
    return Response(content=data, media_type="application/octet-stream")
