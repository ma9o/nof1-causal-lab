"""Scientific action execution and versioned study reads over HTTP."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pathlib
import re
from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Query, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from nof1_causal_lab.actions.contracts import ScientificActionRequest  # noqa: TC001
from nof1_causal_lab.actions.data_diff import DataDiffReport, DataDiffRequest
from nof1_causal_lab.actions.results import ActionPoll, ActionReceipt, RunningAction
from nof1_causal_lab.actions.revisions import ModelDiffReport, RevisionCatalog
from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
from nof1_causal_lab.artifacts.identity import (
    SCIENTIFIC_ACTION_IDS,
    ArtifactId,
    GitOid,
    IndicatorId,
    OperationId,
)
from nof1_causal_lab.artifacts.indicator import IndicatorSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact
from nof1_causal_lab.flows.runtime_events import RuntimeEvent, read_events
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.artifact_files import ARTIFACT_FILE_SPECS, ArtifactFileSpec
from nof1_causal_lab.machine.artifacts import ArtifactRecord  # noqa: TC001
from nof1_causal_lab.machine.execution import (
    freshness_report,
)
from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    CreationClass,
    Derivation,
    Root,
    topological_transition_order,
)
from nof1_causal_lab.machine.hierarchy import ActionSpec, ContextSpec  # noqa: TC001
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.history_models import StudyRevision
from nof1_causal_lab.machine.snapshot_models import ModelSnapshot, Sourced
from nof1_causal_lab.machine.snapshots import (
    ModelReader,
    SnapshotRevisionNotFound,
)
from nof1_causal_lab.machine.status import EpisodeStatus
from nof1_causal_lab.machine.store import (
    ArtifactStore,
    read_current_state,
    read_episode_trace,
)
from nof1_causal_lab.machine.view_models import ArtifactViewResponse
from nof1_causal_lab.machine.visual_models import (
    MechanismCurves,
    MechanismViewRequest,
    ObservationHistory,
    ParameterDraws,
    PredictiveHistory,
    SimulationPaths,
)
from nof1_causal_lab.utils.data import cache_dir
from nof1_causal_lab.utils.llm import LLMTrace

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


# A cached read is a function of immutable Git objects and of the code that renders it.
_CODE_DIGEST = hashlib.sha256(
    b"".join(path.read_bytes() for path in sorted(pathlib.Path(__file__).parent.rglob("*.py")))
).hexdigest()


def _cached_read(
    workspace_id: str, key: tuple[str, ...], adapter: TypeAdapter[Any], render: Callable[[], Any]
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
_ACTION_POLL_JSON = TypeAdapter(ActionPoll)
_INFERENCE_REPORT_JSON = TypeAdapter(Sourced[InferenceReport] | None)
_OBSERVATION_HISTORY_JSON = TypeAdapter(ObservationHistory | None)
_PREDICTIVE_HISTORY_JSON = TypeAdapter(PredictiveHistory | None)
_SIMULATION_PATHS_JSON = TypeAdapter(SimulationPaths | None)
_PARAMETER_DRAWS_JSON = TypeAdapter(ParameterDraws)
_MECHANISM_CURVES_JSON = TypeAdapter(MechanismCurves)

router = APIRouter(prefix="/api/episodes")

capabilities_router = APIRouter(prefix="/api")
workspaces_router = APIRouter(prefix="/api")
uploads_router = APIRouter(prefix="/api")


class CapabilitiesResponse(BaseModel):
    """This response tells clients whether the episode facade supports scientific actions."""

    model_config = ConfigDict(extra="forbid")

    actions_enabled: bool


class WorkspaceEntry(BaseModel):
    """A workspace entry identifies an available model workspace and its research question."""

    model_config = ConfigDict(extra="forbid")

    href: str
    question: str | None = None
    workspaceId: str


class WorkspaceList(BaseModel):
    """A workspace list provides the available model workspaces for client navigation."""

    model_config = ConfigDict(extra="forbid")

    workspaces: list[WorkspaceEntry]


class UploadResponse(BaseModel):
    """An upload response identifies the stored location of an accepted data upload."""

    model_config = ConfigDict(extra="forbid")

    path: str


class ArtifactEnvelope(BaseModel):
    """An artifact envelope delivers a stored payload with its revision and file
    list.
    """

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    artifact_id: ArtifactId
    revision: GitOid
    meta: ArtifactRecord
    payload: UncheckedJsonObject
    binary_files: list[str]


def actions_enabled() -> bool:
    """Whether this facade deployment serves scientific actions.

    A read-only facade (the hosted viewer's backend) runs the same read
    endpoints against a published store with no Temporal attached; the
    viewer reads this capability instead of being built as a fork.
    """
    return os.environ.get("EPISODE_FACADE_READ_ONLY") != "1"


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


@capabilities_router.get("/capabilities", response_model=CapabilitiesResponse)
def get_capabilities() -> CapabilitiesResponse:
    """Whether this deployment serves scientific actions.

    `actions_enabled` is `false` on the hosted read-only viewer backend, where
    every `POST` (scientific actions and study management) returns 403 and only the read
    endpoints are live.
    """
    return CapabilitiesResponse(actions_enabled=actions_enabled())


def _workspace_question(workspace_id: str) -> str | None:
    from nof1_causal_lab.machine.store import ArtifactStore, read_model

    info = read_current_state(workspace_id).get("model")
    if info is None:
        return None
    return read_model(ArtifactStore(workspace_id), info.revision).question


@workspaces_router.get("/workspaces", response_model=WorkspaceList)
def list_workspaces() -> WorkspaceList:
    """Published/local workspaces visible through this facade."""
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils import storage

    workspaces: list[WorkspaceEntry] = []
    for entry in sorted(storage.listdir(data_module.data_root())):
        workspace_id = entry.rstrip("/").rsplit("/", 1)[-1]
        if not workspace_id or workspace_id.startswith("."):
            continue
        workspaces.append(
            WorkspaceEntry(
                href=f"/v2/{workspace_id}",
                question=_workspace_question(workspace_id),
                workspaceId=workspace_id,
            )
        )
    return WorkspaceList(workspaces=workspaces)


@uploads_router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: Annotated[UploadFile, File()],
    workspace_id: Annotated[str, Form(alias="workspaceId")],
) -> UploadResponse:
    """Stage one raw input file for the raw_data transition."""
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
    return UploadResponse(path=f"{safe_workspace_id}/input/{filename}")


machine_router = APIRouter(prefix="/api")


class MachineTransition(BaseModel):
    """A transition declares the artifacts it consumes and produces and how it can run."""

    transition_id: OperationId
    consumes: list[ArtifactId]
    produces: list[ArtifactId]
    produces_optional: list[ArtifactId]
    creation_class: CreationClass


class MachineDescription(BaseModel):
    """The machine description exposes the artifact graph, storage contracts, and action hierarchy."""

    artifact_ids: list[ArtifactId]
    topological_artifact_order: list[ArtifactId]
    topological_transition_order: list[OperationId]
    contexts: list[ContextSpec]
    actions: list[ActionSpec]
    roots: list[Root]
    transitions: list[MachineTransition]
    derivations: list[Derivation]
    files: dict[ArtifactId, ArtifactFileSpec]


def machine_description() -> UncheckedJsonObject:
    """The static shape of the episode machine, independent of any workspace.

    Aggregates the artifact graph, its roots, and the action/context hierarchy
    into the single payload an agent reads once to orient. The route
    `GET /api/machine` returns exactly this; it never touches Temporal or a
    workspace store.
    """
    from nof1_causal_lab.artifacts.identity import ARTIFACT_IDS
    from nof1_causal_lab.machine.graph import (
        DERIVATIONS,
        ROOTS,
        topological_artifact_order,
    )
    from nof1_causal_lab.machine.hierarchy import describe_actions, describe_contexts

    return {
        "artifact_ids": list(ARTIFACT_IDS),
        "files": ARTIFACT_FILE_SPECS,
        "topological_artifact_order": topological_artifact_order(),
        "topological_transition_order": topological_transition_order(),
        "contexts": describe_contexts(),
        "actions": describe_actions(),
        "roots": [{"artifact_id": root.artifact_id} for root in ROOTS],
        "transitions": [
            {
                "transition_id": spec.operation_id,
                "consumes": list(spec.consumes),
                "produces": list(spec.produces),
                "produces_optional": list(spec.produces_optional),
                "creation_class": spec.creation_class,
            }
            for spec in ARTIFACT_GRAPH
        ],
        "derivations": [
            {
                "produces": spec.produces,
                "from": list(spec.from_),
                "optional": spec.optional,
            }
            for spec in DERIVATIONS
        ],
    }


@machine_router.get("/machine", response_model=MachineDescription)
def get_machine() -> UncheckedJsonObject:
    """The static artifact graph and action hierarchy — read once to orient.

    Each transition entry declares what it `consumes`, `produces`, and
    optionally co-produces (`produces_optional`), plus its **creation
    class**: `deterministic` (pure compute, no credentials), `batch_llm` (bulk
    LLM compute on the service's ambient key), or `judgment` (model-authoring
    implementation). These jobs are private; callers submit the four scientific actions.
    """
    return machine_description()


# ---------------------------------------------------------------------------
# Temporal client plumbing (actions and the running attempt; other reads never touch Temporal)
# ---------------------------------------------------------------------------

_client_lock = asyncio.Lock()
_client: Any = None


async def _get_client():
    global _client
    async with _client_lock:
        if _client is None:
            from nof1_causal_lab.machine.temporal.client import connect_client

            _client = await connect_client()
        return _client


async def _episode_handle(workspace_id: str):
    """Start-or-attach the entity workflow for a workspace."""
    from temporalio.common import WorkflowIDConflictPolicy

    from nof1_causal_lab.machine.temporal.client import (
        EPISODE_TASK_QUEUE,
        episode_workflow_id,
    )
    from nof1_causal_lab.machine.temporal.messages import EpisodeInit
    from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

    client = await _get_client()
    # Git supplies the initial query snapshot. Each action captures a branch head
    # inside the workflow before validation and execution.
    journal = StudyRepository(workspace_id)
    return await client.start_workflow(
        EpisodeWorkflow.run,
        EpisodeInit(
            workspace_id=workspace_id,
            initial_state=read_current_state(workspace_id),
            initial_seq=journal.latest_seq(),
        ),
        id=episode_workflow_id(workspace_id),
        task_queue=EPISODE_TASK_QUEUE,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
    )


async def _running_action(workspace_id: str) -> RunningAction | None:
    """The attempt the episode workflow is executing, read from its memo without a worker."""
    from temporalio.client import WorkflowExecutionStatus
    from temporalio.service import RPCError, RPCStatusCode

    from nof1_causal_lab.machine.temporal.client import RUNNING_ACTION_MEMO, episode_workflow_id

    client = await _get_client()
    try:
        description = await client.get_workflow_handle(episode_workflow_id(workspace_id)).describe()
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
    branch: str = "main",
    expected_head: GitOid | None = None,
) -> ActionReceipt:
    """Accept durable work and return its receipt; retrieve results by polling the attempt."""
    from temporalio.client import WorkflowUpdateStage

    from nof1_causal_lab.machine.temporal.messages import ActionRequest
    from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

    _require_actions_enabled()
    workspace_id = _safe_workspace_id(workspace_id)
    try:
        StudyRepository(workspace_id).head(branch)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    attempt_id = uuid4()
    handle = await _episode_handle(workspace_id)
    update = await handle.start_update(
        EpisodeWorkflow.execute_action,
        ActionRequest(
            branch=branch, expected_head=expected_head, request=body, attempt_id=attempt_id
        ),
        id=str(attempt_id),
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
async def poll_scientific_action(workspace_id: str, attempt_id: UUID) -> ActionPoll | Response:
    """Read accumulated labels and the final scientific body without dispatching work."""
    from temporalio.service import RPCError, RPCStatusCode

    from nof1_causal_lab.actions.reads import read_action_body
    from nof1_causal_lab.machine.temporal.client import episode_workflow_id
    from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

    workspace_id = _safe_workspace_id(workspace_id)
    record = StudyRepository(workspace_id).dispatched_attempt(attempt_id)
    if record is not None:
        journaled = record
        # A journaled attempt never changes; build its body off the event loop once.
        return await asyncio.to_thread(
            _cached_read,
            workspace_id,
            ("action", str(attempt_id)),
            _ACTION_POLL_JSON,
            lambda: ActionPoll(
                done=True,
                body=read_action_body(workspace_id, journaled)
                if journaled.status == "applied"
                else None,
                messages=journaled.messages,
            ),
        )
    if not actions_enabled():
        raise HTTPException(404, "Unknown action attempt")
    client = await _get_client()
    handle = client.get_workflow_handle(episode_workflow_id(workspace_id))
    try:
        progress = await handle.query(EpisodeWorkflow.action_progress, attempt_id)
    except RPCError as exc:
        if exc.status == RPCStatusCode.NOT_FOUND:
            raise HTTPException(404, "Unknown action attempt") from exc
        raise
    if progress is None:
        raise HTTPException(404, "Unknown action attempt")
    return progress


async def read_action_poll(workspace_id: str, attempt_id: UUID) -> ActionPoll:
    """Read the typed action result for in-process callers, including cached completions."""
    result = await poll_scientific_action(workspace_id, attempt_id)
    if isinstance(result, Response):
        return _ACTION_POLL_JSON.validate_json(bytes(result.body))
    return result


# ---------------------------------------------------------------------------
# Reads: transition-log-backed (no Temporal dependency)
# ---------------------------------------------------------------------------


def _episode_status(workspace_id: str, *, branch: str = "main") -> UncheckedJsonObject:
    repository = StudyRepository(workspace_id)
    commit_id = repository.head(branch)
    state = repository.state(commit_id)
    return {
        "workspace_id": workspace_id,
        "branch": branch,
        "commit_id": commit_id,
        "seq": repository.latest_seq(),
        "state": state.model_dump(mode="json"),
        "artifacts": [status.model_dump(mode="json") for status in freshness_report(state)],
        "actions": list(SCIENTIFIC_ACTION_IDS),
    }


@router.get("/{workspace_id}", response_model=EpisodeStatus)
async def get_episode(workspace_id: str, branch: str = "main") -> UncheckedJsonObject:
    """Current episode state: the single read to poll while navigating.

    Returns the four scientific action names and per-artifact existence,
    freshness and revision from the selected Git branch snapshot, and the
    attempt the episode's Temporal workflow is executing on any branch, if any.
    """
    status = await asyncio.to_thread(_episode_status, workspace_id, branch=branch)
    running = await _running_action(workspace_id) if actions_enabled() else None
    return {**status, "running": running.model_dump(mode="json") if running else None}


def model_reader(
    workspace_id: Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$", max_length=200)],
    branch: str = "main",
    at: GitOid | None = None,
) -> ModelReader:
    try:
        return ModelReader(workspace_id, at=at, branch=branch)
    except SnapshotRevisionNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{workspace_id}/model", response_model=ModelSnapshot)
def get_model_snapshot(reader: Annotated[ModelReader, Depends(model_reader)]) -> Response:
    """Batch canonical aggregates in one committed read transaction.

    Omit `at` for the selected branch head, or pass an exact Git commit ID.
    Use `context.commit_id` to pin subsequent reads. Failed attempts retain logs without advancing scientific state.
    """
    return _cached_read(
        reader.workspace_id,
        ("snapshot", reader.commit_id, reader.branch),
        _SNAPSHOT_JSON,
        reader.snapshot,
    )


@router.get("/{workspace_id}/revisions", response_model=RevisionCatalog)
def get_revisions(workspace_id: str) -> RevisionCatalog:
    """List stored model, observation and source revisions for deliberate selection."""
    records = StudyRepository(_safe_workspace_id(workspace_id)).attempts()
    committed = {
        (info.artifact_id, info.revision): info
        for record in records
        if record.status == "applied"
        for info in record.produced
    }
    return RevisionCatalog(
        **{
            field: [info for (aid, _), info in committed.items() if aid == identity]
            for field, identity in (
                ("models", "model"),
                ("panels", "panel"),
                ("raw_data", "raw_data"),
            )
        }
    )


@router.get("/{workspace_id}/revisions/model/{revision}", response_model=ModelSpec)
def read_model_revision(workspace_id: str, revision: GitOid) -> Response:
    """Read a historical definition, including the input to an earlier fit."""
    from nof1_causal_lab.machine.store import read_model

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
    except SnapshotRevisionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{workspace_id}/data-diff", response_model=DataDiffReport, operation_id="data_diff")
def post_data_diff(workspace_id: str, request: DataDiffRequest) -> DataDiffReport:
    """Compare existing datasets without creating an action, fitting or simulating.

    Each side accepts a data reference or a nonempty array of references. Panel
    references select artifact revisions; simulation references select applied
    simulation commits and optionally one replicate (otherwise every draw).
    Simulation calendar coordinates come from the saved report's origin.
    Exact anchors and measurement windows determine which predictive comparisons
    are available. Results preserve each history and report incompatible inputs.
    """
    from nof1_causal_lab.actions.data_diff import read_data_diff

    try:
        return read_data_diff(_safe_workspace_id(workspace_id), request)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=404, detail="Data revision or its recorded arrays were not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


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
def get_model_definition(reader: Annotated[ModelReader, Depends(model_reader)]):
    """The canonical scientific value selected by this journal revision."""
    return reader.fact(reader.model, "model", "") if reader.model is not None else None


@router.get(
    "/{workspace_id}/model/inference-report", response_model=Sourced[InferenceReport] | None
)
def get_model_inference_report(reader: Annotated[ModelReader, Depends(model_reader)]) -> Response:
    """Read the inference transition report associated with the selected model revision."""
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
    from nof1_causal_lab.machine.visuals import observation_history

    return _cached_read(
        reader.workspace_id,
        ("observation-history", reader.commit_id, indicator_id),
        _OBSERVATION_HISTORY_JSON,
        lambda: observation_history(reader, indicator_id),
    )


@router.get(
    "/{workspace_id}/model/visuals/predictive/{indicator_id}",
    response_model=PredictiveHistory | None,
)
def get_predictive_history(
    indicator_id: IndicatorId, reader: Annotated[ModelReader, Depends(model_reader)]
) -> Response:
    """Saved predictive paths on the exact schedule of their pinned inputs."""
    from nof1_causal_lab.machine.visuals import predictive_history

    return _cached_read(
        reader.workspace_id,
        ("predictive-history", reader.commit_id, indicator_id),
        _PREDICTIVE_HISTORY_JSON,
        lambda: predictive_history(reader, indicator_id),
    )


@router.get("/{workspace_id}/model/visuals/simulation", response_model=SimulationPaths | None)
def get_simulation_paths(
    reader: Annotated[ModelReader, Depends(model_reader)],
    start: Annotated[int, Query(ge=0)] = 0,
    count: Annotated[int, Query(ge=1, le=128)] = 24,
) -> Response:
    """A contiguous page of original simulation draws, without time thinning."""
    from nof1_causal_lab.machine.visuals import simulation_paths

    try:
        return _cached_read(
            reader.workspace_id,
            ("simulation-paths", reader.commit_id, str(start), str(count)),
            _SIMULATION_PATHS_JSON,
            lambda: simulation_paths(reader, start=start, count=count),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{workspace_id}/model/visuals/parameters", response_model=ParameterDraws)
def get_parameter_draws(reader: Annotated[ModelReader, Depends(model_reader)]) -> Response:
    """All coordinates and all draws of the retained joint posterior."""
    from nof1_causal_lab.machine.visuals import parameter_draws

    return _cached_read(
        reader.workspace_id,
        ("parameter-draws", reader.commit_id),
        _PARAMETER_DRAWS_JSON,
        lambda: parameter_draws(reader),
    )


@router.post("/{workspace_id}/model/visuals/mechanism", response_model=MechanismCurves)
def get_mechanism_curves(
    request: MechanismViewRequest, reader: Annotated[ModelReader, Depends(model_reader)]
) -> Response:
    """Read conditional drift curves using the exact model equations; creates no scientific action."""
    from nof1_causal_lab.machine.mechanism_views import mechanism_curves

    try:
        return _cached_read(
            reader.workspace_id,
            ("mechanism-curves", reader.commit_id, request.model_dump_json()),
            _MECHANISM_CURVES_JSON,
            lambda: mechanism_curves(reader, request),
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/{workspace_id}/model/constructs", response_model=tuple[ConstructSpec, ...])
def get_model_constructs(reader: Annotated[ModelReader, Depends(model_reader)]):
    """Authored constructs, using their canonical domain type."""
    return reader.constructs()


@router.get("/{workspace_id}/model/edges", response_model=tuple[CausalEdgeSpec, ...])
def get_model_edges(reader: Annotated[ModelReader, Depends(model_reader)]):
    """Authored edges, using their canonical domain type."""
    return reader.edges()


@router.get("/{workspace_id}/model/indicators", response_model=tuple[IndicatorSpec, ...])
def get_model_indicators(reader: Annotated[ModelReader, Depends(model_reader)]):
    """Authored indicators whose owners survive at the selected revision."""
    return reader.indicators()


@router.get("/{workspace_id}/model/parameters", response_model=tuple[ParameterSpec, ...])
def get_model_parameters(reader: Annotated[ModelReader, Depends(model_reader)]):
    """Scientific parameter definitions from the selected model, without inference execution."""
    return reader.parameters()


@router.get("/{workspace_id}/model/views/{artifact_id}", response_model=ArtifactViewResponse)
def get_model_view(
    workspace_id: str,
    artifact_id: str,
    branch: str = "main",
    at: GitOid | None = None,
):
    """One display projection from the selected committed model revision."""
    try:
        reader = ModelReader(workspace_id, at=at, branch=branch)
    except SnapshotRevisionNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        value = reader.artifact_view(artifact_id)
    except KeyError as exc:
        raise HTTPException(404, f"Unknown artifact view {artifact_id}") from exc
    if value is None:
        raise HTTPException(404, f"No compatible {artifact_id} view at {reader.commit_id}")
    return ArtifactViewResponse.model_validate(value)


class TimelineResponse(BaseModel):
    """Typed transition journal returned by the episode read plane."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    transitions: list[StudyRevision]
    branches: dict[str, GitOid]


@router.get("/{workspace_id}/timeline", response_model=TimelineResponse)
def get_timeline(workspace_id: str) -> TimelineResponse:
    """The transition journal: every action attempt in order.

    Each record is `applied` (state advanced), `rejected` (rejected action, state
    unchanged), or `raised` (the transition ran but threw — the record carries the
    typed error). Re-running after a `raised`/`rejected` is just proposing the
    action again.
    """
    repository = StudyRepository(workspace_id)
    records = repository.attempts()
    return TimelineResponse(
        workspace_id=workspace_id,
        transitions=records,
        branches=repository.branches(),
    )


class CreateBranchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (pygit2.AlreadyExistsError, pygit2.GitError) as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/{workspace_id}/logs/{commit_id}", response_model=StudyRevision)
def get_attempt_log(workspace_id: str, commit_id: GitOid) -> StudyRevision:
    try:
        return StudyRepository(workspace_id).record(commit_id)
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, f"No action log at {commit_id}") from exc


class EventsResponse(BaseModel):
    """An events response pages runtime telemetry without reconstructing model state."""

    workspace_id: str
    events: list[RuntimeEvent]


@router.get("/{workspace_id}/events", response_model=EventsResponse)
def get_events(
    workspace_id: str, after: str | None = None, at: GitOid | None = None
) -> UncheckedJsonObject:
    """Fine-grained telemetry (e.g. extraction worker fan-out, transition progress).

    Pass the last-seen event id as `after` to page forward; omit it for the full
    stream. This is finer-grained than the timeline, which records only whole
    action outcomes.
    """
    if at is None:
        events = read_events(workspace_id, after=after)
    else:
        repository = StudyRepository(workspace_id)
        try:
            events = json.loads(repository.read_file(at, "logs/events.json"))
        except (KeyError, ValueError) as exc:
            raise HTTPException(404, f"No captured events at {at}") from exc
        if after is not None:
            events = [event for event in events if event["cursor"] > after]
    return {"workspace_id": workspace_id, "events": events}


@router.get("/{workspace_id}/artifacts/{artifact_id}", response_model=ArtifactEnvelope)
def get_artifact(
    workspace_id: str, artifact_id: ArtifactId, revision: GitOid | None = None, branch: str = "main"
) -> ArtifactEnvelope:
    """One artifact revision: meta + inline JSON payloads.

    Defaults to the selected branch's current revision. Binary payload files (parquet, pickle) are listed by name, never
    inlined.
    """
    from nof1_causal_lab.machine.store import ArtifactStore

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
    except (KeyError, ValueError) as exc:
        raise HTTPException(404, f"No {artifact_id} tree at {revision}") from exc
    payload: UncheckedJsonObject = {}
    binary_files: list[str] = []
    for name in filenames:
        if name.endswith(".json"):
            payload[name] = store.read_json_file(artifact_id, revision, name)
        else:
            binary_files.append(name)

    return ArtifactEnvelope(
        workspace_id=workspace_id,
        artifact_id=artifact_id,
        revision=revision,
        meta=store.read_meta(artifact_id, revision),
        payload=payload,
        binary_files=sorted(binary_files),
    )


class TransitionTraceIndex(BaseModel):
    """Promoted traces identified by their committed execution sequence."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    commit_id: GitOid
    trace_ids: list[str]


@router.get("/{workspace_id}/artifacts/{artifact_id}/traces", response_model=TransitionTraceIndex)
def get_artifact_traces(
    workspace_id: str, artifact_id: ArtifactId, revision: GitOid | None = None, branch: str = "main"
) -> TransitionTraceIndex:
    """Traces of the applied transition that produced an artifact revision.

    Defaults to the episode's current revision. The join runs over the
    transition journal, so it works against a published read-only store.
    """
    if revision is None:
        info = read_current_state(workspace_id, branch=branch).get(artifact_id)
        if info is None:
            raise HTTPException(
                404, f"No current '{artifact_id}' artifact for workspace {workspace_id}"
            )
        revision = info.revision
    for record in reversed(StudyRepository(workspace_id).attempts()):
        if record.status != "applied":
            continue
        if any(
            item.artifact_id == artifact_id and item.revision == revision
            for item in record.produced
        ):
            return TransitionTraceIndex(
                workspace_id=workspace_id,
                commit_id=record.commit_id,
                trace_ids=record.trace_ids,
            )
    raise HTTPException(404, f"No applied transition produced {artifact_id} v{revision}")


@router.get("/{workspace_id}/operations/{operation_id}/traces", response_model=TransitionTraceIndex)
def get_operation_traces(
    workspace_id: str, operation_id: OperationId, branch: str = "main"
) -> TransitionTraceIndex:
    """The latest applied operation's traces, independently of later ModelSpec authorship."""
    repository = StudyRepository(workspace_id)
    for record in reversed(repository.records(repository.head(branch))):
        if record.status == "applied" and record.operation_id == operation_id:
            return TransitionTraceIndex(
                workspace_id=workspace_id,
                commit_id=record.commit_id,
                trace_ids=record.trace_ids,
            )
    raise HTTPException(404, f"No applied {operation_id} operation in {workspace_id}")


@router.get("/{workspace_id}/traces/{commit_id}/{subroutine_id}", response_model=LLMTrace)
def get_trace(workspace_id: str, commit_id: GitOid, subroutine_id: str) -> LLMTrace:
    """One trace from the owning attempt's Git commit."""
    if "/" in subroutine_id or ".." in subroutine_id:
        raise HTTPException(400, f"Invalid subroutine id {subroutine_id!r}")
    try:
        return LLMTrace.model_validate(read_episode_trace(workspace_id, commit_id, subroutine_id))
    except FileNotFoundError as exc:
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

    Defaults to the episode's current revision. Unlike the JSON artifact
    endpoint, this serves binary files as bytes and refuses undeclared
    filenames so callers cannot browse arbitrary workspace paths.
    """
    from nof1_causal_lab.machine.artifact_files import is_declared_artifact_file
    from nof1_causal_lab.machine.store import ArtifactStore
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
