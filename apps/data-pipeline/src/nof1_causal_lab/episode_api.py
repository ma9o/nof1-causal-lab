"""Episode facade: HTTP surface over the state machine.

Mounted into the tool server so the web app has a single Python backend.
Reads derive current state by replaying the append-only transition log and
serve timeline/events from their execution logs without touching Temporal.
Moves go through the episode workflow's ``propose`` update, which validates,
executes, and records outcomes durably.

The named observational-study recipe is an optional navigation policy — run enabled
transitions in dependency order while their outputs are missing or stale —
giving the web "run the pipeline" parity as one background driver. An
LLM navigator replaces this policy by calling ``moves`` directly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Path, Query, Response, UploadFile
from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.actions.contracts import ScientificActionRequest  # noqa: TC001
from nof1_causal_lab.actions.revisions import ModelComparison, RevisionCatalog
from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
from nof1_causal_lab.artifacts.identity import ArtifactId, OperationId  # noqa: TC001
from nof1_causal_lab.artifacts.indicator import IndicatorSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact
from nof1_causal_lab.flows.runtime_events import RuntimeEvent, read_events
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo  # noqa: TC001
from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    topological_transition_order,
)
from nof1_causal_lab.utils.llm import LLMTrace

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.graph import Transition
from nof1_causal_lab.machine.artifact_files import ARTIFACT_FILE_SPECS, ArtifactFileSpec
from nof1_causal_lab.machine.graph import CreationClass, Derivation, Root
from nof1_causal_lab.machine.hierarchy import ActionSpec, ContextSpec  # noqa: TC001
from nof1_causal_lab.machine.moves import (
    ExecOptions,
    Move,
    RunOperation,
    WriteArtifact,
    freshness_report,
    is_stale,
    legal_moves,
    validate_move,
)
from nof1_causal_lab.machine.snapshot_models import ModelSnapshot, Sourced
from nof1_causal_lab.machine.snapshots import (
    ModelReader,
    SnapshotRevisionNotFound,
    read_revision,
)
from nof1_causal_lab.machine.status import EpisodeStatus, MoveOutcome
from nof1_causal_lab.machine.store import (
    ArtifactStore,
    EpisodeJournal,
    TransitionRecord,
    derive_current_state,
    read_episode_trace,
    replay_state,
)
from nof1_causal_lab.machine.view_models import ArtifactViewResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/episodes")

capabilities_router = APIRouter(prefix="/api")
workspaces_router = APIRouter(prefix="/api")
uploads_router = APIRouter(prefix="/api")


class CapabilitiesResponse(BaseModel):
    """This response tells clients whether the episode facade supports model-changing moves."""

    model_config = ConfigDict(extra="forbid")

    moves_enabled: bool


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
    """An artifact envelope delivers a stored payload with its version, provenance, and file
    list.
    """

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    artifact_id: ArtifactId
    version: int
    meta: ArtifactVersionInfo
    payload: UncheckedJsonObject
    binary_files: list[str]


def moves_enabled() -> bool:
    """Whether this facade deployment serves the move plane.

    A read-only facade (the hosted viewer's backend) runs the same read
    endpoints against a published store with no Temporal attached; the
    viewer reads this capability instead of being built as a fork.
    """
    return os.environ.get("EPISODE_FACADE_READ_ONLY") != "1"


def _require_moves_enabled() -> None:
    if not moves_enabled():
        raise HTTPException(403, "This facade is read-only: the move plane is not deployed here")


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
    """Whether this deployment serves the move plane.

    `moves_enabled` is `false` on the hosted read-only viewer backend, where
    every `POST` (moves, auto-run, start-episode) returns 403 and only the read
    endpoints are live.
    """
    return CapabilitiesResponse(moves_enabled=moves_enabled())


def _workspace_question(workspace_id: str) -> str | None:
    from nof1_causal_lab.machine.derivations import read_model
    from nof1_causal_lab.machine.store import ArtifactStore

    info = derive_current_state(workspace_id).get("model")
    if info is None:
        return None
    return read_model(ArtifactStore(workspace_id), info.version).question


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
                href=f"/model/{workspace_id}",
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

    _require_moves_enabled()
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
    writable: bool


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
        ARTIFACT_GRAPH,
        DERIVATIONS,
        ROOTS,
        topological_artifact_order,
        topological_transition_order,
    )
    from nof1_causal_lab.machine.hierarchy import describe_actions, describe_contexts

    return {
        "artifact_ids": list(ARTIFACT_IDS),
        "files": ARTIFACT_FILE_SPECS,
        "topological_artifact_order": topological_artifact_order(),
        "topological_transition_order": topological_transition_order(),
        "contexts": describe_contexts(),
        "actions": describe_actions(),
        "roots": [
            {"artifact_id": root.artifact_id, "write_pins": list(root.write_pins)} for root in ROOTS
        ],
        "transitions": [
            {
                "transition_id": spec.operation_id,
                "consumes": list(spec.consumes),
                "produces": list(spec.produces),
                "produces_optional": list(spec.produces_optional),
                "creation_class": spec.creation_class,
                "writable": spec.writable,
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
    LLM compute on the service's ambient key — you trigger it with a `run` move,
    you never supply a key), or `judgment` (proposal work you can author yourself
    by writing the produced artifact directly — these are flagged `writable`).
    """
    return machine_description()


# ---------------------------------------------------------------------------
# Temporal client plumbing (moves only; reads never touch Temporal)
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
    # Reconstruct the resume seed from committed transition effects. On a fresh
    # start this rehydrates a workflow Temporal lost (empty for a new episode);
    # on attach (USE_EXISTING) the live workflow keeps its own state and the
    # seed is ignored.
    journal = EpisodeJournal(workspace_id)
    return await client.start_workflow(
        EpisodeWorkflow.run,
        EpisodeInit(
            workspace_id=workspace_id,
            initial_state=derive_current_state(workspace_id),
            initial_seq=journal.latest_seq(),
        ),
        id=episode_workflow_id(workspace_id),
        task_queue=EPISODE_TASK_QUEUE,
        id_conflict_policy=WorkflowIDConflictPolicy.USE_EXISTING,
    )


async def _propose(workspace_id: str, request_body: MoveBody) -> UncheckedJsonObject:
    from nof1_causal_lab.machine.temporal.messages import MoveRequest
    from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow

    handle = await _episode_handle(workspace_id)
    outcome = await handle.execute_update(
        EpisodeWorkflow.propose,
        MoveRequest(
            move=request_body.move,
            payload=request_body.payload,
            options=request_body.options,
        ),
    )
    return outcome.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Request/response bodies
# ---------------------------------------------------------------------------


class StartEpisodeBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    question: str | None = None


class MoveBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    move: Move
    payload: UncheckedJsonObject | None = None
    options: ExecOptions = Field(default_factory=ExecOptions)


class AutoRunBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    options: ExecOptions = Field(default_factory=ExecOptions)


@router.post("/{workspace_id}/actions", response_model=MoveOutcome)
async def execute_scientific_action(
    workspace_id: str, body: ScientificActionRequest
) -> MoveOutcome:
    """Execute edit_model, prepare_data, fit, or simulate with explicit input revisions."""
    from nof1_causal_lab.actions.commands import action_command

    _require_moves_enabled()
    command = action_command(body)
    result = await _propose(
        _safe_workspace_id(workspace_id),
        MoveBody(move=command.move, payload=command.payload, options=command.options),
    )
    return MoveOutcome.model_validate(result)


# ---------------------------------------------------------------------------
# Reads: transition-log-backed (no Temporal dependency)
# ---------------------------------------------------------------------------


def _episode_status(workspace_id: str) -> UncheckedJsonObject:
    journal = EpisodeJournal(workspace_id)
    records = journal.read_all()
    state = replay_state(records)
    next_move = _next_auto_move(workspace_id, state)
    return {
        "workspace_id": workspace_id,
        "seq": records[-1].seq if records else 0,
        "state": state.model_dump(mode="json"),
        "artifacts": [status.model_dump(mode="json") for status in freshness_report(state)],
        "legal": [move.model_dump(mode="json") for move in legal_moves(state)],
        "next_operation": next_move.operation_id if next_move else None,
        "auto_running": workspace_id in _AUTO_DRIVERS,
    }


@router.get("/{workspace_id}", response_model=EpisodeStatus)
def get_episode(workspace_id: str) -> UncheckedJsonObject:
    """Current episode state: the single read to poll while navigating.

    Returns per-artifact freshness (existence, staleness, version, provenance),
    the `legal` moves available right now, and `auto_running` — whether the
    background driver is active. Replayed from the append-only transition log,
    so it works even against a published read-only store.
    """
    return _episode_status(workspace_id)


def model_reader(
    workspace_id: Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]+$", max_length=200)],
    at_seq: Annotated[int | None, Query(ge=0)] = None,
) -> ModelReader:
    try:
        return ModelReader(workspace_id, at_seq=at_seq)
    except SnapshotRevisionNotFound as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{workspace_id}/model", response_model=ModelSnapshot)
def get_model_snapshot(reader: Annotated[ModelReader, Depends(model_reader)]) -> ModelSnapshot:
    """Batch canonical aggregates in one committed read transaction.

    Omit `at_seq` for the latest applied move, or select a committed journal sequence.
    Zero selects the empty model. Rejected/raised attempts are not revisions (404).
    Use the returned `context.seq` for subsequent aggregate or collection reads at the same revision.
    """
    return reader.snapshot()


@router.get("/{workspace_id}/revisions", response_model=RevisionCatalog)
def get_revisions(workspace_id: str) -> RevisionCatalog:
    """List stored model, observation and source revisions for deliberate selection."""
    store = ArtifactStore(_safe_workspace_id(workspace_id))
    return RevisionCatalog(
        **{
            field: [store.read_meta(identity, v) for v in store.list_versions(identity)]
            for field, identity in (
                ("models", "model"),
                ("panels", "panel"),
                ("raw_data", "raw_data"),
            )
        }
    )


@router.get("/{workspace_id}/revisions/model/{version}", response_model=ModelSpec)
def read_model_revision(workspace_id: str, version: Annotated[int, Path(ge=1)]) -> ModelSpec:
    """Read a historical definition, including the input to an earlier fit."""
    from nof1_causal_lab.machine.derivations import read_model

    return read_model(ArtifactStore(_safe_workspace_id(workspace_id)), version)


@router.get("/{workspace_id}/revisions/compare", response_model=ModelComparison)
def compare_model_revisions(
    workspace_id: str, before: Annotated[int, Query(ge=1)], after: Annotated[int, Query(ge=1)]
) -> ModelComparison:
    """Compare fixed/free decisions, laws and scientific dependencies in the backend."""
    from nof1_causal_lab.actions.revisions import compare_models

    return compare_models(ArtifactStore(_safe_workspace_id(workspace_id)), before, after)


@router.get(
    "/{workspace_id}/revisions/data-profile/{panel_version}", response_model=DataProfileArtifact
)
def read_data_profile(
    workspace_id: str, panel_version: Annotated[int, Path(ge=1)]
) -> DataProfileArtifact:
    """Read the empirical profile for an observation revision independently of the model."""
    store = ArtifactStore(_safe_workspace_id(workspace_id))
    for version in reversed(store.list_versions("data_profile")):
        if store.read_meta("data_profile", version).derived_from["panel"] == panel_version:
            return DataProfileArtifact.model_validate(
                store.read_json_file("data_profile", version, "data_profile.json")
            )
    raise HTTPException(404, "This observation revision has no recorded data profile")


@router.get("/{workspace_id}/model/definition", response_model=Sourced[ModelSpec] | None)
def get_model_definition(reader: Annotated[ModelReader, Depends(model_reader)]):
    """The canonical scientific value selected by this journal revision."""
    return reader.fact(reader.model, "model", "") if reader.model is not None else None


class ModelUpdateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=0)
    model: ModelSpec


@router.put("/{workspace_id}/model", response_model=ModelSnapshot)
async def update_model(workspace_id: str, body: ModelUpdateBody) -> ModelSnapshot:
    """Validate and atomically replace the named base model revision."""
    _require_moves_enabled()
    result = await _propose(
        workspace_id,
        MoveBody(
            move=WriteArtifact(artifact_id="model", expected_model_version=body.expected_version),
            payload=body.model.model_dump(mode="json"),
        ),
    )
    if result["status"] != "applied":
        message = result.get("reason") or result.get("error_message") or result["status"]
        raise HTTPException(409 if "conflict" in message.lower() else 422, message)
    return ModelReader(workspace_id, at_seq=result["seq"]).snapshot()


@router.get(
    "/{workspace_id}/model/inference-report", response_model=Sourced[InferenceReport] | None
)
def get_model_inference_report(reader: Annotated[ModelReader, Depends(model_reader)]):
    """Read the inference transition report associated with the selected model revision."""
    return reader.inference_report()


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
    at_seq: Annotated[int | None, Query(ge=0)] = None,
):
    """One display projection from the selected committed model revision."""
    from nof1_causal_lab.machine.store import ArtifactStore
    from nof1_causal_lab.machine.views import read_artifact_views

    try:
        seq, state, _, _ = read_revision(workspace_id, at_seq)
    except SnapshotRevisionNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    views = read_artifact_views(ArtifactStore(workspace_id), state)
    if artifact_id not in type(views).model_fields:
        raise HTTPException(404, f"Unknown artifact view {artifact_id}")
    value = getattr(views, artifact_id)
    if value is None:
        raise HTTPException(404, f"No compatible {artifact_id} view at revision {seq}")
    return ArtifactViewResponse.model_validate(value)


class TimelineResponse(BaseModel):
    """Typed transition journal returned by the episode read plane."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    transitions: list[TransitionRecord]


@router.get("/{workspace_id}/timeline", response_model=TimelineResponse)
def get_timeline(workspace_id: str) -> TimelineResponse:
    """The transition journal: every move attempt in order.

    Each record is `applied` (state advanced), `rejected` (illegal move, state
    unchanged), or `raised` (the transition ran but threw — the record carries the
    typed error). Re-running after a `raised`/`rejected` is just proposing the
    move again.
    """
    records = EpisodeJournal(workspace_id).read_all()
    return TimelineResponse(workspace_id=workspace_id, transitions=records)


class EventsResponse(BaseModel):
    """An events response pages runtime telemetry without reconstructing model state."""

    workspace_id: str
    events: list[RuntimeEvent]


@router.get("/{workspace_id}/events", response_model=EventsResponse)
def get_events(workspace_id: str, after: str | None = None) -> UncheckedJsonObject:
    """Fine-grained telemetry (e.g. extraction worker fan-out, transition progress).

    Pass the last-seen event id as `after` to page forward; omit it for the full
    stream. This is finer-grained than the timeline, which records only whole
    move outcomes.
    """
    return {
        "workspace_id": workspace_id,
        "events": read_events(workspace_id, after=after),
    }


@router.get("/{workspace_id}/artifacts/{artifact_id}", response_model=ArtifactEnvelope)
def get_artifact(
    workspace_id: str, artifact_id: ArtifactId, version: int | None = None
) -> ArtifactEnvelope:
    """One artifact version: meta + inline JSON payloads.

    Defaults to the episode's *current* version from replayed applied transition
    effects. Binary payload files (parquet, pickle) are listed by name, never
    inlined.
    """
    from nof1_causal_lab.machine.store import ArtifactStore
    from nof1_causal_lab.utils import storage

    store = ArtifactStore(workspace_id)
    if version is None:
        info = derive_current_state(workspace_id).get(artifact_id)
        if info is None:
            raise HTTPException(
                404, f"No current '{artifact_id}' artifact for workspace {workspace_id}"
            )
        version = info.version

    version_dir = store.version_dir(artifact_id, version)
    if not storage.exists(storage.join(version_dir, "meta.json")):
        raise HTTPException(404, f"{artifact_id} v{version} does not exist for {workspace_id}")

    payload: UncheckedJsonObject = {}
    binary_files: list[str] = []
    for entry in storage.listdir(version_dir):
        name = entry.rstrip("/").rsplit("/", 1)[-1]
        if name == "meta.json":
            continue
        if name.endswith(".json"):
            payload[name] = store.read_json_file(artifact_id, version, name)
        else:
            binary_files.append(name)

    return ArtifactEnvelope(
        workspace_id=workspace_id,
        artifact_id=artifact_id,
        version=version,
        meta=store.read_meta(artifact_id, version),
        payload=payload,
        binary_files=sorted(binary_files),
    )


class TransitionTraceIndex(BaseModel):
    """Promoted traces identified by their committed execution sequence."""

    model_config = ConfigDict(extra="forbid")

    workspace_id: str
    seq: int
    trace_ids: list[str]


@router.get("/{workspace_id}/artifacts/{artifact_id}/traces", response_model=TransitionTraceIndex)
def get_artifact_traces(
    workspace_id: str, artifact_id: ArtifactId, version: int | None = None
) -> TransitionTraceIndex:
    """Traces of the applied transition that produced an artifact version.

    Defaults to the episode's current version. The join runs over the
    transition journal, so it works against a published read-only store.
    """
    if version is None:
        info = derive_current_state(workspace_id).get(artifact_id)
        if info is None:
            raise HTTPException(
                404, f"No current '{artifact_id}' artifact for workspace {workspace_id}"
            )
        version = info.version
    for record in reversed(EpisodeJournal(workspace_id).read_all()):
        if record.status != "applied":
            continue
        if any(
            item.artifact_id == artifact_id and item.version == version for item in record.produced
        ):
            return TransitionTraceIndex(
                workspace_id=workspace_id,
                seq=record.seq,
                trace_ids=record.trace_ids,
            )
    raise HTTPException(404, f"No applied transition produced {artifact_id} v{version}")


@router.get("/{workspace_id}/operations/{operation_id}/traces", response_model=TransitionTraceIndex)
def get_operation_traces(workspace_id: str, operation_id: OperationId) -> TransitionTraceIndex:
    """The latest applied operation's traces, independently of later ModelSpec authorship."""
    for record in reversed(EpisodeJournal(workspace_id).read_all()):
        if (
            record.status == "applied"
            and isinstance(record.move, RunOperation)
            and record.move.operation_id == operation_id
        ):
            return TransitionTraceIndex(
                workspace_id=workspace_id,
                seq=record.seq,
                trace_ids=record.trace_ids,
            )
    raise HTTPException(404, f"No applied {operation_id} operation in {workspace_id}")


@router.get("/{workspace_id}/traces/{seq}/{subroutine_id}", response_model=LLMTrace)
def get_trace(workspace_id: str, seq: int, subroutine_id: str) -> LLMTrace:
    """One promoted LLM trace from ``episode/traces/{seq:06d}/{subroutine_id}.json``."""
    if "/" in subroutine_id or ".." in subroutine_id:
        raise HTTPException(400, f"Invalid subroutine id {subroutine_id!r}")
    try:
        return LLMTrace.model_validate(read_episode_trace(workspace_id, seq, subroutine_id))
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{workspace_id}/artifacts/{artifact_id}/files/{filename}")
def get_artifact_file(
    workspace_id: str,
    artifact_id: ArtifactId,
    filename: str,
    version: int | None = None,
) -> Response:
    """One declared payload file from an artifact version.

    Defaults to the episode's current version. Unlike the JSON artifact
    endpoint, this serves binary files as bytes and refuses undeclared
    filenames so callers cannot browse arbitrary workspace paths.
    """
    from nof1_causal_lab.machine.artifact_files import is_declared_artifact_file
    from nof1_causal_lab.machine.store import ArtifactStore
    from nof1_causal_lab.utils import storage

    if "/" in filename or not is_declared_artifact_file(artifact_id, filename):
        raise HTTPException(404, f"{filename} is not a declared file for {artifact_id}")

    store = ArtifactStore(workspace_id)
    if version is None:
        info = derive_current_state(workspace_id).get(artifact_id)
        if info is None:
            raise HTTPException(
                404, f"No current '{artifact_id}' artifact for workspace {workspace_id}"
            )
        version = info.version

    path = store.file_path(artifact_id, version, filename)
    if not storage.exists(path):
        raise HTTPException(404, f"{artifact_id} v{version}/{filename} does not exist")

    with storage.open_file(path, "rb") as handle:
        data = handle.read()
    return Response(content=data, media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------


class StartEpisodeResponse(EpisodeStatus):
    """Starting an episode returns its current status and any model-write outcome."""

    ok: Literal[True] = True
    outcome: MoveOutcome | None


class AutoRunResponse(BaseModel):
    """An auto-run acknowledgement identifies the active background episode driver."""

    ok: Literal[True] = True
    auto_running: Literal[True] = True
    workspace_id: str


@router.post("", response_model=StartEpisodeResponse)
async def start_episode(body: StartEpisodeBody) -> UncheckedJsonObject:
    """Ensure the episode workflow exists; optionally author its model's question.

    Idempotent: attaches to an existing episode or starts a fresh one. Passing
    `question` creates or revises the Model with `human` provenance.
    Upload raw data at `POST /api/upload` before running the `raw_data`
    transition. Returns the same shape as
    `GET /api/episodes/{id}`.
    """
    _require_moves_enabled()
    await _episode_handle(body.workspace_id)
    outcome = None
    if body.question is not None:
        from nof1_causal_lab.machine.derivations import read_model
        from nof1_causal_lab.machine.store import ArtifactStore

        current = derive_current_state(body.workspace_id).get("model")
        payload = (
            read_model(ArtifactStore(body.workspace_id), current.version).model_dump(mode="json")
            if current is not None
            else {}
        )
        payload["question"] = body.question
        outcome = await _propose(
            body.workspace_id,
            MoveBody(
                move=WriteArtifact(
                    artifact_id="model",
                    provenance="human",
                    expected_model_version=current.version if current is not None else 0,
                ),
                payload=payload,
            ),
        )
    return {"ok": True, "outcome": outcome, **_episode_status(body.workspace_id)}


@router.post("/{workspace_id}/moves", response_model=MoveOutcome)
async def propose_move(workspace_id: str, body: MoveBody) -> UncheckedJsonObject:
    """Propose one move; blocks until it is applied, rejected, or raises.

    Two kinds:

    - Run a transition: `{"move": {"kind": "run", "operation_id": "latent_structure"}}`.
    - Author a judgment artifact directly (skip the in-service stage):
      `{"move": {"kind": "write", "artifact_id": "model", "expected_model_version": 0, "provenance":
      "llm"}, "payload": {...}}`. The payload is schema-validated against that
      artifact's contract, journaled, and provenance-stamped; the write becomes a
      revision. Consumers retain their original pins; changed scientific inputs invalidate affected results.

    The synchronous outcome is the same record the timeline stores. Long transitions
    (statistical model specification, posterior — minutes to hours) can outlive a client timeout; for
    those prefer `POST /api/episodes/{workspace_id}/recipes/observational-study` plus polling.
    """
    _require_moves_enabled()
    return await _propose(workspace_id, body)


# ---------------------------------------------------------------------------
# Default navigation policy (auto-run)
# ---------------------------------------------------------------------------

_AUTO_DRIVERS: dict[str, asyncio.Task[None]] = {}


def _needs_run(
    state: EpisodeState,
    spec: Transition,
    model: ModelSpec | None,
    extraction: TransitionRecord | None = None,
    *,
    specification_current: bool = False,
) -> bool:
    """Missing required outputs, or any existing output gone stale.

    An *absent optional* output with a completed execution for the same inputs
    is a standing negative finding, not a reason to rerun — otherwise the driver would loop on
    transitions whose finding was legitimately empty.
    """
    from nof1_causal_lab.machine.inference import inference_is_current

    if spec.operation_id in {"posterior", "statistical_model_spec"}:
        from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError

        if inference_is_current(state):
            return False
        if model is None:
            return spec.operation_id == "statistical_model_spec"
        try:
            model.check_execution()
        except (IncompleteModelError, AggregatedCompileError):
            return spec.operation_id == "statistical_model_spec"
        return spec.operation_id == "posterior" or not specification_current
    if spec.operation_id == "latent_structure":
        return model is None or not model.constructs
    if spec.operation_id == "measurement_structure":
        return model is None or model.measurement_clock is None or not model.indicators
    if spec.operation_id == "measurements" and not state.has("panel"):
        if extraction is None:
            return True
        # A completed empty run is a durable finding about these extraction inputs.
        pins = extraction.diagnostics["input_pins"]
        return (
            any(pins[aid] != state.current[aid].version for aid in spec.consumes if aid != "model")
            or extraction.diagnostics["model_input"]
            != state.current["model"].model_inputs["extraction"]
        )
    if any(not state.has(output) for output in spec.produces):
        return True
    return any(is_stale(state, artifact) for artifact in spec.all_produces if state.has(artifact))


def _next_auto_move(workspace_id: str, state: EpisodeState) -> RunOperation | None:
    from nof1_causal_lab.machine.derivations import read_model
    from nof1_causal_lab.machine.model_spec_results import model_spec_is_current, model_spec_record
    from nof1_causal_lab.machine.store import ArtifactStore

    store = ArtifactStore(workspace_id)
    model = read_model(store, state.current["model"].version) if state.has("model") else None
    records = EpisodeJournal(workspace_id).read_all()
    specification = model_spec_record(records)
    specification_current = specification is not None and model_spec_is_current(
        specification, state, store
    )
    extraction = next(
        (
            record
            for record in reversed(records)
            if record.status == "applied"
            and isinstance(record.move, RunOperation)
            and record.move.operation_id == "measurements"
        ),
        None,
    )
    specs = {spec.operation_id: spec for spec in ARTIFACT_GRAPH}
    for artifact_id in topological_transition_order():
        if artifact_id == "simulate":
            continue  # Prediction requires the agent's explicit design and requested experiment.
        spec = specs[artifact_id]
        move = RunOperation(operation_id=artifact_id)
        if validate_move(state, move) is None and _needs_run(
            state, spec, model, extraction, specification_current=specification_current
        ):
            return move
    return None


async def _auto_drive(workspace_id: str, options: ExecOptions) -> None:
    try:
        while True:
            state = derive_current_state(workspace_id)
            move = _next_auto_move(workspace_id, state)
            if move is None:
                logger.info("auto-run %s: quiescent", workspace_id)
                return
            logger.info("auto-run %s: %s", workspace_id, move.operation_id)
            outcome = await _propose(workspace_id, MoveBody(move=move, options=options))
            if outcome["status"] != "applied":
                logger.warning(
                    "auto-run %s stopped: %s %s (%s)",
                    workspace_id,
                    move.operation_id,
                    outcome["status"],
                    outcome.get("error_type") or outcome.get("reason"),
                )
                return
    finally:
        _AUTO_DRIVERS.pop(workspace_id, None)


@router.post("/{workspace_id}/recipes/observational-study", response_model=AutoRunResponse)
async def auto_run(workspace_id: str, body: AutoRunBody) -> UncheckedJsonObject:
    """Start the default navigation policy in the background.

    Runs enabled stages in dependency order while their outputs are missing or
    stale, stopping when quiescent or when a move fails. Returns immediately;
    follow progress with `GET /api/episodes/{workspace_id}` (`auto_running`) and
    the timeline. An LLM navigator replaces this policy by proposing `moves`
    itself. 409 if a driver is already active for this workspace.
    """
    _require_moves_enabled()
    if workspace_id in _AUTO_DRIVERS:
        raise HTTPException(409, f"auto-run already active for {workspace_id}")
    await _episode_handle(workspace_id)  # fail fast if Temporal is down
    _AUTO_DRIVERS[workspace_id] = asyncio.create_task(_auto_drive(workspace_id, body.options))
    return {"ok": True, "auto_running": True, "workspace_id": workspace_id}
