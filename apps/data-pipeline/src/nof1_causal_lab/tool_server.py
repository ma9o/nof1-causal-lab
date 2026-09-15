"""Lightweight tool execution server for the refinement proxy.

Exposes pipeline tool schemas and execution over HTTP so the Next.js
refinement route can proxy LLM tool calls to the same Python validation
logic the stages use, plus the episode facade (moves via the Temporal
workflow, reads via the append-only transition log).

Run alongside the Temporal dev server and episode worker::

    cd apps/data-pipeline
    uv run uvicorn nof1_causal_lab.tool_server:app --port 8100
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import ModelRevision
from nof1_causal_lab.episode_api import (
    capabilities_router,
    machine_router,
    uploads_router,
    workspaces_router,
)
from nof1_causal_lab.episode_api import router as episode_router
from nof1_causal_lab.flows.context_tools import CONTEXT_TOOLS
from nof1_causal_lab.flows.transitions.latent_structure.grounding import latent_structure_grounding
from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
    measurement_structure_grounding,
)
from nof1_causal_lab.flows.transitions.model_spec.tool_registry import (
    execute_public_search_literature as _execute_search_literature,
)
from nof1_causal_lab.json_types import UncheckedJsonObject
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.models.causal_proofs import (
    CertifiedCausalAnalysis,
    certify_identified_estimand,
)
from nof1_causal_lab.models.ssm import SSMModel
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.dynamics import (
    posterior_dynamics_from_samples,
)
from nof1_causal_lab.models.ssm.runtime import prepare_model_runtime
from nof1_causal_lab.utils.data import data_root
from nof1_causal_lab.utils.model_structure import (
    get_constructs,
    get_manifest_indicators,
    get_model_clock,
    get_state_names,
)

logger = logging.getLogger(__name__)

type ToolImplementation = Callable[
    [UncheckedJsonObject, UncheckedJsonObject],
    UncheckedJsonObject | Awaitable[UncheckedJsonObject],
]

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.flows.contracts_base import ToolDefinition
    from nof1_causal_lab.machine.store import TransitionRecord
    from nof1_causal_lab.models.ssm.dynamics.posterior import PosteriorDynamicsSamples
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
    from nof1_causal_lab.models.ssm.runtime import PreparedModelRuntime

_API_DESCRIPTION = """\
The scientific interface has four actions: `edit_model`, `prepare_data`, `fit`,
and `simulate`. Requests commit through the serialized episode machine; reads
come from its versioned artifacts and append-only transition log.

## Scientific loop

1. Read `GET /api/machine` for action responsibilities, then
   `GET /api/episodes/{workspace_id}/model` for current model/data versions and findings.
2. Submit to `POST /api/episodes/{workspace_id}/actions`:
   - `edit_model`: `{"action":"edit_model","expected_version":0,"model":{"question":"Does workload affect sleep?"}}`.
     Model structure, measurements, mechanisms, constants, and laws can be edited together.
     Valid incomplete models are saved with applicable specification findings.
   - `prepare_data`: `{"action":"prepare_data","source":"files"}` imports uploaded sources.
     `{"action":"prepare_data","source":"raw_data","raw_data_version":1,"model_version":2}`
     extracts observations using the selected measurement definitions.
   - `fit`: `{"action":"fit","model_version":3,"panel_version":1}` conditions the selected
     model on observations. Returns joint uncertainty and fit diagnostics; predictive
     simulation is a separate request. Current fitting supports independent scalar laws.
   - `simulate`: `{"action":"simulate","model_version":3,"design":{"kind":"trajectory","times":[0,1,2,3],"draws":100,"seed":0}}`
     replicates a study from the selected model's current laws. Independent and joint
     parameter laws use the same nonlinear generator. Optional `comparison_panel_version`
     enables predictive comparisons on the matching observation grid; `design.edge_contrasts`
     adds paired edge-off experiments. Replication draws a new initial state from the
     model's initial distribution, even when the model contains fitted trajectories.
3. Read the action outcome and `GET /api/episodes/{workspace_id}/timeline` for
   `applied`, `rejected`, or `raised` records. Numerical arrays have immutable store references.
   `GET /api/episodes/{workspace_id}/model` includes separately sourced specification,
   identification, data-compatibility, fitting, and simulation findings.

The same requests are available as tools through `GET /api/tools/scientific` and
`POST /api/tools/scientific/{action}`. HTTP and tool calls share execution contracts.

## Revisions and optional recipes

Requests name stored input revisions. Fits check the selected model/data pair; model edits reject base revision conflicts. Read `/revisions` to select history and `/revisions/compare` to compare parameter decisions and recorded evidence.
An edit does not require prior simulation or an authoring admission. Causal numerical
claims still require matching identification and production inference evidence.
Simulation reports retain their own model/data versions; a later edit makes that
report historical rather than evidence for the edited model.

The `/moves` endpoint serves internal jobs; `/recipes/observational-study` runs the optional authoring recipe.
Their stage ordering is not a requirement of the scientific actions. A trajectory design selects
`initial_state` (new_study, retained, fixed, equilibrium), `state_time` or `state_values`,
process/observation noise, timed interventions and check criteria. A causal design
uses `kind: "causal"` and `query` with start, clamps, outcome and readout. It uses the same
generator and requires identification plus committed production-fit evidence.
The `analysis` context is read-only model introspection.

## Data in, results out

Upload files at `POST /api/upload` (`multipart/form-data` with `workspaceId` and
`file`) before `prepare_data` with `source=files`. Read artifact payloads at
`GET /api/episodes/{workspace_id}/artifacts/{artifact_id}`; binary files are served
from `.../files/{filename}`. Long jobs may outlive an HTTP client timeout; inspect
the timeline before submitting another request.

## Read-only deployments

`GET /api/capabilities` reports `moves_enabled`. Read-only deployments reject all
scientific action submissions and machine writes with 403.
"""

app = FastAPI(
    title="nof1-causal-lab episode API",
    description=_API_DESCRIPTION,
    docs_url="/api/tools/docs",
)

app.add_middleware(
    cast("Any", CORSMiddleware),
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(episode_router)
app.include_router(capabilities_router)
app.include_router(workspaces_router)
app.include_router(uploads_router)
app.include_router(machine_router)


def _extract_observation_timestamps(observation_data: Any) -> list[datetime]:
    import polars as pl

    if observation_data is None or observation_data.is_empty():
        return []

    anchor = pl.col("anchor_time")
    if observation_data.schema.get("anchor_time") == pl.Utf8:
        anchor = anchor.str.to_datetime(strict=False, time_zone="UTC")

    values = (
        observation_data.select(anchor.alias("anchor_time"))
        .drop_nulls()
        .unique()
        .sort("anchor_time")
        .get_column("anchor_time")
        .to_list()
    )
    out: list[datetime] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, datetime):
            dt = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            out.append(dt.astimezone(UTC))
    return out


@dataclass(frozen=True)
class _LoadedSimulation:
    """Process-local numerical inputs for one immutable fitted posterior."""

    model: ModelSpec
    inference: TransitionRecord
    runtime: PreparedModelRuntime

    @cached_property
    def draws(self) -> JointPosteriorDraws:
        from nof1_causal_lab.models.ssm.inference.persistence import model_draws

        return model_draws(self.model)

    @cached_property
    def dynamics(self) -> PosteriorDynamicsSamples:
        return posterior_dynamics_from_samples(self.model, self.draws.parameters)


@lru_cache(maxsize=2)
def _load_simulation(_data_root: str, workspace_id: str, model_version: int) -> _LoadedSimulation:
    """Reuse loaded fits; the storage root partitions local/test/remote workspaces.

    Only immutable inputs enter this cache. Current identification and freshness
    are checked separately on every request. Eviction simply reloads the fit.
    """
    from nof1_causal_lab.machine.derivations import read_model
    from nof1_causal_lab.machine.inference import inference_record
    from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal

    store = ArtifactStore(workspace_id)
    model_info = store.read_meta("model", model_version)
    record = inference_record(EpisodeJournal(workspace_id).read_all(), model_version)
    if record is None:
        raise HTTPException(404, "This model revision has no committed inference transition")
    panel_pin = model_info.derived_from["panel"]
    data_for_model = store.read_parquet_file("panel", panel_pin, parquet_filename("panel", "panel"))
    model = read_model(store, model_version)

    model.check_execution()
    runtime = prepare_model_runtime(
        data_for_model=data_for_model, model_spec=model, model=SSMModel(model)
    )
    return _LoadedSimulation(model, record, runtime)


def _build_analysis_context(workspace_id: str) -> UncheckedJsonObject:
    """Reuse pinned numerical inputs while checking current serving provenance."""
    from nof1_causal_lab.machine.moves import freshness_report
    from nof1_causal_lab.machine.snapshots import ModelReader
    from nof1_causal_lab.machine.store import ArtifactStore, derive_current_state

    state = derive_current_state(workspace_id)
    from nof1_causal_lab.machine.inference import inference_is_current

    model_info = state.get("model")
    if model_info is None:
        raise HTTPException(404, f"No fitted posterior for workspace {workspace_id}")
    if not inference_is_current(state):
        raise HTTPException(409, "Analysis requires a current conditioned model revision")
    serving_chain = (
        "model",
        "identification_report",
    )
    stale_artifacts = [
        status.artifact_id
        for status in freshness_report(state)
        if status.stale and status.artifact_id in serving_chain
    ]
    if stale_artifacts:
        raise HTTPException(
            409,
            "Analysis requires a current, identified fit; refresh stale inputs: "
            + ", ".join(stale_artifacts),
        )
    store = ArtifactStore(workspace_id)
    loaded = _load_simulation(data_root(), workspace_id, model_info.version)
    model = loaded.model
    model_revision = ModelRevision(workspace_id=workspace_id, version=model_info.version)
    identification_report_info = state.get("identification_report")
    if identification_report_info is None:
        raise HTTPException(404, f"No identification_report for workspace {workspace_id}")
    identification_report = IdentificationReport.model_validate(
        store.read_json_file(
            "identification_report",
            identification_report_info.version,
            json_filename("identification_report", "identification_report"),
        )
    )

    if identification_report.outcome is None:
        raise HTTPException(422, "The model has no identified outcome")
    outcome_name = model.get_construct(identification_report.outcome).name
    treatment_names = [
        model.get_construct(cid).name for cid in identification_report.estimable_treatments
    ]
    estimands = tuple(
        certify_identified_estimand(
            model,
            identification_report,
            model_revision=model_revision,
            treatment=treatment,
            outcome=outcome_name,
        )
        for treatment in treatment_names
    )
    causal_analysis = CertifiedCausalAnalysis(
        model=model,
        model_revision=model_revision,
        identification=identification_report,
        estimands=estimands,
        inference=loaded.inference,
    )

    simulation = ModelReader(workspace_id).simulation()
    checks = (
        simulation.value.predictive_checks
        if simulation is not None and simulation.source.validity == "fresh"
        else None
    )

    return {
        "_workspace_id": workspace_id,
        "model": model.model_dump(mode="json"),
        "identification_report": identification_report.model_dump(mode="json"),
        "inference_report": loaded.inference.diagnostics["report"],
        "predictive_checks": checks.model_dump(mode="json") if checks is not None else {},
        "_causal_analysis": causal_analysis,
        "_prepared_runtime": loaded.runtime,
        "_simulation": loaded,
        "_observation_timestamps": _extract_observation_timestamps(loaded.runtime.observation_data),
        "_outcome_name": outcome_name,
        "_identifiable_treatments": treatment_names,
    }


# ---------------------------------------------------------------------------
# Tool implementations — map (context_id, tool_name) -> execute(context, input)
# ---------------------------------------------------------------------------


def _run_compute(
    args: UncheckedJsonObject,
    param_name: str,
    compute_fn: Any,
) -> UncheckedJsonObject:
    """Parse JSON arg, run compute function, return result + context_output."""
    raw = args.get(param_name, "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"result": f"JSON parse error: {e}", "context_output": None}

    context_output, feedback = compute_fn(data)
    return {"result": feedback, "context_output": context_output}


def _execute_validate_latent_structure(
    _ctx: UncheckedJsonObject, args: UncheckedJsonObject
) -> UncheckedJsonObject:
    return _run_compute(args, "model_json", latent_structure_grounding)


def _execute_validate_measurement_structure(
    _ctx: UncheckedJsonObject, args: UncheckedJsonObject
) -> UncheckedJsonObject:
    return _run_compute(args, "model_json", measurement_structure_grounding)


def _execute_validate_extractions(
    ctx: UncheckedJsonObject, args: UncheckedJsonObject
) -> UncheckedJsonObject:
    from nof1_causal_lab.utils.llm import _validate_json_and_format
    from nof1_causal_lab.workers.schemas import validate_worker_output

    schema = ctx.get("_extraction_schema", {})
    result = _validate_json_and_format(
        args["output_json"],
        lambda data: validate_worker_output(data, schema),
    )
    return {"result": result}


def _build_model_info_payload(
    ctx: UncheckedJsonObject, args: UncheckedJsonObject
) -> UncheckedJsonObject:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    sections = list(args.get("sections") or ["overview", "variables", "capabilities"])
    focused = {str(name) for name in (args.get("names") or [])}
    model = ModelSpec.model_validate(ctx["model"])
    posterior = ctx["inference_report"]
    runtime = ctx["_prepared_runtime"]
    retained_state_names = set(get_state_names(model))
    constructs = [
        construct
        for construct in get_constructs(model)
        if construct.get("name") in retained_state_names
    ]
    indicators = get_manifest_indicators(model)

    if focused:
        constructs = [item for item in constructs if item.get("name") in focused]
        indicators = [
            item
            for item in indicators
            if item.get("name") in focused or item.get("construct_name") in focused
        ]

    payload: UncheckedJsonObject = {}
    if "overview" in sections:
        payload["overview"] = {
            "outcome": ctx.get("_outcome_name"),
            "treatments": ctx["_identifiable_treatments"],
            "n_latent": numeric.n_states(runtime.spec),
            "n_manifest": len(numeric.observation_names(runtime.spec)),
            "inference_method": (posterior.get("inference_metadata") or {}).get("method"),
            "observed_time_range": {
                "start": ctx["_observation_timestamps"][0].isoformat()
                if ctx["_observation_timestamps"]
                else None,
                "end": ctx["_observation_timestamps"][-1].isoformat()
                if ctx["_observation_timestamps"]
                else None,
            },
        }
    if "variables" in sections:
        payload["variables"] = {
            "constructs": [
                {
                    "id": item["id"],
                    "name": item.get("name"),
                    "description": item.get("description"),
                    "role": item.get("role"),
                    "temporal_status": item.get("temporal_status"),
                }
                for item in constructs
            ],
            "indicators": [
                {
                    "id": item["id"],
                    "name": item.get("name"),
                    "construct_id": item["construct_id"],
                    "construct_name": item.get("construct_name"),
                    "measurement_dtype": item.get("measurement_dtype"),
                    "support_kind": item.get("support_kind"),
                    "summary_operator": item.get("summary_operator"),
                    "observation_window": item.get("observation_window"),
                }
                for item in indicators
            ],
        }
    if "measurement" in sections:
        payload["measurement"] = {
            "model_clock": get_model_clock(model),
            "manifest_names": numeric.observation_names(runtime.spec),
        }
    if "identifiability" in sections:
        payload["identifiability"] = {
            "identifiable_treatments": ctx["_identifiable_treatments"],
            "non_identifiable_treatments": {
                identity: finding
                for identity, finding in ctx["identification_report"]["treatments"].items()
                if finding["status"] == "not_identified"
            },
        }
    if "diagnostics" in sections:
        payload["diagnostics"] = {
            "ppc_warning_count": len(
                ctx.get("predictive_checks", {}).get("per_variable_warnings", [])
            ),
        }
    if "capabilities" in sections:
        payload["capabilities"] = {
            "simulate": {
                "supported_targets": ctx["_identifiable_treatments"],
                "start": ["baseline", "abducted"],
                "clamp_modes": ["set", "shift", "ramp", "trajectory"],
                "estimands": ["end_state", "trajectory"],
                "composable": True,
            },
        }
    return payload


def _execute_get_model_info(
    ctx: UncheckedJsonObject, args: UncheckedJsonObject
) -> UncheckedJsonObject:
    return {"result": _build_model_info_payload(ctx, args)}


# Registry: (context_id, tool_name) -> implementation function
_TOOL_IMPLS: dict[tuple[str, str], ToolImplementation] = {
    ("latent-structure", "validate_latent_structure"): _execute_validate_latent_structure,
    (
        "measurement-structure",
        "validate_measurement_structure",
    ): _execute_validate_measurement_structure,
    ("measurement", "validate_extractions"): _execute_validate_extractions,
    ("statistical-model-spec", "search_literature"): _execute_search_literature,
    ("analysis", "get_model_info"): _execute_get_model_info,
}

# Upstream dependencies: which context results need to be loaded for execution.
_CONTEXT_DEPS: dict[str, list[str]] = {
    "latent-structure": [],
    "measurement-structure": ["model"],
    "measurement": [],
    "statistical-model-spec": ["model"],
    "analysis": [],
}


def _load_context_result(workspace_id: str, artifact_id: str) -> UncheckedJsonObject:
    from nof1_causal_lab.machine.store import ArtifactStore, derive_current_state

    state = derive_current_state(workspace_id)
    store = ArtifactStore(workspace_id)
    if artifact_id == "model":
        info = state.get("model")
        if info is None:
            raise HTTPException(404, f"No model for workspace {workspace_id}")
        return store.read_json_file("model", info.version, json_filename("model", "model"))
    raise KeyError(f"No canonical tool context loader for {artifact_id}")


def _build_context(workspace_id: str, context_id: str) -> UncheckedJsonObject:
    """Load upstream results needed for tool execution context."""
    if context_id == "analysis":
        return _build_analysis_context(workspace_id)
    ctx: UncheckedJsonObject = {"_workspace_id": workspace_id}
    for artifact_id in _CONTEXT_DEPS.get(context_id, []):
        ctx[artifact_id] = _load_context_result(workspace_id, artifact_id)
    return ctx


def _get_tool_contract(context_id: str, tool_name: str) -> ToolDefinition | None:
    contracts = CONTEXT_TOOLS.get(context_id) or []
    return next((contract for contract in contracts if contract.name == tool_name), None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class ToolCallRequest(BaseModel):
    workspace_id: str
    input: UncheckedJsonObject


@app.get("/api/tools/{context_id}")
def get_tool_schemas(context_id: str) -> list[UncheckedJsonObject]:
    """List a context's validation/query tools — the same tools the in-service LLM loops use.

    Each entry is `{name, description, parameters, result}` where `parameters`
    and `result` are JSON Schemas. Fetch this first to learn a tool's argument
    shape, then call `POST /api/tools/{context_id}/{tool_name}`. Examples:
    analysis `simulate` / `get_model_info`, statistical-model-spec `search_literature`.
    """
    contracts = CONTEXT_TOOLS.get(context_id)
    if contracts is None:
        raise HTTPException(404, f"No tools defined for context {context_id}")
    return [
        {
            "name": tc.name,
            "description": tc.description,
            "parameters": tc.parameters_json_schema(),
            "result": tc.result_json_schema(),
        }
        for tc in contracts
    ]


@app.post("/api/tools/{context_id}/{tool_name}")
async def execute_tool(
    context_id: str, tool_name: str, request: ToolCallRequest
) -> UncheckedJsonObject:
    """Execute a context tool against the workspace's current artifact-store versions.

    Body is `{"workspace_id": "...", "input": {...}}` where `input` matches the
    tool's `parameters` schema from `GET /api/tools/{context_id}`; 422 on a schema
    violation. Analysis tools reject stale supporting inputs with 409 before
    loading or reusing a fitted context.
    """
    contract = _get_tool_contract(context_id, tool_name)
    if contract is None:
        raise HTTPException(
            404, f"No tool contract for tool {tool_name!r} in context {context_id!r}"
        )

    if context_id == "scientific":
        from pydantic import TypeAdapter

        from nof1_causal_lab.actions.contracts import ScientificActionRequest
        from nof1_causal_lab.episode_api import execute_scientific_action

        try:
            action = TypeAdapter(ScientificActionRequest).validate_python(
                contract.input_schema.model_validate(request.input)
            )
        except ValidationError as exc:
            raise HTTPException(422, detail=exc.errors(include_context=False)) from exc
        outcome = await execute_scientific_action(request.workspace_id, action)
        return {"result": outcome.model_dump(mode="json")}

    impl = _TOOL_IMPLS.get((context_id, tool_name))
    if impl is None:
        raise HTTPException(
            404, f"No implementation for tool {tool_name!r} in context {context_id!r}"
        )

    try:
        validated_input = contract.input_schema.model_validate(request.input).model_dump(
            mode="json"
        )
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors(include_context=False)) from exc

    try:
        ctx = _build_context(request.workspace_id, context_id)
        pending_payload = impl(ctx, validated_input)
        payload = (
            cast("UncheckedJsonObject", await pending_payload)
            if inspect.isawaitable(pending_payload)
            else pending_payload
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Tool execution failed for %s/%s", context_id, tool_name)
        raise HTTPException(
            500,
            detail={
                "message": str(exc) or repr(exc),
                "exception_type": exc.__class__.__name__,
                "context_id": context_id,
                "tool_name": tool_name,
            },
        ) from exc

    if contract.output_schema is None:
        return payload

    try:
        payload["result"] = contract.output_schema.model_validate(payload.get("result")).model_dump(
            mode="json"
        )
    except ValidationError as exc:
        raise HTTPException(
            500,
            detail={
                "message": (
                    f"Tool {tool_name!r} in context {context_id!r} returned a payload "
                    "that violates its declared result contract."
                ),
                "errors": exc.errors(),
            },
        ) from exc
    return payload
