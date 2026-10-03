"""Lightweight tool execution server for the refinement proxy.

Exposes pipeline tool schemas and execution over HTTP so the Next.js
refinement route can proxy LLM tool calls to the same Python validation
logic the stages use, plus the study facade (actions via the Temporal
workflow, reads via the append-only attempt log).

Run alongside the Temporal dev server and study worker::

    cd apps/data-pipeline
    uv run uvicorn nof1_causal_lab.tool_server:app --port 8100
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cached_property, lru_cache, partial
from typing import TYPE_CHECKING, NotRequired, TypedDict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic.json_schema import JsonSchemaValue

from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorPredictiveChecks
from nof1_causal_lab.json_types import JsonObject, JsonValue
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel, compile_executable_model
from nof1_causal_lab.models.ssm.dynamics import (
    dynamics_from_samples,
)
from nof1_causal_lab.models.ssm.runtime import project_observation_data
from nof1_causal_lab.study.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.study_api import (
    TemporalClientProvider,
    capabilities_router,
    study_clients,
    uploads_router,
    workspaces_router,
)
from nof1_causal_lab.study_api import router as study_router
from nof1_causal_lab.tool_contracts import (
    CONTEXT_TOOLS,
    GetModelInfoInput,
    SearchLiteratureInput,
)
from nof1_causal_lab.utils.data import data_root
from nof1_causal_lab.utils.model_structure import (
    get_constructs,
    get_model_clock,
)

logger = logging.getLogger(__name__)


class ToolContext(TypedDict):
    _workspace_id: str


class AnalysisToolContext(ToolContext):
    model: ModelSpec
    identification_report: IdentificationReport
    inference_report: InferenceReport
    predictive_checks: PosteriorPredictiveChecks | None
    _simulation: _LoadedSimulation
    _observation_timestamps: list[datetime]
    _outcome_name: str
    _identifiable_treatments: list[str]


class ToolResult(TypedDict):
    result: JsonValue
    context_output: NotRequired[JsonObject | None]


class ToolSchema(TypedDict):
    name: str
    description: str
    parameters: JsonSchemaValue
    result: JsonSchemaValue | None


type ToolImplementation = Callable[[str, JsonObject], ToolResult | Awaitable[ToolResult]]


async def execute_search_literature(_ctx: object, args: SearchLiteratureInput) -> ToolResult:
    """Search Exa for empirical literature about effect sizes."""
    from nof1_causal_lab.workers.prior_research import search_parameter_literature
    from nof1_causal_lab.workers.prompts.prior_research import format_literature_for_parameter

    if not args.query:
        return {"result": "Error: query is required"}
    sources = await search_parameter_literature(args.query)
    if not sources:
        return {"result": "No relevant literature found for this query."}
    return {"result": format_literature_for_parameter(sources)}


def _parse_tool_call[Input: BaseModel, Context: ToolContext](
    schema: type[Input],
    implementation: Callable[[Context, Input], ToolResult | Awaitable[ToolResult]],
    context_builder: Callable[[str], Context],
    workspace_id: str,
    arguments: JsonObject,
) -> ToolResult | Awaitable[ToolResult]:
    return implementation(context_builder(workspace_id), schema.model_validate(arguments))


if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.actions.tool_definition import ToolDefinition
    from nof1_causal_lab.models.ssm.dynamics.draws import DynamicsDraws
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
    from nof1_causal_lab.study.records import StudyRevision

_API_DESCRIPTION = """\
The scientific interface has five actions: `set_question`, `edit_model`, `prepare_data`,
`fit`, and `simulate`. Requests commit through the serialized study workflow; reads
come from its versioned artifacts and append-only attempt log.

## Scientific loop

1. Read `GET /api/studies/{workspace_id}/model` for current question, model/data versions and findings.
2. Submit to `POST /api/studies/{workspace_id}/actions`:
   - `set_question` is every study's first action, and for now its only point:
     `{"action":"set_question","question":{"text":"Does workload affect sleep?","outcome":"construct:sleep_quality","queries":{"lighter weeks":{"start":"2026-05-15","horizon":"4w","interventions":[{"target":"construct:workload","value":2}]}}}}`.
     Each query is a contrast of its interventions with the recorded course. It has one
     calendar day, `start`; the window is a `horizon` and interventions sit `after` an
     offset from the start (omitted means at the start), both in `s|m|h|d|w` durations,
     where `m` is minutes. Name constructs by the identities the model will define.
     Other actions are rejected until the question exists.
   - `edit_model`: `{"action":"edit_model","expected_revision":null,"model":{"edges":[...]}}`.
     Model structure, measurements, mechanisms, constants, and laws can be edited together.
     Define the question's constructs with their identities; question checks report
     "not evaluated" until the model does. Coefficients that enter only as a sum or only
     as a product in every use are rejected; merge them into one parameter. Give each
     parameter's prior law its `reasoning` and `sources`.
     Each observation law has a distribution tag and direct Expression fields, e.g.
     `{"distribution":"Delta","v":{"kind":"state","construct_id":"construct:workload"}}`.
     Bernoulli uses `BernoulliLogits` with `logits` or `BernoulliProbs` with `probs`.
     Edges carry drift mechanisms; construct dynamics may also carry potentials.
     Parameter transforms own their interval: `{"kind":"dt_effect_to_ct_rate","interval_days":7}`
     or an explicit `"model_clock"` duration; native-scale laws use `{"kind":"identity"}`.
     Valid incomplete models are saved with applicable specification findings.
   - `prepare_data`: supply `input={"source":{"files":["diary.csv"]},"definition":{...}}`
     with `default_window`, `variables`, and optional interpretation `context` in the definition.
     Each variable has a stable ID, dtype, summary, scoring rubric, extraction mode,
     source columns, window and codebook as appropriate. The action ingests and extracts
     in one call, retaining the semantic worker fan-out and deterministic scoring paths.
     Alternatively, `input={"revision":"<simulation commit OID>","replicate":0}`
     selects one recorded simulation draw. Its observations, schema and support layout
     are already defined, so extraction, re-encoding and filling are skipped.
     Files may declare optional source coverage `start` and `end` dates; only complete
     support windows inside that span are prepared. Computed variables may use Polars
     `fill_null` strategies or a numeric constant, with `fill_null_limit` for forward/backward.
     Ingestion and one-variable, one-window extraction requests reuse retained validated
     results by content across studies; reuse is reported in action messages. The committed
     panel and recipe remain the scientific record. Both sources run numerical data checks without loading a model.
     Latent paths and parameter truths stay in simulation sources.
     See the [prepare_data chart](../../../docs/assets/action-flows/prepare-data.svg) for branches
     and [time semantics](../../../docs/assumptions.md#time) for the panel origin.
   - `fit`: `{"action":"fit","model_revision":"<model tree OID>","panel_revision":"<panel tree OID>"}` conditions the selected
     model on observations. Returns joint uncertainty and fit diagnostics; predictive
     simulation is a separate request. Current fitting supports independent scalar laws.
   - `simulate`: `{"action":"simulate","model_revision":"<model tree OID>","start":"2026-05-15","horizon":"30d","interventions":[]}`
     generates a window from the model's current laws, in the same shape as a question query.
     The record's model day zero places the start: the fit's origin for fitted laws,
     otherwise the current panel's; without a panel the start is day zero. Interventions are
     optional: `{"target":"<construct ID>","after":"5d","value":1}` assigns a state that long
     after the start, then its natural dynamics resume. The framework derives the grid and always includes
     process and observation uncertainty. The saved report includes all state and indicator
     summaries, law provenance and fit reliability, with causal intervals only when certified.
     See [time semantics](../../../docs/assumptions.md#time) for initial laws and calendar binding.
     Compare the saved observations separately with `data_diff`; simulation does not
     accept comparison data or change its generation rules for predictive checks.
3. Dispatch returns HTTP 202 with only `{"attempt_id":"<UUID>"}` after durable acceptance.
   Poll `GET /api/studies/{workspace_id}/actions/{attempt_id}` until `kind` is `completed`.
   A `running` poll carries messages. A `completed` poll carries the correlated attempt,
   its commit ID and messages. Its outcome is `applied` with a result, `rejected` with a
   reason and detail, or `raised` with the execution error. Messages accumulate as
   `{timestamp, level, label}` with UTC timestamps, `debug|info|warn|error` levels,
   and stable `SCREAMING_SNAKE_CASE` labels. Warnings can accompany a saved result;
   failed actions leave the scientific branch unchanged. Do not redispatch while polling.
   While `prepare_data` runs, `GET /api/studies/{workspace_id}/events?attempt_id=...` pages
   its live step and extraction progress; pass the last `cursor` as `after`.
   `GET /api/studies/{workspace_id}/timeline` retains `applied`, `rejected`, or `raised`
   attempts and their messages. Numerical arrays have immutable store references.
   `GET /api/studies/{workspace_id}/model` includes separately sourced specification,
   identification, data-compatibility, fitting, and simulation findings.

The same requests are available as tools through `GET /api/tools/scientific` and
`POST /api/tools/scientific/{action}`; these tools return the receipt in `result`.
Use `poll_action` with `{attempt_id}` to read the same poll response in `result`.
HTTP and tool calls share execution contracts. V2 is a read-only inspector.

## Revisions and execution

Requests name stored input revisions. Fits check the selected model/data pair; model edits reject base revision conflicts. Read `/revisions` to select history. `GET /model-diff?before=...&after=...` compares model trees or Git checkpoints; `POST /data-diff` compares saved data selections in `left` and `right`. Model diffs create no attempt. Data diffs return an attempt_id to poll and save a read-only timeline leaf off the call-time branch head, without moving it; GET /data-diff/{commit_id} reads the saved comparison.
An edit does not require prior simulation or an authoring admission. Causal numerical
claims still require matching identification and production inference evidence.
Model edits automatically run affected checks, including one exact whole-model
predictive batch when a compatible panel is available. Data preparation runs only
data checks. Unchanged checks reuse
their recorded results. Scientific failures save as findings; missing prerequisites carry
not_evaluated reasons. Read predictive details and law provenance in the applied result.
Simulation reports retain their own generating model revision; a later edit makes that
report historical rather than evidence for the edited model.

Only the five scientific actions submit scientific work. Execution jobs and
LLM subroutines are private implementation details; callers do not select them.
Simulation always uses the same nonlinear generator. Paired intervention histories share
joint parameter/state draws and random streams. Causal effects on the question's
outcome are reported only when identification and committed production-fit evidence support
that interpretation; otherwise the report keeps its histories with an explicit reason.
The `analysis` context is read-only model introspection.

## Data in, results out

Upload files at `POST /api/upload` (`multipart/form-data` with `workspaceId` and
`file`) before `prepare_data` with its file preparation input. Read artifact payloads at
`GET /api/studies/{workspace_id}/artifacts/{artifact_id}`; binary files are served
from `.../files/{filename}`. Long jobs may outlive an HTTP client timeout; inspect
the timeline before submitting another request.

## Read-only deployments

`GET /api/actions-enabled` returns a boolean. Read-only deployments reject scientific action submissions with 403.
"""


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    from nof1_causal_lab.utils.config import configure_jax_persistent_cache

    configure_jax_persistent_cache()
    yield


app = FastAPI(
    lifespan=_lifespan,
    title="nof1-causal-lab study API",
    description=_API_DESCRIPTION,
    docs_url="/api/tools/docs",
)
app.state.study_clients = TemporalClientProvider()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(study_router)
app.include_router(capabilities_router)
app.include_router(workspaces_router)
app.include_router(uploads_router)


def _extract_observation_timestamps(observation_data: pl.DataFrame | None) -> list[datetime]:
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
    inference: StudyRevision
    observation_data: pl.DataFrame
    report: InferenceReport
    compiled: CompiledModel

    @cached_property
    def draws(self) -> JointPosteriorDraws:
        from nof1_causal_lab.models.ssm.inference.shared import model_draws
        from nof1_causal_lab.models.ssm.runtime import replay_input_events, replay_input_values

        events = replay_input_events(
            self.compiled,
            self.observation_data,
            time_origin=self.report.core.time_origin,
            start=self.model.time_points[0],
            end=self.model.time_points[-1],
            indicator_column="indicator",
        )
        from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure

        if isinstance(events, ObservationPreflightFailure):
            raise RuntimeError(f"Committed fit input events are inconsistent: {events.message}")
        return model_draws(
            self.compiled,
            input_values=replay_input_values(self.compiled, self.model.time_points, events),
        )

    @cached_property
    def dynamics(self) -> DynamicsDraws:
        return dynamics_from_samples(
            self.compiled, self.draws.parameters, n_draws=self.draws.describe().n_draws
        )


@lru_cache(maxsize=2)
def _load_simulation(
    _data_root: str, workspace_id: str, model_revision: GitOid
) -> _LoadedSimulation:
    """Reuse loaded fits; the storage root partitions local/test/remote workspaces.

    Only immutable inputs enter this cache. Current identification and freshness
    are checked separately on every request. Eviction simply reloads the fit.
    """
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import inference_record
    from nof1_causal_lab.study.store import ArtifactStore, read_model

    store = ArtifactStore(workspace_id)
    model_info = store.read_meta("model", model_revision)
    record = inference_record(StudyRepository(workspace_id).attempts(), model_revision)
    if record is None:
        raise HTTPException(404, "This model revision has no committed fit")
    panel_pin = model_info.derived_from["panel"]
    data_for_model = store.read_parquet_file("panel", panel_pin, parquet_filename("panel", "panel"))
    model = read_model(store, model_revision)

    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.study.store import read_current_state, read_question

    # A study's question is set once, so its outcome scoped every fit in it.
    question = read_question(store, read_current_state(workspace_id).current["question"].revision)
    compiled = compile_executable_model(StructuralSelection.for_question(model, question))

    from nof1_causal_lab.study.records import Applied

    assert record.record.attempt.action == "fit"
    assert isinstance(record.record.attempt.outcome, Applied)
    report = record.record.attempt.outcome.result.report
    projected = project_observation_data(
        data_for_model=data_for_model,
        model_spec=compiled,
        time_origin=report.core.time_origin,
    )
    from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure

    if isinstance(projected, ObservationPreflightFailure):
        raise RuntimeError(f"Committed fit observations are inconsistent: {projected.message}")
    _, observation_data = projected
    return _LoadedSimulation(model, record, observation_data, report, compiled)


def _build_analysis_context(workspace_id: str) -> AnalysisToolContext:
    """Reuse pinned numerical inputs while checking current input revisions."""
    from nof1_causal_lab.study.state import freshness_report
    from nof1_causal_lab.study.store import ArtifactStore, read_current_state

    state = read_current_state(workspace_id)
    from nof1_causal_lab.study.lineage import inference_is_current

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
        status.record.artifact_id
        for status in freshness_report(state)
        if status.kind == "present"
        and status.validity == "stale"
        and status.record.artifact_id in serving_chain
    ]
    if stale_artifacts:
        raise HTTPException(
            409,
            "Analysis requires a current, identified fit; refresh stale inputs: "
            + ", ".join(stale_artifacts),
        )
    store = ArtifactStore(workspace_id)
    loaded = _load_simulation(data_root(), workspace_id, model_info.revision)
    model = loaded.model
    identification_report_info = state.get("identification_report")
    if identification_report_info is None:
        raise HTTPException(404, f"No identification_report for workspace {workspace_id}")
    identification_report = IdentificationReport.model_validate(
        store.read_json_file(
            "identification_report",
            identification_report_info.revision,
            json_filename("identification_report", "identification_report"),
        )
    )

    if identification_report.outcome is None:
        raise HTTPException(422, "The model has no identified outcome")
    outcome_name = model.get_construct(identification_report.outcome).name
    treatment_names = [
        model.get_construct(cid).name for cid in identification_report.estimable_treatments
    ]

    predictive = state.checks.predictive if state.checks is not None else None
    checks = (
        predictive.evaluation.predictive_checks
        if predictive is not None and predictive.evaluation.kind == "evaluated"
        else None
    )

    return {
        "_workspace_id": workspace_id,
        "model": model,
        "identification_report": identification_report,
        "inference_report": loaded.report,
        "predictive_checks": checks,
        "_simulation": loaded,
        "_observation_timestamps": _extract_observation_timestamps(loaded.observation_data),
        "_outcome_name": outcome_name,
        "_identifiable_treatments": treatment_names,
    }


# ---------------------------------------------------------------------------
# Tool implementations — map (context_id, tool_name) -> execute(context, input)
# ---------------------------------------------------------------------------


def _build_model_info_payload(ctx: AnalysisToolContext, args: GetModelInfoInput) -> JsonObject:

    sections = args.sections or ["overview", "variables", "capabilities"]
    focused = set(args.names)
    model = ctx["model"]
    posterior = ctx["inference_report"]
    compiled = ctx["_simulation"].compiled
    retained_state_names = {state.name for state in compiled.states}
    constructs = [
        construct for construct in get_constructs(model) if construct.name in retained_state_names
    ]
    indicators = [
        (model.indicator_owner(observation.id), model.indicator(observation.id))
        for observation in compiled.observations
    ]

    if focused:
        constructs = [item for item in constructs if item.name in focused]
        indicators = [
            (construct, indicator)
            for construct, indicator in indicators
            if indicator.observation.name in focused or construct.name in focused
        ]

    payload: dict[str, JsonValue] = {}
    if "overview" in sections:
        payload["overview"] = {
            "outcome": ctx.get("_outcome_name"),
            "treatments": [*ctx["_identifiable_treatments"]],
            "n_latent": numeric.n_states(compiled),
            "n_manifest": len(numeric.observation_names(compiled)),
            "inference_method": posterior.core.inference_metadata.method,
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
                    "id": item.id,
                    "name": item.name,
                    "description": item.description,
                    "role": item.role,
                    "temporal_status": item.temporal_status,
                }
                for item in constructs
            ],
            "indicators": [
                {
                    "id": item.observation.id,
                    "name": item.observation.name,
                    "construct_id": construct.id,
                    "construct_name": construct.name,
                    "measurement_dtype": item.observation.measurement_dtype,
                    "support_kind": item.observation.support_kind.value,
                    "summary_operator": item.observation.summary_operator.value,
                    "observation_window": item.observation.observation_window.source
                    if item.observation.observation_window is not None
                    else None,
                }
                for construct, item in indicators
            ],
        }
    if "measurement" in sections:
        payload["measurement"] = {
            "model_clock": get_model_clock(model).source,
            "manifest_names": [*numeric.observation_names(compiled)],
        }
    if "identifiability" in sections:
        payload["identifiability"] = {
            "identifiable_treatments": [*ctx["_identifiable_treatments"]],
            "non_identifiable_treatments": {
                identity: finding.model_dump(mode="json")
                for identity, finding in ctx["identification_report"].non_identifiable.items()
            },
        }
    if "diagnostics" in sections:
        payload["diagnostics"] = {
            "ppc_warning_count": len(checks.per_variable_warnings)
            if (checks := ctx.get("predictive_checks")) is not None
            else 0,
        }
    if "capabilities" in sections:
        from nof1_causal_lab.actions.contracts import SimulateRequest

        payload["capabilities"] = {
            "simulate": {
                "intervention_targets": [*numeric.state_ids(compiled)],
                "request": SimulateRequest.model_json_schema(),
            },
        }
    return payload


def _execute_get_model_info(ctx: AnalysisToolContext, args: GetModelInfoInput) -> ToolResult:
    return {"result": _build_model_info_payload(ctx, args)}


# Registry: (context_id, tool_name) -> implementation function
_TOOL_IMPLS: dict[tuple[str, str], ToolImplementation] = {
    ("literature", "search_literature"): partial(
        _parse_tool_call,
        SearchLiteratureInput,
        execute_search_literature,
        lambda workspace_id: ToolContext(_workspace_id=workspace_id),
    ),
    ("analysis", "get_model_info"): partial(
        _parse_tool_call, GetModelInfoInput, _execute_get_model_info, _build_analysis_context
    ),
}


def _get_tool_contract(context_id: str, tool_name: str) -> ToolDefinition | None:
    contracts = CONTEXT_TOOLS.get(context_id) or []
    return next((contract for contract in contracts if contract.name == tool_name), None)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


class ToolCallRequest(BaseModel):
    branch: str = "main"
    expected_head: GitOid | None = None
    workspace_id: str
    input: JsonObject


@app.get("/api/tools/{context_id}")
def get_tool_schemas(context_id: str) -> list[ToolSchema]:
    """List a context's validation/query tools — the same tools the in-service LLM loops use.

    Each entry is `{name, description, parameters, result}` where `parameters`
    and `result` are JSON Schemas. Fetch this first to learn a tool's argument
    shape, then call `POST /api/tools/{context_id}/{tool_name}`. Examples:
    analysis `simulate` / `get_model_info`, literature `search_literature`.
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
    context_id: str, tool_name: str, request: ToolCallRequest, http_request: Request
) -> ToolResult:
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
        if tool_name == "poll_action":
            from nof1_causal_lab.actions.results import PollActionRequest
            from nof1_causal_lab.study_api import read_action_poll

            try:
                query = PollActionRequest.model_validate(request.input)
            except ValidationError as exc:
                raise HTTPException(422, detail=exc.errors(include_context=False)) from exc
            result = await read_action_poll(
                request.workspace_id, query.attempt_id, study_clients(http_request)
            )
            return {"result": result.model_dump(mode="json")}

        from nof1_causal_lab.actions.contracts import ScientificActionRequest
        from nof1_causal_lab.study_api import execute_scientific_action

        try:
            action = TypeAdapter(ScientificActionRequest).validate_python(
                contract.input_schema.model_validate(request.input)
            )
        except ValidationError as exc:
            raise HTTPException(422, detail=exc.errors(include_context=False)) from exc
        outcome = await execute_scientific_action(
            request.workspace_id,
            action,
            study_clients(http_request),
            branch=request.branch,
            expected_head=request.expected_head,
        )
        return {"result": outcome.model_dump(mode="json")}

    impl = _TOOL_IMPLS.get((context_id, tool_name))
    if impl is None:
        raise HTTPException(
            404, f"No implementation for tool {tool_name!r} in context {context_id!r}"
        )

    pending_payload = impl(request.workspace_id, request.input)
    payload: ToolResult = (
        await pending_payload if inspect.isawaitable(pending_payload) else pending_payload
    )

    if contract.output_schema is None:
        return payload

    try:
        payload["result"] = (
            TypeAdapter(contract.output_schema)
            .validate_python(payload.get("result"))
            .model_dump(mode="json")
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
