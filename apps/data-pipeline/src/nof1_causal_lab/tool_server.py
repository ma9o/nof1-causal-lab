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
from math import ceil
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import ModelRevision
from nof1_causal_lab.artifacts.scenarios import ScenarioRequest
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
from nof1_causal_lab.models.ssm.counterfactual import (
    ClampSpec,
    summarize_draws,
    vmap_simulate_clamps_from_state,
)
from nof1_causal_lab.models.ssm.dynamics import (
    Intervention,
    SimulationConfig,
    VectorField,
    compute_steady_state,
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
    from nof1_causal_lab.artifacts.scenarios import ScenarioQueryInput, ScenarioStartInput
    from nof1_causal_lab.flows.contracts_base import ToolDefinition
    from nof1_causal_lab.machine.store import TransitionRecord
    from nof1_causal_lab.models.ssm.dynamics.posterior import PosteriorDynamicsSamples
    from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
    from nof1_causal_lab.models.ssm.runtime import PreparedModelRuntime

_API_DESCRIPTION = """\
The episode machine is the single interface to an N-of-1 causal analysis. An
external agent drives it entirely over this HTTP API — the same surface the web
viewer uses. There is no SDK and no MCP server: `curl` is the interface.

## Orientation

Call `GET /api/machine` once. It returns the static artifact graph — every
transition with what it consumes, produces, and optionally co-produces — plus
each transition's creation class and the derivation graph:

- `deterministic` — pure compute, no credentials (e.g. identification).
- `batch_llm` — bulk LLM compute on the service's ambient key. You trigger it
  with a `run` move; you never supply a key.
- `judgment` — proposal work you can do yourself by writing the produced
  artifact directly. These transitions are flagged `writable`.

## The loop

1. `GET /api/machine` once, then `GET /api/episodes/{workspace_id}` for the live
   state: per-artifact freshness, the legal moves, and whether an auto-run is
   active.
2. Propose a move at `POST /api/episodes/{workspace_id}/moves` — either
   `{"move": {"kind": "run", "operation_id": "latent_structure"}}` to run a transition, or
   `{"move": {"kind": "write", "artifact_id": "model", "expected_model_version": 0, "provenance": "llm"}, "payload": {...}}`
   to create the scientific model directly (use its current version for later writes).
3. Long transitions (`statistical_model_spec`, `posterior` — minutes to hours) can outlive a client
   timeout. Prefer `POST /api/episodes/{workspace_id}/auto` (a background driver
   that runs enabled transitions in dependency order) and poll the state.
4. Read what happened at `GET /api/episodes/{workspace_id}/timeline`: `applied`,
   `rejected` (illegal, state unchanged), or `raised` (typed transition error).

## Staleness

A `write` becomes a new provenance root and marks everything downstream stale
until re-run. Numeric tools (`simulate`, `get_model_info`) hard-flag
stale provenance chains in their warnings — never report numbers past those
flags.

## Data in, results out

Upload raw data at `POST /api/upload` (`multipart/form-data` with `workspaceId`
and `file`) before running the `raw_data` transition. Read artifact payloads at
`GET /api/episodes/{workspace_id}/artifacts/{artifact_id}`; binary files
(parquet, pickle) are served individually from `.../files/{filename}`.

## Read-only deployments

The hosted viewer's backend serves these same read endpoints against a published
store with no move plane. `GET /api/capabilities` reports `moves_enabled`; every
move returns 403 when it is `false`.
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


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


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


def _manifest_effects(
    samples: UncheckedJsonObject,
    outcome_idx: int,
    effect_mean: float,
    manifest_names: list[str],
) -> dict[str, float] | None:
    lambda_draws = samples.get("lambda")
    if lambda_draws is None:
        return None
    lambda_mean = (
        jnp.mean(lambda_draws, axis=0) if getattr(lambda_draws, "ndim", 0) == 3 else lambda_draws
    )
    if getattr(lambda_mean, "ndim", 0) != 2:
        return None

    effects: dict[str, float] = {}
    for idx, loading in enumerate(lambda_mean[:, outcome_idx]):
        loading_value = float(loading)
        if abs(loading_value) <= 1e-9:
            continue
        name = manifest_names[idx] if idx < len(manifest_names) else f"manifest_{idx}"
        effects[name] = loading_value * effect_mean
    return effects or None


def _serialize_effect_trajectory(
    trajectory: jnp.ndarray, time_grid_days: jnp.ndarray
) -> list[dict[str, float]]:
    days = time_grid_days.tolist()
    values = trajectory.tolist()
    return [
        {"day": round(float(day), 3), "effect": float(value)}
        for day, value in zip(days, values, strict=False)
    ]


def _serialize_node_trajectories(
    state_paths: jnp.ndarray,
    latent_names: list[str],
) -> dict[str, list[float]]:
    mean_paths = jnp.mean(state_paths, axis=0)
    return {
        name: [float(value) for value in mean_paths[:, idx].tolist()]
        for idx, name in enumerate(latent_names)
    }


def _serialize_latent_state(state: jnp.ndarray, latent_names: list[str]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(latent_names, state.tolist(), strict=False)}


def _resolve_counterfactual_start(
    ctx: UncheckedJsonObject,
    start: ScenarioStartInput,
    *,
    n_timepoints: int,
) -> tuple[int, str | None]:
    if n_timepoints <= 0:
        raise HTTPException(400, "Persisted fitted latent paths contain no timepoints.")

    raw_time_index = start.time_index
    raw_time = start.time
    timestamps = list(ctx.get("_observation_timestamps") or [])

    if raw_time_index is not None:
        time_index = int(raw_time_index)
    elif raw_time:
        if not timestamps:
            raise HTTPException(
                400, "start.time requires observed timestamps in the fitted workspace."
            )
        requested = _parse_iso_datetime(str(raw_time))
        matches = [
            idx for idx, timestamp in enumerate(timestamps[:n_timepoints]) if timestamp == requested
        ]
        if not matches:
            raise HTTPException(
                400, "start.time must exactly match a retained fitted-state timestamp."
            )
        time_index = matches[0]
    else:
        time_index = n_timepoints - 1

    if time_index < 0 or time_index >= n_timepoints:
        raise HTTPException(
            400,
            f"start.time_index must be between 0 and {n_timepoints - 1}; got {time_index}.",
        )

    time = timestamps[time_index].isoformat() if time_index < len(timestamps) else None
    return time_index, time


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

    @cached_property
    def baseline_states(self):
        draws = self.dynamics.param_samples
        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *draws)
        return jax.vmap(
            lambda params: compute_steady_state(
                self.dynamics.vector_field, params, Intervention.none()
            )
        )(stacked)


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


def _build_ranking_context(workspace_id: str) -> UncheckedJsonObject:
    """Reuse pinned numerical inputs while checking current serving provenance."""
    from nof1_causal_lab.machine.moves import freshness_report
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
    ranking_info = state.get("baseline_report")
    baseline_report = (
        store.read_json_file(
            "baseline_report",
            ranking_info.version,
            json_filename("baseline_report", "baseline_report"),
        )
        if ranking_info is not None
        else {}
    )
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

    return {
        "_workspace_id": workspace_id,
        "model": model.model_dump(mode="json"),
        "identification_report": identification_report.model_dump(mode="json"),
        "inference_report": loaded.inference.diagnostics["report"],
        "baseline_report": baseline_report,
        "_causal_analysis": causal_analysis,
        "_prepared_runtime": loaded.runtime,
        "_simulation": loaded,
        "_observation_timestamps": _extract_observation_timestamps(loaded.runtime.observation_data),
        "_outcome_name": outcome_name,
        "_identifiable_treatments": treatment_names,
    }


@dataclass(frozen=True)
class AnalysisSimulationSetup:
    causal_analysis: CertifiedCausalAnalysis
    samples: UncheckedJsonObject
    readout: ScenarioQueryInput
    clamps: list[ClampSpec]
    outcome: str
    latent_names: list[str]
    manifest_names: list[str]
    outcome_idx: int
    time_grid: jnp.ndarray
    vector_field: VectorField
    # ``param_samples`` is the canonical per-draw component-shape
    # parameter list rebuilt from ``ModelSpec`` and posterior sample sites.
    param_samples: list[tuple[UncheckedJsonObject, ...]] | None = None


@dataclass(frozen=True)
class AnalysisEffectOutputs:
    summary: dict[str, float]
    effect_trajectory: list[dict[str, float]] | None
    visualization: UncheckedJsonObject | None
    manifest_effects: dict[str, float] | None


def _tool_error_result(
    message: str,
    *,
    identifiable_treatments: list[str] | None = None,
) -> UncheckedJsonObject:
    result: UncheckedJsonObject = {"error": message}
    if identifiable_treatments is not None:
        result["identifiable_treatments"] = identifiable_treatments
    return {"result": result}


def _prepare_analysis_simulation(
    ctx: UncheckedJsonObject,
    request: ScenarioRequest,
) -> tuple[AnalysisSimulationSetup | None, UncheckedJsonObject | None]:
    causal_analysis: CertifiedCausalAnalysis = ctx["_causal_analysis"]
    samples = ctx["_simulation"].draws.parameters

    model = causal_analysis.model
    outcome_id = request.outcome.id
    constructs = {item.id: item for item in model.constructs}
    if outcome_id not in constructs:
        return None, _tool_error_result("Outcome is absent from the fitted model.")
    outcome = constructs[outcome_id].name
    if outcome != causal_analysis.outcome:
        return None, _tool_error_result("Outcome does not match the identified estimand.")
    spec = causal_analysis.model
    latent_names = list(numeric.state_names(spec) or [])
    manifest_names = list(numeric.observation_names(spec) or [])
    name_to_idx = {name: idx for idx, name in enumerate(latent_names)}
    outcome_idx = name_to_idx.get(outcome)
    if outcome_idx is None:
        return None, _tool_error_result("Outcome not present in fitted latent structure.")

    identifiable = causal_analysis.treatments
    clamps: list[ClampSpec] = []
    for clamp in request.clamps:
        target_id = clamp.target.id
        if target_id not in constructs:
            return None, _tool_error_result("Clamp target is absent from the fitted model.")
        variable = constructs[target_id].name
        if variable not in identifiable:
            return None, _tool_error_result(
                f"Clamp target '{variable}' is not an identifiable ranking target.",
                identifiable_treatments=identifiable,
            )
        index = name_to_idx.get(variable)
        if index is None:
            return None, _tool_error_result(
                f"Clamp target '{variable}' is not present in the fitted latent structure."
            )
        clamps.append(
            ClampSpec(
                index=index,
                mode=clamp.mode,
                from_day=clamp.from_day,
                to_day=clamp.to_day,
                value=clamp.value,
                amount=clamp.amount,
                value_start=clamp.value_start,
                value_end=clamp.value_end,
                values=tuple(clamp.values) if clamp.values is not None else None,
            )
        )

    horizon = request.readout.horizon_days
    steps = max(1, ceil(horizon / model.model_clock_days))
    time_grid = jnp.linspace(0.0, float(horizon), steps + 1)

    posterior_dynamics = ctx["_simulation"].dynamics
    vector_field = posterior_dynamics.vector_field
    param_samples = posterior_dynamics.param_samples
    if not param_samples:
        return None, _tool_error_result("Posterior dynamics samples are unavailable.")

    return (
        AnalysisSimulationSetup(
            causal_analysis=causal_analysis,
            samples=samples,
            readout=request.readout,
            clamps=clamps,
            outcome=outcome,
            latent_names=latent_names,
            manifest_names=manifest_names,
            outcome_idx=outcome_idx,
            time_grid=time_grid,
            vector_field=vector_field,
            param_samples=param_samples,
        ),
        None,
    )


def _build_visualization_payload(
    latent_names: list[str],
    *,
    reference_node_paths: jnp.ndarray | None = None,
    action_node_paths: jnp.ndarray | None = None,
    node_effect_paths: jnp.ndarray | None = None,
    start_state: dict[str, float] | None = None,
) -> UncheckedJsonObject | None:
    reference_node_trajectories = (
        _serialize_node_trajectories(reference_node_paths[:, 1:], latent_names)
        if reference_node_paths is not None
        else None
    )
    action_node_trajectories = (
        _serialize_node_trajectories(action_node_paths[:, 1:], latent_names)
        if action_node_paths is not None
        else None
    )
    node_effect_trajectories = (
        _serialize_node_trajectories(node_effect_paths[:, 1:], latent_names)
        if node_effect_paths is not None
        else None
    )
    if (
        reference_node_trajectories is None
        and action_node_trajectories is None
        and node_effect_trajectories is None
        and start_state is None
    ):
        return None
    return {
        "reference_node_trajectories": reference_node_trajectories,
        "action_node_trajectories": action_node_trajectories,
        "node_effect_trajectories": node_effect_trajectories,
        "start_state": start_state,
    }


def _build_effect_outputs(
    setup: AnalysisSimulationSetup,
    *,
    effect_draws: jnp.ndarray | None = None,
    effect_paths: jnp.ndarray | None = None,
    reference_node_paths: jnp.ndarray | None = None,
    action_node_paths: jnp.ndarray | None = None,
    node_effect_paths: jnp.ndarray | None = None,
    start_state: dict[str, float] | None = None,
) -> AnalysisEffectOutputs:
    if effect_paths is not None:
        effect_draws = effect_paths[:, -1]
        mean_effect_trajectory = jnp.mean(effect_paths, axis=0)
        effect_trajectory = _serialize_effect_trajectory(
            mean_effect_trajectory[1:], setup.time_grid[1:]
        )
    else:
        effect_trajectory = None

    if effect_draws is None:
        raise ValueError("Either effect_draws or effect_paths must be provided.")

    summary = summarize_draws(effect_draws).model_dump(mode="json")
    manifest_effects = None
    if setup.readout.projection in {"manifest", "both"}:
        manifest_effects = _manifest_effects(
            setup.samples,
            setup.outcome_idx,
            summary["mean"],
            setup.manifest_names,
        )

    construct_ids = {item.name: item.id for item in setup.causal_analysis.model.constructs}
    return AnalysisEffectOutputs(
        summary=summary,
        effect_trajectory=effect_trajectory,
        visualization=_build_visualization_payload(
            [construct_ids[name] for name in setup.latent_names],
            reference_node_paths=reference_node_paths,
            action_node_paths=action_node_paths,
            node_effect_paths=node_effect_paths,
            start_state={construct_ids[name]: value for name, value in start_state.items()}
            if start_state is not None
            else None,
        ),
        manifest_effects=manifest_effects,
    )


def _collect_analysis_warnings(
    ctx: UncheckedJsonObject,
    *,
    treatments: list[str] | None = None,
    include_diagnostic_warnings: bool = False,
    extra_warnings: list[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    if include_diagnostic_warnings and treatments:
        posterior = ctx.get("posterior", {})
        for item in posterior["assessment"].get("ppc", {}).get("per_variable_warnings", []) or []:
            message = item.get("message")
            if message:
                warnings.append(str(message))

    for warning in extra_warnings or []:
        if warning:
            warnings.append(str(warning))
    return warnings


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
    posterior = ctx.get("posterior", {})
    baseline_report = ctx.get("baseline_report", {})
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
            "non_identifiable_treatments": (
                ctx["identification_report"]["status"]["non_identifiable_treatments"] or {}
            ),
        }
    if "diagnostics" in sections:
        payload["diagnostics"] = {
            "ppc_warning_count": len(
                (posterior["assessment"].get("ppc") or {}).get("per_variable_warnings", []) or []
            ),
        }
    if "baseline_effects" in sections:
        baseline = list(baseline_report.get("intervention_results", []) or [])
        if focused:
            baseline = [entry for entry in baseline if entry.get("treatment") in focused]

        def _draws_summary(draws):
            if not draws:
                return None, None
            return sum(draws) / len(draws), sum(1 for d in draws if d > 0) / len(draws)

        payload["baseline_effects"] = [
            {
                "treatment": entry.get("treatment"),
                "effect_size": _draws_summary(entry.get("posterior_draws"))[0],
                "prob_positive": _draws_summary(entry.get("posterior_draws"))[1],
            }
            for entry in baseline[:10]
        ]
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


def _execute_simulate(ctx: UncheckedJsonObject, args: UncheckedJsonObject) -> UncheckedJsonObject:
    """Run a composable scenario: a start state + a list of timed latent clamps.

    The start is the population baseline steady state (interventional) or an abducted
    fitted latent state (counterfactual); the clamps are do-operators over time windows.
    The Pearl rung is emergent from the start rather than a separate query type.
    """
    request = ScenarioRequest.model_validate(args)
    setup, error = _prepare_analysis_simulation(ctx, request)
    if error is not None:
        return error
    assert setup is not None
    assert setup.param_samples is not None

    estimand = setup.readout.estimand

    if request.start.kind == "abducted":
        latent_paths = ctx["_simulation"].draws.latent_paths
        if latent_paths is None:
            return _tool_error_result(
                "Posterior fitted artifact is missing persisted latent state paths required "
                "for an abducted start."
            )
        start_index, start_time = _resolve_counterfactual_start(
            ctx, request.start, n_timepoints=int(latent_paths.shape[1])
        )
        initial_states = latent_paths[:, start_index, :]
        n_draws = len(setup.param_samples)
        if int(initial_states.shape[0]) != n_draws:
            return _tool_error_result(
                "Persisted fitted latent path draw count does not match posterior dynamics "
                f"draw count ({int(initial_states.shape[0])} != {n_draws})."
            )
    else:
        initial_states = ctx["_simulation"].baseline_states
        start_index, start_time = None, None

    config = SimulationConfig()

    start_state = _serialize_latent_state(jnp.mean(initial_states, axis=0), setup.latent_names)

    baseline_state_paths, action_state_paths, effect_state_paths = vmap_simulate_clamps_from_state(
        setup.vector_field,
        setup.param_samples,
        initial_states=initial_states,
        clamps=setup.clamps,
        time_grid=setup.time_grid,
        config=config,
    )
    outcome_effect = effect_state_paths[:, :, setup.outcome_idx]
    reference_mean = float(jnp.mean(baseline_state_paths[:, -1, setup.outcome_idx]))

    common_viz: UncheckedJsonObject = {
        "reference_node_paths": baseline_state_paths,
        "action_node_paths": action_state_paths,
        "node_effect_paths": effect_state_paths,
        "start_state": start_state,
    }
    if estimand == "trajectory":
        outputs = _build_effect_outputs(setup, effect_paths=outcome_effect, **common_viz)
    else:
        outputs = _build_effect_outputs(setup, effect_draws=outcome_effect[:, -1], **common_viz)

    clamp_variables = [setup.latent_names[clamp.index] for clamp in setup.clamps]

    return {
        "result": {
            "request": request.model_dump(mode="json"),
            "provenance": {
                "model": setup.causal_analysis.model_revision.model_dump(mode="json"),
                "engine": "nonlinear_drift_v1",
                "solver": "Tsit5",
                "rtol": config.rtol,
                "atol": config.atol,
                "max_steps": config.max_steps,
                "draw_count": len(setup.param_samples),
                "time_grid_days": setup.time_grid.tolist(),
                "start_time_index": start_index,
                "start_time": start_time,
            },
            "labels": {item.id: item.name for item in setup.causal_analysis.model.constructs},
            "summary": outputs.summary,
            "effect_trajectory": outputs.effect_trajectory,
            "trajectory_peak": max(
                outputs.effect_trajectory, key=lambda point: abs(point["effect"])
            )
            if outputs.effect_trajectory
            else None,
            "visualization": outputs.visualization,
            "manifest_effects": outputs.manifest_effects,
            "reference_mean": reference_mean,
            "warnings": _collect_analysis_warnings(
                ctx,
                treatments=clamp_variables,
                include_diagnostic_warnings=True,
            ),
        }
    }


# Registry: (context_id, tool_name) -> implementation function
_TOOL_IMPLS: dict[tuple[str, str], ToolImplementation] = {
    ("latent-structure", "validate_latent_structure"): _execute_validate_latent_structure,
    (
        "measurement-structure",
        "validate_measurement_structure",
    ): _execute_validate_measurement_structure,
    ("measurement", "validate_extractions"): _execute_validate_extractions,
    ("statistical-model-spec", "search_literature"): _execute_search_literature,
    ("ranking", "get_model_info"): _execute_get_model_info,
    ("ranking", "simulate"): _execute_simulate,
}

# Upstream dependencies: which context results need to be loaded for execution.
_CONTEXT_DEPS: dict[str, list[str]] = {
    "latent-structure": [],
    "measurement-structure": ["model"],
    "measurement": [],
    "statistical-model-spec": ["model"],
    "ranking": [],
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
    if context_id == "ranking":
        return _build_ranking_context(workspace_id)
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
    ranking `simulate` / `get_model_info`, statistical-model-spec `search_literature`.
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
        raise HTTPException(422, detail=exc.errors()) from exc

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
