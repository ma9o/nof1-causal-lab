"""Run fits, simulations and simulated-data preparation against exact input revisions.

Each runner reads exactly its pinned input versions, writes new artifact versions with
the same pins in ``derived_from``, and returns effects for the workflow to publish.
Fits can be routed to Modal in production; routing is infra-only and cannot change
pinned versions. The enclosing action evaluates its checks before publication.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_lab.actions.contracts import FitRequest, PrepareDataRequest, SimulateRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.study.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.study.store import ArtifactStore

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
    from nof1_causal_lab.artifacts.simulation import FitReliability
    from nof1_causal_lab.study.state import StudyState


def _panel_df(store: ArtifactStore, pins: dict[ArtifactId, GitOid]) -> pl.DataFrame:
    return store.read_parquet_file("panel", pins["panel"], parquet_filename("panel", "panel"))


async def _run_fit(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    settings: FitSettingsSpec,
) -> ActionEffects:
    from nof1_causal_lab.actions.fit import (
        build_sampler_config,
        fit,
    )
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.utils.config import get_config

    panel = _panel_df(store, pins)
    from functools import cache

    from nof1_causal_lab.study.store import read_model

    model_spec = read_model(store, pins["model"])
    model_spec.check_execution()
    from nof1_causal_lab.actions.data_checks import require_data_binding
    from nof1_causal_lab.study.lineage import read_data_metadata

    require_data_binding(store, model_spec, pins["panel"])
    sampler_config = build_sampler_config()
    sampler_config.update(settings.model_dump(exclude_none=True))

    config = get_config().inference
    if config.compute_backend == "modal" and os.environ.get("DEPLOYMENT_ENV") != "production":
        from nof1_causal_lab.actions.modal_fit import fit_on_modal

        compute_fit = fit_on_modal
    else:
        compute_fit = fit
    result = await asyncio.to_thread(
        compute_fit,
        model_spec=model_spec,
        data_for_model=panel,
        time_origin=read_data_metadata(store, pins["panel"]).time_origin,
        sampler_config=sampler_config,
        array_writer=store.write_array,
        array_loader=cache(store.read_array),
        compute_loo_diagnostics=config.compute_loo_diagnostics,
    )

    conditioned = result["_model"]
    evidence = result["engine_evidence"]
    report = InferenceReport.model_validate(
        {key: value for key, value in result.items() if key not in {"_model", "engine_evidence"}}
    )
    info = store.write_artifact(
        "model",
        derived_from=pins,
        produced_by="fit",
        json_files={json_filename("model", "model"): conditioned.model_dump(mode="json")},
    )
    return ActionEffects(
        produced=[info],
        diagnostics={
            "input_pins": pins,
            "engine_evidence": evidence,
            "report": report.model_dump(mode="json"),
        },
    )


async def _run_simulate(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    design: SimulationSpec,
) -> ActionEffects:
    from nof1_causal_lab.actions.scenarios import summarize_causal_simulation
    from nof1_causal_lab.actions.simulate import simulate
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.lineage import inference_record
    from nof1_causal_lab.study.store import read_model

    model = read_model(store, pins["model"])
    from nof1_causal_lab.models.ssm.inference.convergence import convergence_failures
    from nof1_causal_lab.study.lineage import fitted_law_report, law_provenance, read_data_metadata

    records = StudyRepository(workspace_id).attempts()
    law = law_provenance(store, store.read_meta("model", pins["model"]), model, None)
    time_origin = None
    origin_panel_revision = pins.get("panel")
    if origin_panel_revision is not None:
        time_origin = read_data_metadata(store, origin_panel_revision).time_origin

    reliability: FitReliability = "unknown" if law.kind == "unknown" else "not_fitted"
    if law.fitted_model_revision is not None:
        fit_report = fitted_law_report(records, law.fitted_model_revision)
        time_origin = fit_report.time_origin
        origin_panel_revision = law.fitted_panel_revision
        reliability = (
            "unconverged" if convergence_failures(fit_report.inference_diagnostics) else "converged"
        )

    report = await asyncio.to_thread(
        simulate,
        model,
        design,
        revision=GitRef(workspace_id=workspace_id, revision=pins["model"], path="model.json"),
        write_array=store.write_array,
        time_origin=time_origin,
        fit_reliability=reliability,
    )
    report = type(report).model_validate(
        {**report.model_dump(), "law": law, "origin_panel_revision": origin_panel_revision}
    )
    report = summarize_causal_simulation(
        model,
        report,
        store=store,
        inference=inference_record(records, pins["model"]),
    )
    return ActionEffects(
        diagnostics={
            "input_pins": pins,
            "report": report.model_dump(mode="json"),
        }
    )


async def _run_simulated_data(
    workspace_id: str,
    store: ArtifactStore,
    source: SimulationReplicateRef,
) -> ActionEffects:
    """Materialize one recorded simulation replicate; its source owns every input selection."""
    from datetime import timedelta

    from nof1_causal_lab.actions.prepare_data import prepare_simulation_panel
    from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.study.history import StudyRepository

    record = StudyRepository(workspace_id).record(source.revision)
    if record.status != "applied" or record.action != "simulate":
        raise ValueError("The source revision must be an applied simulation commit")
    report = TypeAdapter(SimulationReport).validate_python(record.diagnostics["report"])
    if report.model.workspace_id != workspace_id:
        raise ValueError("The simulation must belong to the current study")
    panel = await asyncio.to_thread(
        prepare_simulation_panel,
        report,
        source.replicate,
        read_array=store.read_array,
    )
    used_pins: dict[ArtifactId, GitOid] = {}
    info = store.write_artifact(
        "panel",
        derived_from=used_pins,
        produced_by="prepare_data",
        json_files={
            json_filename("panel", "metadata"): PreparedDataMetadata(
                source=source,
                variables=report.observation_layout.variables,
                time_origin=report.time_origin + timedelta(days=report.times[0])
                if report.time_origin is not None
                else None,
            ).model_dump(mode="json")
        },
        parquet_files={parquet_filename("panel", "panel"): panel},
    )
    return ActionEffects(
        produced=[info],
        diagnostics={
            "input_pins": used_pins,
            "simulation_source": source.model_dump(mode="json"),
            "n_observations": panel["value"].count(),
        },
    )


async def run_action_locally(
    workspace_id: str,
    request: FitRequest | SimulateRequest | PrepareDataRequest,
    pins: dict[ArtifactId, GitOid],
) -> ActionEffects:
    """Run a fit, simulation or simulated-data preparation on this process."""
    store = ArtifactStore(workspace_id)
    if isinstance(request, FitRequest):
        return await _run_fit(store, pins, request.settings)
    if isinstance(request, SimulateRequest):
        design = SimulationSpec(
            start=request.start, end=request.end, interventions=request.interventions
        )
        return await _run_simulate(workspace_id, store, pins, design)
    if not isinstance(request.input, SimulationReplicateRef):
        raise TypeError("Uploaded files are prepared by the ingestion and extraction workflows")
    return await _run_simulated_data(workspace_id, store, request.input)


def _pinned(store: ArtifactStore, selected: dict[ArtifactId, GitOid]) -> dict[ArtifactId, GitOid]:
    """Exact selected input revisions; every one must exist in the store."""
    for artifact_id, revision in selected.items():
        store.read_meta(artifact_id, revision)
    return dict(selected)


async def run_action(
    workspace_id: str,
    request: FitRequest | SimulateRequest | PrepareDataRequest,
    state: StudyState,
) -> ActionEffects:
    """Pin a request's inputs and run it, routing fits to Modal in production."""
    store = ArtifactStore(workspace_id)
    if isinstance(request, FitRequest):
        pins = _pinned(store, {"model": request.model_revision, "panel": request.panel_revision})
        if os.environ.get("DEPLOYMENT_ENV") == "production":
            from nof1_causal_lab.actions.modal_runners import run_fit_on_modal
            from nof1_causal_lab.study.store import read_model

            read_model(store, pins["model"]).check_execution()
            return await run_fit_on_modal(workspace_id, request, pins)
    elif isinstance(request, SimulateRequest):
        pins = _pinned(store, {"model": request.model_revision})
        # A simulation compares against the current observations when the study has them.
        if (panel := state.get("panel")) is not None:
            pins["panel"] = panel.revision
    else:
        pins = {}
    return await run_action_locally(workspace_id, request, pins)
