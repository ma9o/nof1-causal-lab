"""Run fits, simulations and simulated-data preparation against exact input revisions.

Each runner reads exactly its pinned input versions, writes new artifact versions with
the same pins in ``derived_from``, and returns effects for the workflow to publish.
Local studies can place numerical fitting on Modal without moving their history
or changing pinned versions. The enclosing action evaluates its checks before publication.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from nof1_causal_lab.actions.contracts import FitRequest, PrepareDataRequest, SimulateRequest
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.artifacts.predictive_provenance import FittedLawProvenance, MixedLawProvenance
from nof1_causal_lab.artifacts.simulation import SimulationSpec
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.study.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.records import (
    Applied,
    DataPreparationResult,
    ModelFitResult,
    ModelSimulationResult,
)
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.study.view_models import DataDiffRequest

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
    from nof1_causal_lab.study.state import StudyState


def _panel_df(store: ArtifactStore, pins: dict[ArtifactId, GitOid]) -> pl.DataFrame:
    return store.read_parquet_file("panel", pins["panel"], parquet_filename("panel", "panel"))


async def _run_fit(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    settings: FitSettingsSpec,
) -> Applied[ModelFitResult]:
    from nof1_causal_lab.actions.fit import fit, read_inference_report
    from nof1_causal_lab.actions.inference.fit import resolve_sampler_spec
    from nof1_causal_lab.utils.config import get_config

    panel = _panel_df(store, pins)
    from functools import cache

    from nof1_causal_lab.study.store import read_model, read_question

    model_spec = read_model(store, pins["model"])
    from nof1_causal_lab.actions.data_checks import require_data_binding
    from nof1_causal_lab.study.lineage import read_data_metadata

    require_data_binding(store, model_spec, pins["panel"])
    selection = StructuralSelection.for_question(model_spec, read_question(store, pins["question"]))
    sampler = resolve_sampler_spec(settings)

    config = get_config().inference
    if config.compute_backend == "modal" and os.environ.get("DEPLOYMENT_ENV") != "production":
        from nof1_causal_lab.actions.modal_fit import fit_on_modal

        compute_fit = fit_on_modal
    else:
        compute_fit = fit
    result = await asyncio.to_thread(
        compute_fit,
        selection=selection,
        data_for_model=panel,
        time_origin=read_data_metadata(store, pins["panel"]).time_origin,
        sampler=sampler,
        array_writer=store.write_array,
        array_loader=cache(store.read_array),
        compute_loo_diagnostics=config.compute_loo_diagnostics,
    )

    conditioned = result["_model"]
    evidence = result["evidence"]
    info = store.write_artifact(
        "model",
        derived_from=pins,
        produced_by="fit",
        json_files={
            json_filename("model", "model"): conditioned.model_dump(mode="json", round_trip=True)
        },
    )
    read_inference_report(store, info.revision, evidence)
    return Applied(
        result=ModelFitResult(
            model=GitRef(
                workspace_id=store.workspace_id, revision=pins["model"], path="model.json"
            ),
            panel=GitRef(
                workspace_id=store.workspace_id, revision=pins["panel"], path="panel.parquet"
            ),
            evidence=evidence,
        ),
        effects=ActionEffects(produced=(info,)),
    )


async def _run_simulate(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    design: SimulationSpec,
) -> Applied[ModelSimulationResult]:
    from nof1_causal_lab.actions.simulate import simulate, read_simulation_report
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.store import read_model

    model = read_model(store, pins["model"])
    from nof1_causal_lab.study.lineage import fitted_law_report, law_provenance, read_data_metadata
    from nof1_causal_lab.study.store import read_question

    selection = StructuralSelection.for_question(model, read_question(store, pins["question"]))

    records = StudyRepository(workspace_id).attempts()
    law = law_provenance(store, store.read_meta("model", pins["model"]), model, None)
    # Without a record, the design's start is model day zero for the initial-state law.
    origin_panel_revision = pins.get("panel")
    time_origin = (
        read_data_metadata(store, origin_panel_revision).time_origin
        if origin_panel_revision is not None
        else design.start_instant
    )

    if isinstance(law, (FittedLawProvenance, MixedLawProvenance)):
        fit_report = fitted_law_report(store, records, law.fitted_model_revision)
        time_origin = fit_report.time_origin
        origin_panel_revision = law.fitted_panel_revision
    if time_origin is None:
        raise AggregatedCompileError(["A calendar-free record cannot place a dated simulation."])

    report = await asyncio.to_thread(
        simulate,
        selection,
        design,
        revision=GitRef(workspace_id=workspace_id, revision=pins["model"], path="model.json"),
        write_array=store.write_array,
        time_origin=time_origin,
        origin_panel_revision=origin_panel_revision,
        input_data=store.read_parquet_file(
            "panel", origin_panel_revision, parquet_filename("panel", "panel")
        )
        if origin_panel_revision is not None
        and any(construct.role == "exogenous" for construct in model.constructs)
        else None,
    )
    from nof1_causal_lab.actions.errors import ActionExecutionError
    from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure

    if isinstance(report, ObservationPreflightFailure):
        raise ActionExecutionError(report.message)
    read_simulation_report(store, report, pins["question"])
    return Applied(result=ModelSimulationResult(evidence=report), effects=ActionEffects())


async def _run_simulated_data(
    workspace_id: str,
    store: ArtifactStore,
    source: SimulationReplicateRef,
) -> Applied[DataPreparationResult]:
    """Materialize one recorded simulation replicate; its source owns every input selection."""
    from datetime import timedelta

    from nof1_causal_lab.actions.prepare_data import prepare_simulation_panel
    from nof1_causal_lab.artifacts.data_preparation import SimulationPreparedDataMetadata
    from nof1_causal_lab.study.history import StudyRepository

    record = StudyRepository(workspace_id).record(source.revision)
    if record.record.attempt.action != "simulate" or not isinstance(
        record.record.attempt.outcome, Applied
    ):
        raise StudyLookupError("The source revision must be an applied simulation commit")
    report = record.record.attempt.outcome.result.evidence
    if report.model.workspace_id != workspace_id:
        raise StudyLookupError("The simulation must belong to the current study")
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
            json_filename("panel", "metadata"): SimulationPreparedDataMetadata(
                source=source,
                variables=report.observation_layout.variables,
                time_origin=report.time_origin + timedelta(days=report.times[0]),
            ).model_dump(mode="json")
        },
        parquet_files={parquet_filename("panel", "panel"): panel},
    )
    return Applied(
        result=DataPreparationResult(),
        effects=ActionEffects(produced=(info,)),
    )


async def run_action_locally(
    workspace_id: str,
    request: FitRequest | SimulateRequest | PrepareDataRequest | DataDiffRequest,
    pins: dict[ArtifactId, GitOid],
) -> (
    Applied[ModelFitResult]
    | Applied[ModelSimulationResult]
    | Applied[DataPreparationResult]
    | Applied[None]
):
    """Run a fit, simulation or simulated-data preparation on this process."""
    store = ArtifactStore(workspace_id)
    if isinstance(request, DataDiffRequest):
        from nof1_causal_lab.actions.data_diff import read_data_diff

        await asyncio.to_thread(read_data_diff, workspace_id, request)
        return Applied(result=None, effects=ActionEffects())
    if isinstance(request, FitRequest):
        return await _run_fit(store, pins, request.settings)
    if isinstance(request, SimulateRequest):
        design = SimulationSpec(
            start=request.start, horizon=request.horizon, interventions=request.interventions
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
    request: FitRequest | SimulateRequest | PrepareDataRequest | DataDiffRequest,
    state: StudyState,
) -> (
    Applied[ModelFitResult]
    | Applied[ModelSimulationResult]
    | Applied[DataPreparationResult]
    | Applied[None]
):
    """Pin a request's inputs and run it against the local study store."""
    store = ArtifactStore(workspace_id)
    if isinstance(request, FitRequest):
        # The question's outcome scopes which part of the model the fit learns.
        pins = _pinned(
            store,
            {
                "model": request.model_revision,
                "panel": request.panel_revision,
                "question": state.current["question"].revision,
            },
        )
        from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model
        from nof1_causal_lab.study.store import read_model, read_question

        compile_executable_model(
            StructuralSelection.for_question(
                read_model(store, pins["model"]), read_question(store, pins["question"])
            )
        )
    elif isinstance(request, SimulateRequest):
        pins = _pinned(
            store,
            {"model": request.model_revision, "question": state.current["question"].revision},
        )
        if request.panel_revision is not None:
            pins["panel"] = _pinned(store, {"panel": request.panel_revision})["panel"]
    else:
        pins = {}
    return await run_action_locally(workspace_id, request, pins)
