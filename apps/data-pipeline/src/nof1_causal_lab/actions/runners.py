"""Run fits and simulations against exact input revisions.

Each runner reads exactly its pinned input versions, writes new artifact versions with
the same pins in ``derived_from``, and returns effects for the workflow to publish.
Local studies can place numerical fitting on Modal without moving their history
or changing pinned versions. The enclosing action evaluates its checks before publication.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from nof1_causal_lab.actions.contracts import (
    DataDiffRequest,
    FitRequest,
    ModelDiffRequest,
    SimulateRequest,
)
from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.posterior import ModelFitResult
from nof1_causal_lab.artifacts.simulation import ModelSimulationResult
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.store import ArtifactStore

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
    from nof1_causal_lab.artifacts.simulation import SimulationSpec
    from nof1_causal_lab.study.state import StudyState


async def _run_fit(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    settings: FitSettingsSpec,
    source: DataRef[GitOid, int],
) -> Applied[ModelFitResult]:
    from nof1_causal_lab.actions.fit import fit, read_inference_report
    from nof1_causal_lab.actions.inference.fit import resolve_sampler_spec
    from nof1_causal_lab.study.data import read_data_history
    from nof1_causal_lab.utils.config import get_config

    history = read_data_history(store, source)
    from functools import cache

    from nof1_causal_lab.study.store import read_model, read_question

    model_spec = read_model(store, pins["model"])
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
        data_for_model=history.observations,
        time_origin=history.time_origin,
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
    retained = ModelFitResult(model=store.model_ref(pins["model"]), data=source, evidence=evidence)
    report = read_inference_report(store, info.revision, retained, result["metadata"])
    return Applied(
        result=retained,
        effects=ActionEffects(produced=(info,), reports={"inference": store.write_report(report)}),
    )


async def _run_simulate(
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    design: SimulationSpec,
) -> Applied[ModelSimulationResult]:
    from nof1_causal_lab.actions.simulate import read_simulation_report, simulate
    from nof1_causal_lab.study.store import read_model, read_question

    model = read_model(store, pins["model"])
    selection = StructuralSelection.for_question(model, read_question(store, pins["question"]))
    report = await asyncio.to_thread(
        simulate,
        selection,
        design,
        revision=store.model_ref(pins["model"]),
    )
    from nof1_causal_lab.actions.errors import ActionExecutionError
    from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure

    if isinstance(report, ObservationPreflightFailure):
        raise ActionExecutionError(report.message)
    retained = ModelSimulationResult(evidence=report)
    findings = read_simulation_report(store, retained.evidence, pins["question"])
    return Applied(
        result=retained,
        effects=ActionEffects(reports={"simulation": store.write_report(findings)}),
    )


async def run_action_locally(
    workspace_id: str,
    request: FitRequest[GitOid]
    | SimulateRequest[GitOid]
    | DataDiffRequest[GitOid]
    | ModelDiffRequest[GitOid],
    pins: dict[ArtifactId, GitOid],
) -> Applied[ModelFitResult] | Applied[ModelSimulationResult] | Applied[None]:
    """Run an action directly against its recorded inputs."""
    store = ArtifactStore(workspace_id)
    if isinstance(request, ModelDiffRequest):
        from nof1_causal_lab.actions.revisions import read_model_diff

        report = await asyncio.to_thread(
            read_model_diff, workspace_id, request.input.before_ref, request.input.after_ref
        )
        return Applied(
            result=None, effects=ActionEffects(reports={"model-diff": store.write_result(report)})
        )
    if isinstance(request, DataDiffRequest):
        from nof1_causal_lab.actions.data_diff import read_data_diff

        comparison = await asyncio.to_thread(read_data_diff, workspace_id, request)
        return Applied(
            result=None,
            effects=ActionEffects(reports={"data-diff": store.write_report(comparison)}),
        )
    if isinstance(request, FitRequest):
        return await _run_fit(
            store,
            pins,
            request.input.settings,
            DataRef[GitOid, int](
                revision=request.input.data_ref, replicate_index=request.input.replicate_index
            ),
        )
    return await _run_simulate(store, pins, request.input.simulation)


def _pinned(store: ArtifactStore, selected: dict[ArtifactId, GitOid]) -> dict[ArtifactId, GitOid]:
    """Exact selected input revisions; every one must exist in the store."""
    for artifact_id, revision in selected.items():
        store.read_meta(artifact_id, revision)
    return dict(selected)


async def run_action(
    workspace_id: str,
    request: FitRequest[GitOid]
    | SimulateRequest[GitOid]
    | DataDiffRequest[GitOid]
    | ModelDiffRequest[GitOid],
    state: StudyState,
) -> Applied[ModelFitResult] | Applied[ModelSimulationResult] | Applied[None]:
    """Pin a request's inputs and run it against the local study store."""
    store = ArtifactStore(workspace_id)
    if isinstance(request, FitRequest):
        pins = _pinned(
            store,
            {
                "model": request.input.model_ref,
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
            {"model": request.input.model_ref, "question": state.current["question"].revision},
        )
    else:
        pins = {}
    return await run_action_locally(workspace_id, request, pins)
