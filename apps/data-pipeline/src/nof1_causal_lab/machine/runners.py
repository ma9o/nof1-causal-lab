"""Transition execution against the versioned artifact store.

Each runner receives explicit input pins selected by the machine before
execution. It must read exactly those versions, write new artifact versions with
the same pins in ``derived_from``, and return effects for the workflow to apply.
Heavy transitions can be routed to Modal, but routing is infra-only: it cannot
change pinned versions. The enclosing action evaluates its checks before publication.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, assert_never

from pydantic import TypeAdapter

from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.execution import (
    FitOperation,
    LocalOperation,
    PrepareSimulationOperation,
    SimulateOperation,
    TransitionEffects,
    run_retractions,
)
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.store import ArtifactStore

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
    from nof1_causal_lab.artifacts.simulation import SimulationSpec
    from nof1_causal_lab.machine.artifacts import EpisodeState


def _panel_df(store: ArtifactStore, pins: dict[ArtifactId, GitOid]) -> pl.DataFrame:
    return store.read_parquet_file("panel", pins["panel"], parquet_filename("panel", "panel"))


async def _run_posterior(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    settings: FitSettingsSpec,
) -> TransitionEffects:
    from nof1_causal_lab.actions.fit import (
        build_sampler_config,
        fit,
    )
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.utils.config import get_config

    panel = _panel_df(store, pins)
    from functools import cache

    from nof1_causal_lab.machine.store import read_model

    model_spec = read_model(store, pins["model"])
    model_spec.check_execution()
    from nof1_causal_lab.actions.data_checks import require_data_binding

    require_data_binding(store, model_spec, pins["panel"])
    sampler_config = build_sampler_config()
    sampler_config.update(settings.model_dump(exclude_none=True))

    result = await asyncio.to_thread(
        fit,
        model_spec=model_spec,
        data_for_model=panel,
        sampler_config=sampler_config,
        array_writer=store.write_array,
        array_loader=cache(store.read_array),
        workspace_id=workspace_id,
        compute_loo_diagnostics=get_config().inference.compute_loo_diagnostics,
    )

    conditioned = result.pop("_model")
    evidence = result.pop("engine_evidence")
    report = InferenceReport.model_validate(result)
    info = store.write_artifact(
        "model",
        derived_from=pins,
        produced_by="run:posterior",
        json_files={json_filename("model", "model"): conditioned.model_dump(mode="json")},
    )
    return TransitionEffects(
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
) -> TransitionEffects:
    from nof1_causal_lab.actions.scenarios import summarize_causal_simulation
    from nof1_causal_lab.actions.simulate import simulate
    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.machine.history import StudyRepository
    from nof1_causal_lab.machine.inference import inference_record
    from nof1_causal_lab.machine.store import read_model

    model = read_model(store, pins["model"])
    from nof1_causal_lab.actions.predictive_checks import law_provenance

    report = await asyncio.to_thread(
        simulate,
        model,
        design,
        revision=GitRef(workspace_id=workspace_id, revision=pins["model"], path="model.json"),
        write_array=store.write_array,
    )
    report = report.model_copy(
        update={
            "law": law_provenance(
                store,
                store.read_meta("model", pins["model"]),
                model,
                None,
            )
        }
    )
    report = summarize_causal_simulation(
        model,
        report,
        store=store,
        inference=inference_record(StudyRepository(workspace_id).attempts(), pins["model"]),
    )
    return TransitionEffects(
        diagnostics={
            "input_pins": pins,
            "report": report.model_dump(mode="json"),
        }
    )


async def _run_simulated_measurements(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, GitOid],
    source: SimulationReplicateRef,
) -> TransitionEffects:
    from nof1_causal_lab.actions.prepare_data import prepare_simulation_panel
    from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.machine.history import StudyRepository

    del pins  # The explicit simulation source owns all scientific input selections.
    record = StudyRepository(workspace_id).record(source.revision)
    if record.status != "applied" or record.operation_id != "simulate":
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
        produced_by="run:simulated_measurements",
        json_files={
            json_filename("panel", "metadata"): PreparedDataMetadata(
                source=source,
                variables=report.observation_layout.variables,
            ).model_dump(mode="json")
        },
        parquet_files={parquet_filename("panel", "panel"): panel},
    )
    return TransitionEffects(
        produced=[info],
        diagnostics={
            "input_pins": used_pins,
            "simulation_source": source.model_dump(mode="json"),
            "n_observations": panel["value"].count(),
        },
    )


async def execute_transition_locally(
    workspace_id: str,
    operation: LocalOperation,
    pins: dict[ArtifactId, GitOid],
    state: EpisodeState,
) -> TransitionEffects:
    """Run a transition on this process against pinned input versions."""
    from nof1_causal_lab.flows.runtime_events import emit_transition_event

    store = ArtifactStore(workspace_id)
    operation_id = operation.operation_id
    emit_transition_event(workspace_id, operation_id, "running")
    try:
        if isinstance(operation, FitOperation):
            run = await _run_posterior(workspace_id, store, pins, operation.settings)
        elif isinstance(operation, SimulateOperation):
            run = await _run_simulate(workspace_id, store, pins, operation.design)
        elif isinstance(operation, PrepareSimulationOperation):
            run = await _run_simulated_measurements(workspace_id, store, pins, operation.source)
        else:
            assert_never(operation)
        effects = run.model_copy(
            update={
                "retracted": run_retractions(state, transition_spec(operation_id), run.produced),
            }
        )
    except Exception as exc:
        emit_transition_event(
            workspace_id,
            operation_id,
            "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    emit_transition_event(workspace_id, operation_id, "completed")
    return effects


async def execute_transition(
    workspace_id: str,
    operation: LocalOperation,
    state: EpisodeState,
    selected_inputs: dict[ArtifactId, GitOid] | None = None,
) -> TransitionEffects:
    """Run a transition, routing heavy transitions to Modal in production."""
    spec = transition_spec(operation.operation_id)
    from nof1_causal_lab.machine.selection import resolve_input_pins

    pins = resolve_input_pins(ArtifactStore(workspace_id), state, spec, selected_inputs or {})
    if os.environ.get("DEPLOYMENT_ENV") == "production" and isinstance(operation, FitOperation):
        from nof1_causal_lab.flows.modal_runners import run_transition_on_modal
        from nof1_causal_lab.machine.store import read_model

        store = ArtifactStore(workspace_id)
        read_model(store, pins["model"]).check_execution()
        return await run_transition_on_modal(workspace_id, operation, pins, state)
    return await execute_transition_locally(workspace_id, operation, pins, state)
