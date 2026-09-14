"""Transition execution against the versioned artifact store.

Each runner receives explicit input pins selected by the machine before
execution. It must read exactly those versions, write new artifact versions with
the same pins in ``derived_from``, and return effects for the workflow to apply.
Heavy transitions can be routed to Modal, but routing is infra-only: it cannot
change the pinned versions or the derivation cascade applied to the result.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.derivations import complete_computed_transition
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import (
    ExecOptions,
    TransitionEffects,
    input_pins,
)
from nof1_causal_lab.machine.store import ArtifactStore

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.identity import ArtifactId, OperationId
    from nof1_causal_lab.machine.artifacts import EpisodeState


def _panel_df(store: ArtifactStore, pins: dict[ArtifactId, int]) -> pl.DataFrame:
    return store.read_parquet_file("panel", pins["panel"], parquet_filename("panel", "panel"))


async def _run_posterior(
    workspace_id: str,
    store: ArtifactStore,
    pins: dict[ArtifactId, int],
    options: ExecOptions,
) -> TransitionEffects:
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.flows.transitions.inference.flow import (
        build_sampler_config,
        run_inference_with_data,
    )
    from nof1_causal_lab.utils.config import get_config

    panel = _panel_df(store, pins)
    from functools import cache

    from nof1_causal_lab.machine.derivations import read_model
    from nof1_causal_lab.machine.inference import inference_input_version

    pins = {**pins, "model": inference_input_version(store, pins["model"])}
    model_spec = read_model(store, pins["model"])
    model_spec.check_execution()

    result = await asyncio.to_thread(
        run_inference_with_data,
        model_spec=model_spec,
        data_for_model=panel,
        sampler_config=build_sampler_config(options.inference_method),
        array_writer=store.write_array,
        array_loader=cache(store.read_array),
        workspace_id=workspace_id,
        compute_loo_diagnostics=get_config().inference.compute_loo_diagnostics,
    )

    conditioned = result.pop("_model")
    evidence = result.pop("engine_evidence")
    report = InferenceReport.model_validate(result)
    info = store.write_version(
        "model",
        provenance="computed",
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


_TRANSITION_RUNNERS = {
    "posterior": _run_posterior,
}

_TEMPORAL_ONLY_TRANSITIONS = frozenset(
    {
        "raw_data",
        "latent_structure",
        "measurement_structure",
        "measurements",
        "statistical_model_spec",
        "baseline_report",
    }
)

_MODAL_TRANSITIONS = frozenset({"posterior"})


async def execute_transition_locally(
    workspace_id: str,
    artifact_id: OperationId,
    pins: dict[ArtifactId, int],
    state: EpisodeState,
    options: ExecOptions,
) -> TransitionEffects:
    """Run a transition on this process against pinned input versions."""
    from nof1_causal_lab.flows.runtime_events import emit_transition_event

    store = ArtifactStore(workspace_id)
    if artifact_id in _TEMPORAL_ONLY_TRANSITIONS:
        raise RuntimeError(f"{artifact_id} is implemented only as a Temporal child workflow")
    runner = _TRANSITION_RUNNERS[artifact_id]
    emit_transition_event(workspace_id, artifact_id, "running")
    try:
        run = await runner(workspace_id, store, pins, options)
        effects = complete_computed_transition(store, state, artifact_id, run.produced)
        effects = effects.model_copy(update={"diagnostics": run.diagnostics})
    except Exception as exc:
        emit_transition_event(
            workspace_id,
            artifact_id,
            "failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    emit_transition_event(workspace_id, artifact_id, "completed")
    return effects


async def execute_transition(
    workspace_id: str,
    artifact_id: OperationId,
    state: EpisodeState,
    options: ExecOptions,
) -> TransitionEffects:
    """Run a transition, routing heavy transitions to Modal in production."""
    spec = transition_spec(artifact_id)
    pins = input_pins(state, spec)
    if artifact_id in _TEMPORAL_ONLY_TRANSITIONS:
        raise RuntimeError(f"{artifact_id} is implemented only as a Temporal child workflow")
    if os.environ.get("DEPLOYMENT_ENV") == "production" and artifact_id in _MODAL_TRANSITIONS:
        from nof1_causal_lab.flows.modal_runners import run_transition_on_modal
        from nof1_causal_lab.machine.derivations import read_model
        from nof1_causal_lab.machine.inference import inference_input_version

        store = ArtifactStore(workspace_id)
        read_model(store, inference_input_version(store, pins["model"])).check_execution()
        return await run_transition_on_modal(workspace_id, artifact_id, pins, state, options)
    return await execute_transition_locally(workspace_id, artifact_id, pins, state, options)
