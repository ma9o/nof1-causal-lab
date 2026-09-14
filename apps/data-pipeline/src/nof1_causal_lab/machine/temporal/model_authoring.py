"""Commit canonical authoring results through one validated revision boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.machine.derivations import complete_computed_transition
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.writes import write_model_revision
from nof1_causal_lab.utils import storage

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import OperationId
    from nof1_causal_lab.machine.moves import TransitionEffects
    from nof1_causal_lab.machine.temporal.messages import SingleLLMTransitionFinalizeInput


def finalize_model_revision(
    input: SingleLLMTransitionFinalizeInput, operation_id: OperationId
) -> TransitionEffects:
    if input.result_ref is None:
        raise RuntimeError(f"{operation_id} completed without a model result ref")
    store = ArtifactStore(input.workspace_id)
    info = write_model_revision(
        store,
        input.state,
        storage.read_json(input.result_ref),
        provenance="computed",
        expected_model_version=input.pins.get("model", 0),
        derived_from=input.pins,
        produced_by=f"run:{operation_id}",
    )
    return complete_computed_transition(store, input.state, operation_id, [info])
