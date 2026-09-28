"""Read retained historical authoring findings from the operation journal."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.machine.execution import is_stale
from nof1_causal_lab.machine.inference import inference_input_revision

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord


def model_spec_record[T: TransitionRecord](records: Iterable[T]) -> T | None:
    """Select the latest successfully completed model-authoring operation."""
    return next(
        (
            record
            for record in reversed(list(records))
            if record.status == "applied"
            and record.operation_id == "statistical_model_spec"
            and any(info.artifact_id == "model" for info in record.produced)
        ),
        None,
    )


def model_spec_is_current(
    record: TransitionRecord, state: EpisodeState, store: ArtifactStore
) -> bool:
    """Match the checked prior model and every other input against the selected state."""
    if not state.has("model"):
        return False
    checked = next(info for info in record.produced if info.artifact_id == "model")
    selected = store.read_meta(
        "model", inference_input_revision(store, state.current["model"].revision)
    )
    return checked.model_inputs["belief"] == selected.model_inputs["belief"] and all(
        state.has(artifact_id)
        and state.current[artifact_id].revision == revision
        and not is_stale(state, artifact_id)
        for artifact_id, revision in checked.derived_from.items()
        if artifact_id != "model"
    )
