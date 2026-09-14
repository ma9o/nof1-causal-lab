"""Inference lineage is a journal query, never a field of the scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.machine.moves import RunOperation, is_stale

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord


def inference_record(
    records: Iterable[TransitionRecord], model_version: int
) -> TransitionRecord | None:
    """Find the committed inference operation that produced this exact model value."""
    return next(
        (
            record
            for record in reversed(list(records))
            if record.status == "applied"
            and isinstance(record.move, RunOperation)
            and record.move.operation_id == "posterior"
            and any(
                info.artifact_id == "model" and info.version == model_version
                for info in record.produced
            )
        ),
        None,
    )


def inference_is_current(state: EpisodeState) -> bool:
    """Check the fitted revision's inputs; the model itself has no fitted-status flag."""
    info = state.get("model")
    return (
        info is not None
        and info.produced_by == "run:posterior"
        and state.matches_inputs("model", "panel")
        and not is_stale(state, "panel")
    )


def inference_report_record(
    records: Iterable[TransitionRecord], state: EpisodeState
) -> TransitionRecord | None:
    """Reports can survive in history even when a numerical value was not retained."""
    model = state.get("model")
    if model is None:
        return None
    return next(
        (
            record
            for record in reversed(list(records))
            if record.status == "applied"
            and isinstance(record.move, RunOperation)
            and record.move.operation_id == "posterior"
            and (
                inference_record([record], model.version) is not None
                or (
                    record.diagnostics.get("retention") == "report_only"
                    and record.diagnostics["input_pins"]["model"] == model.version
                )
            )
        ),
        None,
    )


def inference_report_is_current(record: TransitionRecord, state: EpisodeState) -> bool:
    pins = record.diagnostics["input_pins"]
    return (
        state.has("panel")
        and state.current["panel"].version == pins["panel"]
        and not is_stale(state, "panel")
    )


def inference_input_version(store: ArtifactStore, model_version: int) -> int:
    """Refitting restarts from the input law, so observations are counted exactly once."""
    info = store.read_meta("model", model_version)
    while info.produced_by == "run:posterior":
        parent = info.derived_from["model"]
        if parent >= info.version:
            raise ValueError("Inference input revisions must precede their output")
        info = store.read_meta("model", parent)
    return info.version
