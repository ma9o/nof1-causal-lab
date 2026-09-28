"""Inference lineage is a journal query, never a field of the scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.machine.execution import is_stale

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord


def inference_record[T: TransitionRecord](records: Iterable[T], model_revision: GitOid) -> T | None:
    """Find the committed inference operation that produced this exact model value."""
    return next(
        (
            record
            for record in reversed(list(records))
            if record.status == "applied"
            and record.operation_id == "posterior"
            and any(
                info.artifact_id == "model" and info.revision == model_revision
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


def inference_report_record[T: TransitionRecord](
    records: Iterable[T], state: EpisodeState
) -> T | None:
    """Reports can survive in history even when a numerical value was not retained."""
    model = state.get("model")
    if model is None:
        return None
    return next(
        (
            record
            for record in reversed(list(records))
            if record.status == "applied"
            and record.operation_id == "posterior"
            and (
                inference_record([record], model.revision) is not None
                or (
                    record.diagnostics.get("retention") == "report_only"
                    and record.diagnostics["input_pins"]["model"] == model.revision
                )
            )
        ),
        None,
    )


def inference_report_is_current(record: TransitionRecord, state: EpisodeState) -> bool:
    pins = record.diagnostics["input_pins"]
    return (
        state.has("panel")
        and state.current["panel"].revision == pins["panel"]
        and not is_stale(state, "panel")
    )


def inference_input_revision(store: ArtifactStore, model_revision: GitOid) -> GitOid:
    """Refitting restarts from the input law, so observations are counted exactly once."""
    info = store.read_meta("model", model_revision)
    while info.produced_by == "run:posterior" or "model" in info.derived_from:
        parent = info.derived_from["model"]
        previous = store.read_meta("model", parent)
        if (
            info.produced_by != "run:posterior"
            and info.model_inputs["belief"] != previous.model_inputs["belief"]
        ):
            break
        info = previous
    return info.revision
