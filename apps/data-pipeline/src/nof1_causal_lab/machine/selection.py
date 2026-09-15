"""Resolve deliberate historical selections without moving the current workspace cursor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.machine.moves import input_pins

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.graph import Transition
    from nof1_causal_lab.machine.store import ArtifactStore


def resolve_input_pins(
    store: ArtifactStore, state: EpisodeState, spec: Transition, selected: dict[ArtifactId, int]
) -> dict[ArtifactId, int]:
    permitted = {*spec.consumes, *spec.optional_consumes}
    if set(selected) - permitted:
        raise ValueError(f"Unexpected inputs for {spec.operation_id}: {set(selected) - permitted}")
    revisions = [store.read_meta(identity, version) for identity, version in selected.items()]
    selected_state = state.with_versions(revisions)
    pins = input_pins(selected_state, spec)
    # An observation table is reusable across parameter/dynamics edits, but only for
    # the measurement definitions which produced it. This checks the selected pair.
    if (
        "panel" in pins
        and "model" in pins
        and spec.operation_id == "posterior"
        and not selected_state.matches_inputs("panel", "model")
    ):
        raise ValueError(
            "Selected observations were prepared for different measurement definitions"
        )
    return pins
