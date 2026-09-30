"""Resolve deliberate historical selections without moving the current workspace cursor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.machine.execution import input_pins

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.graph import Transition
    from nof1_causal_lab.machine.store import ArtifactStore


def resolve_input_pins(
    store: ArtifactStore, state: EpisodeState, spec: Transition, selected: dict[ArtifactId, GitOid]
) -> dict[ArtifactId, GitOid]:
    permitted = {*spec.consumes, *spec.optional_consumes}
    if set(selected) - permitted:
        raise ValueError(f"Unexpected inputs for {spec.operation_id}: {set(selected) - permitted}")
    revisions = [store.read_meta(identity, revision) for identity, revision in selected.items()]
    selected_state = state.with_artifacts(revisions)
    return input_pins(selected_state, spec)
