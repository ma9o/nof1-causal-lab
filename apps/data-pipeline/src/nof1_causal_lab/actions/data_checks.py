"""Compute data-only findings while preparing observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.study.data import read_data_history
from nof1_causal_lab.study.state import apply_effects
from nof1_causal_lab.study.store import ArtifactStore

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact
    from nof1_causal_lab.study.records import Applied, DataPreparationResult
    from nof1_causal_lab.study.state import StudyState


def read_data_profile(store: ArtifactStore, source: DataRef[GitOid, int]) -> DataProfileArtifact:
    """Compute the empirical data profile for one exact replicate selection."""
    from nof1_causal_lab.actions.validation.flow import profile_data

    def render() -> DataProfileArtifact:
        history = read_data_history(store, source)
        return profile_data(
            history.observations.recorded.frame,
            definitions=history.variables,
            metadata=history.metadata,
        )

    return render()


def evaluate_data_checks(
    workspace_id: str,
    state: StudyState,
    applied: Applied[DataPreparationResult],
) -> DataProfileArtifact:
    """Return the prepared panel's report for publication with its action."""
    selected = apply_effects(state, applied.effects.produced, applied.effects.retracted)
    panel = selected.get("panel")
    if panel is None:
        raise IncompleteModelError("Data preparation produced no usable observations")
    return read_data_profile(
        ArtifactStore(workspace_id),
        DataRef[GitOid, int](revision=panel.revision, replicate_index=0),
    )
