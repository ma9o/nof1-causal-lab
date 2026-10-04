"""Cache data-only findings and bind models to their selected observations."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact
from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError
from nof1_causal_lab.models.model_inputs import data_binding_issues
from nof1_causal_lab.study.lineage import read_data_metadata
from nof1_causal_lab.study.state import apply_effects
from nof1_causal_lab.study.store import ArtifactStore, cached_value

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.study.records import Applied, DataPreparationResult
    from nof1_causal_lab.study.state import StudyState


def require_data_binding(store: ArtifactStore, model: ModelSpec, revision: GitOid) -> None:
    issues = data_binding_issues(model, read_data_metadata(store, revision))
    if issues:
        raise AggregatedCompileError(["Selected model and observations are incompatible: " + "; ".join(issues)])


def read_data_profile(store: ArtifactStore, revision: GitOid) -> DataProfileArtifact:
    from nof1_causal_lab.actions.validation.flow import profile_data
    from nof1_causal_lab.study.artifact_files import parquet_filename

    def render() -> DataProfileArtifact:
        return profile_data(store.read_parquet_file("panel", revision, parquet_filename("panel", "panel")),
            metadata=read_data_metadata(store, revision))

    value, _ = cached_value(store.workspace_id, ("data-profile", revision), TypeAdapter(DataProfileArtifact), render)
    return value


def evaluate_data_checks(
    workspace_id: str, state: StudyState, applied: Applied[DataPreparationResult],
) -> DataProfileArtifact:
    """Write through the prepared panel's cache; publish no derived artifact."""
    selected = apply_effects(state, applied.effects.produced, applied.effects.retracted)
    panel = selected.get("panel")
    if panel is None:
        raise IncompleteModelError("Data preparation produced no usable observations")
    return read_data_profile(ArtifactStore(workspace_id), panel.revision)
