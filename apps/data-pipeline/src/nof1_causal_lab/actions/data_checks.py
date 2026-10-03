"""Data-only checks and explicit model/observation schema binding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError
from nof1_causal_lab.models.model_inputs import data_binding_issues
from nof1_causal_lab.study.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.study.lineage import read_data_metadata
from nof1_causal_lab.study.records import Applied
from nof1_causal_lab.study.state import RetractedArtifact, apply_effects
from nof1_causal_lab.study.store import ArtifactStore

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.study.records import DataPreparationResult
    from nof1_causal_lab.study.state import StudyState


def require_data_binding(store: ArtifactStore, model: ModelSpec, revision: GitOid) -> None:
    issues = data_binding_issues(model, read_data_metadata(store, revision))
    if issues:
        raise AggregatedCompileError(
            ["Selected model and observations are incompatible: " + "; ".join(issues)]
        )


def evaluate_data_checks(
    workspace_id: str, state: StudyState, applied: Applied[DataPreparationResult]
) -> Applied[DataPreparationResult]:
    """Profile newly prepared observations without reading or evaluating any model."""
    from nof1_causal_lab.actions.validation.flow import profile_data

    store = ArtifactStore(workspace_id)
    selected = apply_effects(state, applied.effects.produced, applied.effects.retracted)
    panel = selected.get("panel")
    if panel is None:
        raise IncompleteModelError("Data preparation produced no usable observations")
    metadata = read_data_metadata(store, panel.revision)
    data = store.read_parquet_file("panel", panel.revision, parquet_filename("panel", "panel"))
    profile = profile_data(data, metadata=metadata)
    report = store.write_artifact(
        "data_profile",
        derived_from={"panel": panel.revision},
        produced_by="check:data_profile",
        json_files={json_filename("data_profile", "data_profile"): profile.model_dump(mode="json")},
    )
    retracted = list(applied.effects.retracted)
    if selected.has("validation_report"):
        retracted.append(
            RetractedArtifact(artifact_id="validation_report", reason_ref="panel.changed")
        )
    return Applied(
        result=applied.result,
        effects=applied.effects.with_checks(
            produced=(*applied.effects.produced, report),
            retracted=tuple(retracted),
            checks=applied.effects.checks,
        ),
    )
