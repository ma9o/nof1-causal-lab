"""Optimistic model revision storage shared by scientific execution jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.errors import ArtifactWriteRejected

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.study.state import ArtifactRecord, StudyState
    from nof1_causal_lab.study.store import ArtifactStore


def write_model_revision(
    store: ArtifactStore,
    state: StudyState,
    payload: ModelSpec,
    *,
    expected_model_revision: GitOid | None,
    derived_from: dict[ArtifactId, GitOid],
    produced_by: str | None,
) -> ArtifactRecord:
    """The common optimistic commit boundary for model revisions."""
    from nof1_causal_lab.study.state import validate_model_base

    if reason := validate_model_base(state, expected_model_revision):
        raise ArtifactWriteRejected(reason, artifact_id="model")
    if derived_from.get("model") != expected_model_revision:
        raise ArtifactWriteRejected(
            "Model inputs must name the expected base revision", artifact_id="model"
        )
    return store.write_artifact(
        "model",
        derived_from=derived_from,
        produced_by=produced_by,
        json_files={json_filename("model", "model"): payload.model_dump(mode="json")},
    )
