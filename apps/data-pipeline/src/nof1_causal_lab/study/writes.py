"""Optimistic model revision storage shared by scientific execution jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS
from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.errors import ArtifactWriteRejected

if TYPE_CHECKING:
    from pydantic import BaseModel

    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.json_types import JsonObject
    from nof1_causal_lab.study.state import ArtifactRecord, StudyState
    from nof1_causal_lab.study.store import ArtifactStore


def _validated(
    artifact_id: ArtifactId,
    model_cls: type[BaseModel],
    payload: object,
) -> JsonObject:
    try:
        return model_cls.model_validate(payload).model_dump(mode="json")
    except Exception as exc:
        raise ArtifactWriteRejected(str(exc), artifact_id=artifact_id) from exc


def write_model_revision(
    store: ArtifactStore,
    state: StudyState,
    payload: object,
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
    validated = _validated("model", ARTIFACT_CONTRACTS["model"], payload)
    return store.write_artifact(
        "model",
        derived_from=derived_from,
        produced_by=produced_by,
        json_files={json_filename("model", "model"): validated},
    )
