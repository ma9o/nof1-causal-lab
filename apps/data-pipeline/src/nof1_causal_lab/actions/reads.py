"""Read a completed action's scientific output from its immutable checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_lab.actions.results import (
    ActionBody,
    DataComparisonResult,
    DataPreparationResult,
    ModelEditResult,
    ModelFitResult,
    ModelSimulationResult,
)
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
)
from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.views import measurements_view

if TYPE_CHECKING:
    from nof1_causal_lab.study.records import StudyRevision


def read_action_body(workspace_id: str, record: StudyRevision) -> ActionBody:
    """Read output only after publication; never regenerate scientific evidence."""
    if record.status != "applied":
        raise ValueError("An unsuccessful attempt has no scientific response body")
    if record.action == "data_diff":
        from nof1_causal_lab.actions.data_diff import DataDiffReport
        from nof1_causal_lab.study.history import StudyRepository

        return DataComparisonResult(
            commit_id=record.commit_id,
            report=DataDiffReport.model_validate_json(
                StudyRepository(workspace_id).read_file(record.commit_id, "logs/data-diff.json")
            ),
        )
    if record.action == "simulate":
        return ModelSimulationResult(
            commit_id=record.commit_id,
            report=TypeAdapter(SimulationReport).validate_python(record.diagnostics["report"]),
        )
    reader = ModelReader(workspace_id, at=record.commit_id, branch=record.branch)
    if record.action == "edit_model":
        assert reader.model is not None
        assert record.checks is not None
        assert record.checks.predictive is not None
        identification = reader.identification()
        assert identification is not None
        validation = reader.validation_report
        return ModelEditResult(
            commit_id=record.commit_id,
            model_revision=reader.state.current["model"].revision,
            model=reader.model,
            specification=record.checks.specification,
            predictive=record.checks.predictive,
            identification=identification.value,
            validation=validation.value if validation is not None else None,
        )
    if record.action == "fit":
        assert reader.model is not None
        assert record.checks is not None
        identification = reader.identification()
        assert identification is not None
        return ModelFitResult(
            commit_id=record.commit_id,
            model_revision=reader.state.current["model"].revision,
            model=reader.model,
            report=InferenceReport.model_validate(record.diagnostics["report"]),
            specification=record.checks.specification,
            identification=identification.value,
        )
    from nof1_causal_lab.study.lineage import read_data_metadata

    artifact = next(item for item in record.produced if item.artifact_id == "panel")
    panel = reader.store.read_parquet_file("panel", artifact.revision, "panel.parquet")
    profile = reader.state.current["data_profile"]
    return DataPreparationResult(
        commit_id=record.commit_id,
        data_revision=GitRef(
            workspace_id=workspace_id, revision=artifact.revision, path="panel.parquet"
        ),
        data=measurements_view(panel, set(panel["indicator_id"].to_list())),
        metadata=read_data_metadata(reader.store, artifact.revision),
        profile=DataProfileArtifact.model_validate(
            reader.store.read_json_file(
                "data_profile",
                profile.revision,
                json_filename("data_profile", "data_profile"),
            )
        ),
    )
