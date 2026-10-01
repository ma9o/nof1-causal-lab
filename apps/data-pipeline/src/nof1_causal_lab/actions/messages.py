"""Action-scoped labels from findings already evaluated during execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
    ValidationReportArtifact,
)
from nof1_causal_lab.json_types import JsonObject
from nof1_causal_lab.models.ssm.inference.convergence import convergence_failures
from nof1_causal_lab.study.records import ActionMessage
from nof1_causal_lab.study.store import ArtifactStore

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from nof1_causal_lab.artifacts.identity import ActionId
    from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
    from nof1_causal_lab.study.state import ArtifactRecord


def completion_messages(
    workspace_id: str,
    action: ActionId,
    produced: list[ArtifactRecord],
    diagnostics: Mapping[str, object],
    timestamp: datetime,
    *,
    checks: ModelCheckReport | None = None,
) -> tuple[ActionMessage, ...]:
    """Warnings annotate a completed result; they never decide whether to publish it."""
    labels: dict[str, Literal["debug", "info", "warn"]] = {}
    if checks is not None:
        for finding in checks.specification.findings:
            if finding.check == "model_execution" and finding.status == "not_evaluated":
                labels["MODEL_INCOMPLETE"] = "warn"
            elif finding.check == "dt_ct_approximation_warning":
                labels["DT_CT_APPROXIMATION"] = "warn"
            elif finding.status == "failed":
                labels[
                    "MODEL_NOT_EXECUTABLE"
                    if finding.check == "model_execution"
                    else "FIT_LAWS_UNSUPPORTED"
                ] = "warn"
        predictive = checks.predictive
        if predictive is not None:
            if predictive.status == "failed":
                labels["PREDICTIVE_CHECK_FAILED"] = "warn"
            if predictive.status == "not_evaluated" or any(
                finding.passed is None for finding in predictive.findings
            ):
                labels["SIMULATION_CHECK_NOT_EVALUATED"] = "info"
            if "predictive" in checks.reused:
                labels["PREDICTIVE_CHECKS_REUSED"] = "debug"

    store = ArtifactStore(workspace_id)
    for artifact in produced:
        if artifact.artifact_id == "identification_report":
            report = IdentificationReport.model_validate(
                store.read_json_file(
                    artifact.artifact_id, artifact.revision, "identification_report.json"
                )
            )
            if report.non_identifiable:
                labels["TARGET_NOT_IDENTIFIED"] = "warn"
        elif artifact.artifact_id == "data_profile":
            profile = DataProfileArtifact.model_validate(
                store.read_json_file(
                    "data_profile",
                    artifact.revision,
                    "data_profile.json",
                )
            )
            issues = profile.dataset_issues + [
                issue for audit in profile.indicators.values() for issue in audit.issues
            ]
            if any(issue.severity in {"error", "warning"} for issue in issues):
                labels["DATA_QUALITY_FINDINGS"] = "warn"
        elif artifact.artifact_id == "validation_report":
            validation = ValidationReportArtifact.model_validate(
                store.read_json_file(
                    artifact.artifact_id, artifact.revision, "validation_report.json"
                )
            )
            issues = validation.dataset_issues + [
                issue for audit in validation.indicators.values() for issue in audit.issues
            ]
            if any(issue.severity in {"error", "warning"} for issue in issues):
                labels["DATA_QUALITY_FINDINGS"] = "warn"
            if any(finding.status == "failed" for finding in validation.preflight.findings):
                labels["MODEL_DATA_INCOMPATIBLE"] = "warn"

    if action == "fit" and convergence_failures(
        InferenceReport.model_validate(diagnostics["report"]).inference_diagnostics
    ):
        labels["CONVERGENCE_CHECK_FAILED"] = "warn"
    if action == "prepare_data" and any(
        worker["status"] == "failed"
        for worker in TypeAdapter(list[JsonObject]).validate_python(diagnostics.get("workers", []))
    ):
        labels["EXTRACTION_PARTIAL"] = "warn"
    if action == "prepare_data":
        if diagnostics.get("ingestion_reused"):
            labels["INGESTION_REUSED"] = "info"
        if diagnostics.get("extraction_reused"):
            labels["EXTRACTION_REUSED"] = "info"
    if action == "simulate":
        simulation = TypeAdapter(SimulationReport).validate_python(diagnostics["report"])
        if any(finding.passed is False for finding in simulation.findings):
            labels["PREDICTIVE_CHECK_FAILED"] = "warn"
        if any(finding.passed is None for finding in simulation.findings):
            labels["SIMULATION_CHECK_NOT_EVALUATED"] = "info"
        if simulation.causal_unavailable_reason is not None:
            labels["CAUSAL_EFFECT_NOT_REPORTABLE"] = "info"
    return tuple(
        ActionMessage(timestamp=timestamp, level=level, label=label)
        for label, level in labels.items()
    )
