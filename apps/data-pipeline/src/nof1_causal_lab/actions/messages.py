"""Action-scoped labels from findings already evaluated during execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from nof1_causal_lab.artifacts.availability import Unavailable
from nof1_causal_lab.artifacts.checks import Evaluated, NotEvaluated
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.model_checks import EvaluatedPredictiveChecks
from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
    ValidationReportArtifact,
)
from nof1_causal_lab.models.ssm.inference.convergence import convergence_failures
from nof1_causal_lab.study.records import ActionMessage, DataPreparationResult

if TYPE_CHECKING:
    from datetime import datetime

    from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
    from nof1_causal_lab.artifacts.posterior import InferenceReport, ModelFitResult
    from nof1_causal_lab.artifacts.simulation import ModelSimulationResult, SimulationReport


_QUESTION_LABELS = {
    "outcome": "QUESTION_MODEL_INCOMPATIBLE",
    "target": "QUESTION_MODEL_INCOMPATIBLE",
    "identification": "QUERY_NOT_IDENTIFIED",
    "window": "QUERY_OUTSIDE_RECORD",
    "range": "INTERVENTION_OUTSIDE_RECORDED_RANGE",
}


def completion_messages(
    result: DataPreparationResult | ModelFitResult | ModelSimulationResult | None,
    timestamp: datetime,
    reports: tuple[IdentificationReport | DataProfileArtifact | ValidationReportArtifact, ...] = (),
    *,
    checks: ModelCheckReport | None = None,
    inference: InferenceReport | None = None,
    simulation: SimulationReport | None = None,
) -> tuple[ActionMessage, ...]:
    """Warnings annotate a completed result; they never decide whether to publish it."""
    labels: dict[str, Literal["debug", "info", "warn"]] = {}
    if checks is not None:
        for finding in checks.specification:
            if finding.subject == "model_execution" and isinstance(finding, NotEvaluated):
                labels["MODEL_INCOMPLETE"] = "warn"
            elif finding.subject == "dt_ct_approximation_warning":
                labels["DT_CT_APPROXIMATION"] = "warn"
            elif isinstance(finding, Evaluated) and finding.outcome in {"failed", "error"}:
                labels[
                    "MODEL_NOT_EXECUTABLE"
                    if finding.subject == "model_execution"
                    else "FIT_LAWS_UNSUPPORTED"
                ] = "warn"
        question = checks.question
        for finding in question.findings if question is not None else ():
            if isinstance(finding, NotEvaluated):
                if finding.reason == "CONSTRUCT_UNDEFINED":
                    labels["QUESTION_CONSTRUCTS_UNDEFINED"] = "info"
            elif finding.outcome in {"failed", "error"}:
                labels[_QUESTION_LABELS[finding.subject.check]] = "warn"
        predictive = checks.predictive
        if predictive is not None:
            if predictive.status == "failed":
                labels["PREDICTIVE_CHECK_FAILED"] = "warn"
            if predictive.status == "not_evaluated" or (
                isinstance(predictive.evaluation, EvaluatedPredictiveChecks)
                and any(
                    isinstance(finding, NotEvaluated) for finding in predictive.evaluation.findings
                )
            ):
                labels["SIMULATION_CHECK_NOT_EVALUATED"] = "info"
            if "predictive" in checks.reused:
                labels["PREDICTIVE_CHECKS_REUSED"] = "debug"

    for report in reports:
        if isinstance(report, IdentificationReport):
            if report.non_identifiable:
                labels["TARGET_NOT_IDENTIFIED"] = "warn"
        else:
            data = report.data if isinstance(report, ValidationReportArtifact) else report
            issues = (
                *data.dataset_issues,
                *(issue for audit in data.indicators.values() for issue in audit.issues),
            )
            if any(issue.severity in {"error", "warning"} for issue in issues):
                labels["DATA_QUALITY_FINDINGS"] = "warn"
            if isinstance(report, ValidationReportArtifact) and any(
                isinstance(finding, Evaluated) and finding.outcome in {"failed", "error"}
                for finding in report.preflight
            ):
                labels["MODEL_DATA_INCOMPATIBLE"] = "warn"

    if inference is not None and convergence_failures(inference.core.convergence):
        labels["CONVERGENCE_CHECK_FAILED"] = "warn"
    if isinstance(result, DataPreparationResult):
        if any(worker.status == "failed" for worker in result.workers):
            labels["EXTRACTION_PARTIAL"] = "warn"
        if result.extraction_reused:
            labels["EXTRACTION_REUSED"] = "info"
    if simulation is not None:
        if any(
            isinstance(finding, Evaluated) and finding.outcome in {"failed", "error"}
            for finding in simulation.findings
        ):
            labels["PREDICTIVE_CHECK_FAILED"] = "warn"
        if any(isinstance(finding, NotEvaluated) for finding in simulation.findings):
            labels["SIMULATION_CHECK_NOT_EVALUATED"] = "info"
        if isinstance(simulation.causal, Unavailable):
            labels["CAUSAL_EFFECT_NOT_REPORTABLE"] = "info"
    return tuple(
        ActionMessage(timestamp=timestamp, level=level, label=label)
        for label, level in labels.items()
    )
