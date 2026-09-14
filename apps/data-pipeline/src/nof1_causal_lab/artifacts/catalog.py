"""Canonical JSON payload contracts for every file-backed machine artifact.

Raw data and panel are Parquet tables with no separate JSON payload.
The machine file catalog declares all JSON, Parquet, and binary members.
"""

from pydantic import BaseModel

from .admission import AdmissionReport
from .baseline_report import BaselineReportArtifact
from .identification import IdentificationReport
from .identity import ArtifactId
from .model_spec import ModelSpec
from .question import QuestionArtifact
from .validation_report import ValidationReportArtifact

ARTIFACT_CONTRACTS: dict[ArtifactId, type[BaseModel]] = {
    "question": QuestionArtifact,
    "model": ModelSpec,
    "identification_report": IdentificationReport,
    "validation_report": ValidationReportArtifact,
    "admission_report": AdmissionReport,
    "baseline_report": BaselineReportArtifact,
}
