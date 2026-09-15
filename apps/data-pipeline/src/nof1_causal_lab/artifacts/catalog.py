"""Canonical JSON payload contracts for every file-backed machine artifact.

Raw data and panel are Parquet tables with no separate JSON payload.
The machine file catalog declares all JSON, Parquet, and binary members.
"""

from pydantic import BaseModel

from .identification import IdentificationReport
from .identity import ArtifactId
from .model_spec import ModelSpec
from .validation_report import DataProfileArtifact, ValidationReportArtifact

ARTIFACT_CONTRACTS: dict[ArtifactId, type[BaseModel]] = {
    "model": ModelSpec,
    "identification_report": IdentificationReport,
    "data_profile": DataProfileArtifact,
    "validation_report": ValidationReportArtifact,
}
