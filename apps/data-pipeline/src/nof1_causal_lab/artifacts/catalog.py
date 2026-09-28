"""Canonical JSON payload contracts for every file-backed machine artifact.

Prepared panels include data-owned metadata beside their Parquet observations.
The machine file catalog declares all JSON, Parquet, and binary members.
"""

from pydantic import BaseModel

from .data_preparation import PreparedDataMetadata
from .identification import IdentificationReport
from .identity import ArtifactId
from .model_spec import ModelSpec
from .validation_report import DataProfileArtifact, ValidationReportArtifact

ARTIFACT_CONTRACTS: dict[ArtifactId, type[BaseModel]] = {
    "panel": PreparedDataMetadata,
    "model": ModelSpec,
    "identification_report": IdentificationReport,
    "data_profile": DataProfileArtifact,
    "validation_report": ValidationReportArtifact,
}
