"""Canonical JSON payload contracts for every file-backed machine artifact.

Prepared panels include data-owned metadata beside their Parquet observations.
The machine file catalog declares all JSON, Parquet, and binary members.
"""

from typing import TypeAliasType

from pydantic import BaseModel

from .data_preparation import PreparedDataMetadata
from .identity import ArtifactId
from .model_spec import ModelSpec
from .question import QuestionSpec

ARTIFACT_CONTRACTS: dict[ArtifactId, type[BaseModel] | TypeAliasType] = {
    "question": QuestionSpec,
    "panel": PreparedDataMetadata,
    "model": ModelSpec,
}
