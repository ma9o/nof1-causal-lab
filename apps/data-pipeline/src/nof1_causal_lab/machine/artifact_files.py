"""Canonical file layout for versioned machine artifacts.

The artifact graph names semantic dependencies such as ``panel`` or
``model``. This module is the single map from those artifact ids to the
files inside ``store/{artifact_id}/v{N}/``. UI projections, fixture seeders,
stage runners, and tool contexts should refer to this map instead of spelling
filenames independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ArtifactId


@dataclass(frozen=True)
class ArtifactFileSpec:
    """An artifact file specification declares its JSON payloads and tables."""

    json: dict[str, str] = field(default_factory=dict)
    parquet: dict[str, str] = field(default_factory=dict)

    def all_filenames(self) -> frozenset[str]:
        return frozenset([*self.json.values(), *self.parquet.values()])


ARTIFACT_FILE_SPECS: dict[ArtifactId, ArtifactFileSpec] = {
    "raw_data": ArtifactFileSpec(parquet={"raw": "raw.parquet"}),
    "model": ArtifactFileSpec(json={"model": "model.json"}),
    "identification_report": ArtifactFileSpec(
        json={"identification_report": "identification_report.json"}
    ),
    "panel": ArtifactFileSpec(parquet={"panel": "panel.parquet"}),
    "validation_report": ArtifactFileSpec(json={"validation_report": "validation_report.json"}),
}


def artifact_file_spec(artifact_id: ArtifactId) -> ArtifactFileSpec:
    return ARTIFACT_FILE_SPECS[artifact_id]


def json_filename(artifact_id: ArtifactId, key: str) -> str:
    return artifact_file_spec(artifact_id).json[key]


def parquet_filename(artifact_id: ArtifactId, key: str) -> str:
    return artifact_file_spec(artifact_id).parquet[key]


def is_declared_artifact_file(artifact_id: ArtifactId, filename: str) -> bool:
    return filename in artifact_file_spec(artifact_id).all_filenames()
