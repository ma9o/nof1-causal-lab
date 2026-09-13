"""Authored descriptions of uploaded columns."""

from pydantic import BaseModel, ConfigDict

from .base import ArtifactPayload


class ColumnDescription(BaseModel):
    """A column description explains the meaning of one column in the uploaded dataset."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str


class RawDataArtifact(ArtifactPayload):
    """This artifact describes the uploaded dataset's columns for downstream scientific
    interpretation.
    """

    column_descriptions: list[ColumnDescription]
