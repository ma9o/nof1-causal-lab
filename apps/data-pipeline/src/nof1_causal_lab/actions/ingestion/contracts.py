"""Inputs of the ingestion worker's tools."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ListFilesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(default=".", description="Relative path within the input directory.")


class ReadFileSampleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Relative path to the file within the input directory.")
    n_lines: int = Field(default=50, description="Number of lines to read.")


class ExecutePythonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Python code to execute.")


class SubmitTableInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column_descriptions_json: str = Field(
        description="JSON object mapping column names to descriptions."
    )
