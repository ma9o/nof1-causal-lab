"""Inputs of the ingestion worker's tools."""

from __future__ import annotations

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value


class ListFilesInput(Value):
    path: str = Field(default=".", description="Relative path within the input directory.")


class ReadFileSampleInput(Value):
    path: str = Field(description="Relative path to the file within the input directory.")
    n_lines: int = Field(default=50, description="Number of lines to read.")


class ExecutePythonInput(Value):
    code: str = Field(description="Python code to execute.")


class SubmitTableInput(Value):
    column_descriptions_json: str = Field(
        description="JSON object mapping column names to descriptions."
    )
