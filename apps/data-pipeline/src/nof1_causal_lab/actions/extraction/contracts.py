"""Input of the extraction worker's validation tool."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ValidateExtractionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_json: str = Field(
        description="The JSON string containing the worker output to validate."
    )
