"""extraction contracts and tool metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.flows.contracts_base import ToolDefinition


class ValidateExtractionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_json: str = Field(
        description="The JSON string containing the worker output to validate."
    )


EXTRACTION_TOOL_CONTRACTS: list[ToolDefinition] = [
    ToolDefinition(
        name="validate_extractions",
        description="Tool for validating worker extraction output JSON.",
        input_schema=ValidateExtractionsInput,
    ),
]
