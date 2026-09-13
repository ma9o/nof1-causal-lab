"""measurement-structure contracts and tool metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.flows.contracts_base import ToolDefinition


class ValidateMeasurementStructureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    measurement_json: str = Field(
        description=(
            "The JSON string containing the measurement structure and known-input "
            "declarations to validate."
        )
    )


MEASUREMENT_STRUCTURE_TOOL_CONTRACTS: list[ToolDefinition] = [
    ToolDefinition(
        name="validate_measurement_structure",
        description=(
            "Validate measurement structure, known-input declarations, and compiler constraints."
        ),
        input_schema=ValidateMeasurementStructureInput,
    ),
]
