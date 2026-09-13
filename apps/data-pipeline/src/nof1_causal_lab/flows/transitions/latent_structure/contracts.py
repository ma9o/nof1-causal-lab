"""latent-structure contracts and tool metadata."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.flows.contracts_base import ToolDefinition


class ValidateLatentStructureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    structure_json: str = Field(
        description="The JSON string containing the latent structure to validate."
    )


LATENT_STRUCTURE_TOOL_CONTRACTS: list[ToolDefinition] = [
    ToolDefinition(
        name="validate_latent_structure",
        description="Tool for validating latent structure JSON (latent-structure).",
        input_schema=ValidateLatentStructureInput,
    ),
]
