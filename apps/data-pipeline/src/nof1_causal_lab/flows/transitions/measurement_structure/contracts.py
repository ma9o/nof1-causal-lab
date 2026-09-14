"""Whole-model submission for the measurement structure operation."""

from nof1_causal_lab.flows.contracts_base import ToolDefinition
from nof1_causal_lab.flows.model_authoring import ModelSubmission

MEASUREMENT_STRUCTURE_TOOL_CONTRACTS: list[ToolDefinition] = [
    ToolDefinition(
        name="validate_measurement_structure",
        description="Validate the full candidate Model with its existing identities and owned details.",
        input_schema=ModelSubmission,
    ),
]
