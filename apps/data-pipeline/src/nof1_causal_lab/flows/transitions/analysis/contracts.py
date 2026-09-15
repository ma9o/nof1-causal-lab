"""Read-only fitted-model introspection. Computation belongs to scientific actions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.flows.contracts_base import ToolDefinition

ModelInfoSection = Literal[
    "overview",
    "variables",
    "measurement",
    "identifiability",
    "diagnostics",
    "capabilities",
]


def _default_model_info_sections() -> list[ModelInfoSection]:
    return ["overview", "variables", "capabilities"]


class GetModelInfoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[ModelInfoSection] = Field(
        default_factory=_default_model_info_sections,
        description="Named sections to include in the read-only model summary.",
    )
    names: list[str] = Field(
        default_factory=list,
        description="Optional construct or indicator names to focus the summary on.",
    )


ANALYSIS_TOOL_CONTRACTS: list[ToolDefinition] = [
    ToolDefinition(
        name="get_model_info",
        description=(
            "Return a read-only summary of the fitted model, variables with persistent IDs, identifiability status, "
            "and diagnostics."
        ),
        input_schema=GetModelInfoInput,
    ),
]
