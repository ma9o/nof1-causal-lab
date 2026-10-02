"""Public tool contracts: the scientific actions, fitted-model introspection and literature search."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from nof1_causal_lab.actions.contracts import scientific_tool_contracts
from nof1_causal_lab.actions.tool_definition import ToolDefinition
from nof1_causal_lab.artifacts.base import Value

ModelInfoSection = Literal[
    "overview",
    "variables",
    "measurement",
    "identifiability",
    "diagnostics",
    "capabilities",
]


def _default_model_info_sections() -> tuple[ModelInfoSection, ...]:
    return ("overview", "variables", "capabilities")


class GetModelInfoInput(Value):
    sections: tuple[ModelInfoSection, ...] = Field(
        default_factory=_default_model_info_sections,
        description="Named sections to include in the read-only model summary.",
    )
    names: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Optional construct or indicator names to focus the summary on.",
    )


class SearchLiteratureInput(Value):
    query: str = Field(description="Search query for empirical literature about effect sizes.")


CONTEXT_TOOLS: dict[str, list[ToolDefinition]] = {
    "scientific": scientific_tool_contracts(),
    "literature": [
        ToolDefinition(
            name="search_literature",
            description="Search for empirical literature about effect sizes for model parameters.",
            input_schema=SearchLiteratureInput,
        )
    ],
    "analysis": [
        ToolDefinition(
            name="get_model_info",
            description=(
                "Return a read-only summary of the fitted model, variables with persistent IDs, "
                "identifiability status, and diagnostics."
            ),
            input_schema=GetModelInfoInput,
        )
    ],
}
