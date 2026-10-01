"""Public tool contracts: the scientific actions, fitted-model introspection and literature search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.actions.contracts import scientific_tool_contracts
from nof1_causal_lab.actions.tool_definition import ToolDefinition

if TYPE_CHECKING:
    from nof1_causal_lab.tool_server import ToolResult

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


class SearchLiteratureInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Search query for empirical literature about effect sizes.")


async def execute_search_literature(_ctx: object, args: SearchLiteratureInput) -> ToolResult:
    """Search Exa for empirical literature about effect sizes."""
    from nof1_causal_lab.workers.prior_research import search_parameter_literature
    from nof1_causal_lab.workers.prompts.prior_research import format_literature_for_parameter

    if not args.query:
        return {"result": "Error: query is required"}
    sources = await search_parameter_literature(args.query)
    if not sources:
        return {"result": "No relevant literature found for this query."}
    return {"result": format_literature_for_parameter(sources)}


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
