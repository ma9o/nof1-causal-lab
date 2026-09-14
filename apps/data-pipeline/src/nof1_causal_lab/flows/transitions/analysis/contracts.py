"""analysis contracts and tool metadata.

analysis exposes a single composable simulation tool. A *scenario* is one
operation on the fitted latent SSM:

    start state  →  apply timed latent clamp(s)  →  roll forward  →  contrast vs reference

The start is either the deterministic baseline equilibrium or an abducted individual
state (conditioning on observed evidence up to a boundary). A clamp is a do-operator
on one latent variable over a time window; a clamp whose window opens at the start is a
forward "intervention", and the same machinery expresses counterfactual "what-if" edits.
The Pearl rung is therefore emergent from the start (baseline → rung 2, abducted →
rung 3), not a separate query type.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel

from nof1_causal_lab.artifacts.scenarios import ScenarioRequest, SimulationResult
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


class ToolError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    identifiable_treatments: list[str] | None = None


class SimulateScenarioToolResult(RootModel[SimulationResult | ToolError]):
    pass


ANALYSIS_TOOL_CONTRACTS: list[ToolDefinition] = [
    ToolDefinition(
        name="get_model_info",
        description=(
            "Return a read-only summary of the fitted model, variables with persistent IDs, identifiability status, "
            "and diagnostics."
        ),
        input_schema=GetModelInfoInput,
    ),
    ToolDefinition(
        name="simulate",
        description=(
            "Run a live nonlinear drift simulation across posterior draws, without future process noise. Start from the "
            "deterministic baseline equilibrium (interventional) or an abducted fitted latent state "
            "(counterfactual), apply one or more timed latent clamps (do-operators), and read the "
            "effect on an outcome over a horizon. Targets and outcome use persistent construct IDs. "
            "The result is ephemeral and includes its request, fitted model revision, and trajectory coordinates."
        ),
        input_schema=ScenarioRequest,
        output_schema=SimulateScenarioToolResult,
    ),
]
