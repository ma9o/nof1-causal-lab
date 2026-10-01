"""Typed inputs to the four scientific primitives."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.data_preparation import (
    FilePreparationSpec,
    SimulationReplicateRef,
)
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
from nof1_causal_lab.artifacts.simulation import SimulationSpec


class EditModelRequest(BaseModel):
    """Replace one named base revision with a validated scientific definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["edit_model"] = "edit_model"
    expected_revision: GitOid | None
    model: ModelSpec


class PrepareDataRequest(BaseModel):
    """Prepare uploaded sources or a simulation replicate without a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["prepare_data"] = "prepare_data"
    input: FilePreparationSpec | SimulationReplicateRef


class FitRequest(BaseModel):
    """Condition explicitly selected model and observation revisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["fit"] = "fit"
    model_revision: GitOid = Field()
    panel_revision: GitOid = Field()
    settings: FitSettingsSpec = Field(default_factory=FitSettingsSpec)


class SimulateRequest(SimulationSpec):
    """Generate through end with optional start and interventions; compare saved data with data_diff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["simulate"] = "simulate"
    model_revision: GitOid = Field()


type ScientificActionRequest = Annotated[
    EditModelRequest | PrepareDataRequest | FitRequest | SimulateRequest,
    Field(discriminator="action"),
]


def scientific_tool_contracts():
    """Expose the same typed requests through the tool and study transports."""
    from nof1_causal_lab.actions.results import ActionPoll, ActionReceipt, PollActionRequest
    from nof1_causal_lab.actions.tool_definition import ToolDefinition

    return [
        ToolDefinition(
            name=request.model_fields["action"].default,
            description=description
            + " Dispatch returns only attempt_id. Use poll_action for messages and the final result.",
            input_schema=request,
            output_schema=ActionReceipt,
        )
        for request, description in (
            (
                EditModelRequest,
                "Revise scientific definitions and current laws; run or reuse applicable specification, identification, compatibility and exact predictive checks.",
            ),
            (
                PrepareDataRequest,
                "Prepare uploaded files and a recipe within optional source coverage bounds, or materialize one recorded simulation replicate; return model-independent observations, metadata and numerical data checks.",
            ),
            (
                FitRequest,
                "Condition the selected model on selected observations; return joint uncertainty and fitting diagnostics.",
            ),
            (
                SimulateRequest,
                "Generate requested quantities from current model uncertainty and measure the shared predictive batch.",
            ),
        )
    ] + [
        ToolDefinition(
            name="poll_action",
            description="Read messages and the final body of a dispatched scientific action.",
            input_schema=PollActionRequest,
            output_schema=ActionPoll,
        )
    ]
