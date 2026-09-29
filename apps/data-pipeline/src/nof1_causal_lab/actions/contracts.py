"""Typed inputs to the four scientific primitives."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.data_preparation import (  # noqa: TC001
    FilePreparationSpec,
    ObservationTableSpec,
    SimulationReplicateRef,
)
from nof1_causal_lab.artifacts.identity import GitOid  # noqa: TC001
from nof1_causal_lab.artifacts.model_spec import ModelSpec  # noqa: TC001
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
from nof1_causal_lab.artifacts.simulation import SimulationSpec


class EditModelRequest(BaseModel):
    """Replace one named base revision with a validated scientific definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["edit_model"] = "edit_model"
    expected_revision: GitOid | None
    model: ModelSpec


class PrepareDataRequest(BaseModel):
    """Prepare uploaded sources, a simulation replicate, or extracted observations without a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["prepare_data"] = "prepare_data"
    input: FilePreparationSpec | SimulationReplicateRef | ObservationTableSpec


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
    """Expose the same typed requests through the tool and episode transports."""
    from nof1_causal_lab.actions.results import ActionPoll, ActionReceipt, PollActionRequest
    from nof1_causal_lab.flows.contracts_base import ToolDefinition
    from nof1_causal_lab.machine.hierarchy import ACTIONS_BY_ID

    return [
        ToolDefinition(
            name=request.model_fields["action"].default,
            description=ACTIONS_BY_ID[request.model_fields["action"].default].description
            + " Dispatch returns only attempt_id. Use poll_action for messages and the final result.",
            input_schema=request,
            output_schema=ActionReceipt,
        )
        for request in (EditModelRequest, PrepareDataRequest, FitRequest, SimulateRequest)
    ] + [
        ToolDefinition(
            name="poll_action",
            description="Read messages and the final body of a dispatched scientific action.",
            input_schema=PollActionRequest,
            output_schema=ActionPoll,
        )
    ]
