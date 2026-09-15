"""Typed inputs to the four scientific primitives."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nof1_causal_lab.artifacts.model_spec import ModelSpec  # noqa: TC001
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
from nof1_causal_lab.artifacts.simulation import SimulationDesign  # noqa: TC001


class EditModelRequest(BaseModel):
    """Replace one named base revision with a validated scientific definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["edit_model"] = "edit_model"
    expected_version: int = Field(ge=0)
    model: ModelSpec


class PrepareDataRequest(BaseModel):
    """Import uploaded files or extract measurements from a pinned source table."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["prepare_data"] = "prepare_data"
    source: Literal["files", "raw_data"]
    raw_data_version: int | None = Field(default=None, ge=1)
    model_version: int | None = Field(default=None, ge=1)
    max_windows: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def source_inputs(self) -> Self:
        if self.source == "raw_data":
            if self.raw_data_version is None or self.model_version is None:
                raise ValueError("Extraction requires raw_data_version and model_version")
        elif any(
            value is not None
            for value in (self.raw_data_version, self.model_version, self.max_windows)
        ):
            raise ValueError("File import does not consume a model, source table, or window limit")
        return self


class FitRequest(BaseModel):
    """Condition explicitly selected model and observation revisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["fit"] = "fit"
    model_version: int = Field(ge=1)
    panel_version: int = Field(ge=1)
    settings: FitSettingsSpec = Field(default_factory=FitSettingsSpec)


class SimulateRequest(BaseModel):
    """Simulate current model uncertainty, optionally comparing with observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["simulate"] = "simulate"
    model_version: int = Field(ge=1)
    design: SimulationDesign
    comparison_panel_version: int | None = Field(default=None, ge=1)


type ScientificActionRequest = Annotated[
    EditModelRequest | PrepareDataRequest | FitRequest | SimulateRequest,
    Field(discriminator="action"),
]


def scientific_tool_contracts():
    """Expose the same typed requests through the tool and episode transports."""
    from nof1_causal_lab.flows.contracts_base import ToolDefinition
    from nof1_causal_lab.machine.hierarchy import ACTIONS_BY_ID
    from nof1_causal_lab.machine.status import MoveOutcome

    return [
        ToolDefinition(
            name=request.model_fields["action"].default,
            description=ACTIONS_BY_ID[request.model_fields["action"].default].description,
            input_schema=request,
            output_schema=MoveOutcome,
        )
        for request in (EditModelRequest, PrepareDataRequest, FitRequest, SimulateRequest)
    ]
