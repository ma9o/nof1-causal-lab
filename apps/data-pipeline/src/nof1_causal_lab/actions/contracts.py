"""Typed inputs to the five scientific primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_preparation import (
    FilePreparationSpec,
    SimulationReplicateRef,
)
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.artifacts.simulation import SimulationSpec

if TYPE_CHECKING:
    from nof1_causal_lab.actions.tool_definition import ToolDefinition


class SetQuestionRequest(Value):
    """Set the study question; it is the first action of every study."""

    action: Literal["set_question"] = "set_question"
    question: QuestionSpec


class EditModelRequest(Value):
    """Replace one named base revision with a validated scientific definition."""

    action: Literal["edit_model"] = "edit_model"
    expected_revision: GitOid | None
    model: ModelSpec = Field(
        description="Endogenous constructs are modeled, with or without parents, and include every latent construct. Exogenous constructs are given by direct exact Delta readings and have no dynamics, diffusion, initial coefficients or trajectory law."
    )


class PrepareDataRequest(Value):
    """Prepare uploaded sources or a simulation replicate without a model."""

    action: Literal["prepare_data"] = "prepare_data"
    input: FilePreparationSpec | SimulationReplicateRef


class FitRequest(Value):
    """Condition explicitly selected model and observation revisions."""

    action: Literal["fit"] = "fit"
    model_revision: GitOid = Field()
    panel_revision: GitOid = Field()
    settings: FitSettingsSpec = Field(default_factory=FitSettingsSpec)


class SimulateRequest(SimulationSpec):
    """Generate a dated window with optional interventions; compare saved data with data_diff."""

    action: Literal["simulate"] = "simulate"
    model_revision: GitOid = Field()


type ScientificActionRequest = Annotated[
    SetQuestionRequest | EditModelRequest | PrepareDataRequest | FitRequest | SimulateRequest,
    Field(discriminator="action"),
]


def scientific_tool_contracts() -> list[ToolDefinition]:
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
                SetQuestionRequest,
                "Set the study question as the study's first action: the user's words, the outcome, and named queries, each a dated window whose interventions are contrasted with the recorded course. Name constructs by the identities the model will define.",
            ),
            (
                EditModelRequest,
                "Revise scientific definitions and current laws; run or reuse applicable specification, identification, question, compatibility and exact predictive checks.",
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
                "Generate a window starting on a calendar day, with interventions placed after the start, from current model uncertainty; measure the shared predictive batch.",
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
