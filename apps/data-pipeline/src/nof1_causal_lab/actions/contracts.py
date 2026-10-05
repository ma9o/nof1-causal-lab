"""Content-named inputs to scientific calls."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from nof1_causal_lab.artifacts.action import ACTION_REASONING_DESCRIPTION
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_preparation import (
    FilePreparationSpec,
    SimulationReplicateRef,
)
from nof1_causal_lab.artifacts.identity import GitOid, scientific_id
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import FitSettingsSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.artifacts.simulation import SimulationSpec


class SetQuestionRequest(Value):
    """Set the study question; it is the first action of every study."""

    action: Literal["set_question"] = "set_question"
    question: QuestionSpec
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


class EditModelRequest(Value):
    """Replace one named base revision with a validated scientific definition."""

    action: Literal["edit_model"] = "edit_model"
    expected_revision: GitOid | None
    panel_revision: GitOid | None = Field(
        default=None,
        description="Exact observations used by this edit's checks; omitted runs no predictive check.",
    )
    model: ModelSpec = Field(
        description="Endogenous constructs are modeled, with or without parents, and include every latent construct. Exogenous constructs are given by direct exact Delta readings and have no dynamics, diffusion, initial coefficients or trajectory law."
    )
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


class PrepareDataRequest(Value):
    """Prepare uploaded sources or a simulation replicate without a model."""

    action: Literal["prepare_data"] = "prepare_data"
    input: FilePreparationSpec | SimulationReplicateRef
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


class FitRequest(Value):
    """Condition explicitly selected model and observation revisions."""

    action: Literal["fit"] = "fit"
    model_revision: GitOid = Field()
    panel_revision: GitOid = Field()
    settings: FitSettingsSpec = Field(default_factory=FitSettingsSpec)
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


class SimulateRequest(SimulationSpec):
    """Generate a dated window with optional interventions; compare saved data with data_diff."""

    action: Literal["simulate"] = "simulate"
    model_revision: GitOid = Field()
    panel_revision: GitOid | None = Field(
        default=None,
        description="Exact panel dating authored-law simulations; fitted laws retain their own origin.",
    )
    reasoning: str | None = Field(default=None, description=ACTION_REASONING_DESCRIPTION)


type ScientificActionRequest = Annotated[
    SetQuestionRequest | EditModelRequest | PrepareDataRequest | FitRequest | SimulateRequest,
    Field(discriminator="action"),
]


def call_identity(request: Value) -> str:
    """Scientific arguments, including file hashes but excluding intent, name the call."""
    return scientific_id(
        "call", request.model_dump(mode="json", round_trip=True, exclude={"reasoning"})
    )
