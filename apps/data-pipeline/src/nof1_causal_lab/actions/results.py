"""Dispatch receipts, scientific result bodies, and timestamped action labels."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from nof1_causal_lab.artifacts.checks import SpecificationReport
from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import GitOid, GitRef, ScientificActionId
from nof1_causal_lab.artifacts.model_checks import ModelPredictiveReport
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
    ValidationReportArtifact,
)
from nof1_causal_lab.machine.view_models import MeasurementsData


class ActionMessage(BaseModel):
    """One label emitted by an action; scientific measurements belong in its body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: AwareDatetime
    level: Literal["debug", "info", "warn", "error"]
    label: str = Field(pattern=r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*$")


class ActionReceipt(BaseModel):
    """A durable dispatch acknowledgment, without a scientific result body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: UUID


class ModelEditResult(BaseModel):
    """The committed scientific model and checks produced by an edit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["edit_model"] = "edit_model"
    commit_id: GitOid
    model_revision: GitOid
    model: ModelSpec
    specification: SpecificationReport
    predictive: ModelPredictiveReport
    identification: IdentificationReport
    validation: ValidationReportArtifact | None = None


class DataPreparationResult(BaseModel):
    """Prepared observations and their committed revision, with data-quality findings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["prepare_data"] = "prepare_data"
    commit_id: GitOid
    data_revision: GitRef
    data: MeasurementsData
    metadata: PreparedDataMetadata
    profile: DataProfileArtifact


class ModelFitResult(BaseModel):
    """The committed model, inference report, and checks produced by a fit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["fit"] = "fit"
    commit_id: GitOid
    model_revision: GitOid
    model: ModelSpec
    report: InferenceReport
    specification: SpecificationReport
    identification: IdentificationReport


class ModelSimulationResult(BaseModel):
    """A committed trajectory or causal simulation report for its selected model and design."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: Literal["simulate"] = "simulate"
    commit_id: GitOid
    report: SimulationReport


type ActionBody = Annotated[
    ModelEditResult | DataPreparationResult | ModelFitResult | ModelSimulationResult,
    Field(
        discriminator="action",
        description=(
            "The scientific result published by a successful action, including immutable "
            "revision references and the resulting model, data, or simulation findings."
        ),
    ),
]


class ActionPoll(BaseModel):
    """Read an attempt: messages accumulate; a successful commit supplies the body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    done: bool
    body: ActionBody | None = None
    messages: tuple[ActionMessage, ...] = ()


class RunningAction(BaseModel):
    """The attempt an episode is executing, with the labels it has emitted so far."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: UUID
    action: ScientificActionId
    branch: str
    messages: tuple[ActionMessage, ...]


class PollActionRequest(BaseModel):
    """Read one previously dispatched action through the scientific tool interface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    attempt_id: UUID
