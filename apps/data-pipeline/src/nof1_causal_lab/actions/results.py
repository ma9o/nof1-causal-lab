"""The shared polling envelope and action-specific scientific result bodies."""

from typing import Annotated, Literal

from pydantic import Field

from nof1_causal_lab.actions.contracts import (
    DataDiffRequest,
    ModelDiffRequest,
    ScientificActionRequest,
)
from nof1_causal_lab.actions.io import (
    DataDiffOutput,
    EditModelOutput,
    EditQuestionOutput,
    FitOutput,
    ModelDiffOutput,
    PrepareDataOutput,
    SimulateOutput,
)
from nof1_causal_lab.actions.logs import ExecutionMessage
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ActionId, CallId, GitOid


class RunningPoll(Value):
    """An accepted call that can be polled by an external action client."""

    call_id: CallId  # noqa: FIELD003 -- External action clients use this ID for GET polling.
    action: ActionId  # noqa: FIELD003 -- External action clients use this name for GET polling.
    status: Literal["running"] = "running"
    commit_id: None = None  # noqa: FIELD003 -- The shared action wire envelope requires null before publication.
    body: None = None  # noqa: FIELD003 -- The shared action wire envelope requires null while running.
    messages: tuple[ExecutionMessage, ...] = ()


class FailedPoll(Value):
    """A terminal failure with its full details in the accumulated messages."""

    call_id: CallId  # noqa: FIELD003 -- External action clients correlate cached failures by call ID.
    action: ActionId  # noqa: FIELD003 -- The shared action wire envelope identifies the failed action.
    status: Literal["failed"] = "failed"
    commit_id: GitOid | None  # noqa: FIELD003 -- External action clients retain the recorded failure's Git reference.
    body: None = None  # noqa: FIELD003 -- The shared action wire envelope requires null on failure.
    messages: tuple[ExecutionMessage, ...]


class SuccessfulPoll[ActionT: str, BodyT](Value):
    """A published scientific result and the complete execution log of its call."""

    call_id: CallId  # noqa: FIELD003 -- External action clients correlate cached results by call ID.
    action: ActionT
    status: Literal["success"] = "success"  # noqa: FIELD002 -- Discriminates ActionPoll through the nested generic ActionSuccess union.
    commit_id: GitOid  # noqa: FIELD003 -- External action clients use this revision as a subsequent action input.
    body: BodyT
    messages: tuple[ExecutionMessage, ...]


type ActionSuccess = Annotated[
    SuccessfulPoll[Literal["edit_question"], EditQuestionOutput]
    | SuccessfulPoll[Literal["edit_model"], EditModelOutput]
    | SuccessfulPoll[Literal["prepare_data"], PrepareDataOutput]
    | SuccessfulPoll[Literal["fit"], FitOutput]
    | SuccessfulPoll[Literal["simulate"], SimulateOutput]
    | SuccessfulPoll[Literal["data_diff"], DataDiffOutput]
    | SuccessfulPoll[Literal["model_diff"], ModelDiffOutput],
    Field(discriminator="action"),
]

type ActionPoll = Annotated[RunningPoll | FailedPoll | ActionSuccess, Field(discriminator="status")]


class RunningAction(Value):
    """Timeline discovery of the active call and its accumulated messages."""

    call_id: CallId  # noqa: FIELD003 -- External clients discover the active call here before GET polling.
    action: ActionId
    request: ScientificActionRequest | DataDiffRequest[GitOid] | ModelDiffRequest[GitOid]
    messages: tuple[ExecutionMessage, ...]
