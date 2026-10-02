"""Dispatch receipts and disjoint running/completed poll contracts."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import ActionId, GitOid
from nof1_causal_lab.study.records import ActionAttempt, ActionMessage


class ActionReceipt(Value):
    attempt_id: UUID


class RunningPoll(Value):
    kind: Literal["running"] = "running"
    messages: tuple[ActionMessage, ...] = ()


class CompletedPoll(Value):
    kind: Literal["completed"] = "completed"
    # Failure before publication is still a completed typed outcome.
    commit_id: GitOid | None
    attempt: ActionAttempt
    messages: tuple[ActionMessage, ...] = ()


type ActionPoll = Annotated[RunningPoll | CompletedPoll, Field(discriminator="kind")]


class RunningAction(Value):
    attempt_id: UUID
    action: ActionId
    branch: str
    messages: tuple[ActionMessage, ...]


class PollActionRequest(Value):
    attempt_id: UUID
