"""Explicit availability of an owned result and the reason for its absence."""

from typing import Annotated, Literal

from pydantic import Field

from .base import Value


class Available[PayloadT](Value):
    """An available result owns its payload, including a legitimately empty one."""

    kind: Literal["available"] = "available"
    value: PayloadT


class Unavailable(Value):
    """An applicable result could not be produced, for an explicit reason."""

    kind: Literal["unavailable"] = "unavailable"
    reason: str


class NotApplicable(Value):
    """The selected operation does not call for this result."""

    kind: Literal["not_applicable"] = "not_applicable"
    reason: str


type Evaluation[PayloadT] = Annotated[
    Available[PayloadT] | Unavailable | NotApplicable, Field(discriminator="kind")
]
