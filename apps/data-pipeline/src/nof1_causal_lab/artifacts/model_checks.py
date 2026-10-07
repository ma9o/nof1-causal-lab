"""Recorded checks of a model and its selected observation design."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.data_ref import DataRef

from .checks import (
    Assessment,
    SpecificationAssessment,
)
from .identity import ConstructRef, GitOid


class OutcomeSubject(Value):
    """Whether the model defines the question's outcome as a measured, modeled course."""

    check: Literal["outcome"] = "outcome"
    outcome: ConstructRef


class QueryTargetSubject(Value):
    """One query's intervention target: defined, identified, or set inside the record."""

    check: Literal["target", "identification", "range"]
    query: str
    target: ConstructRef


class QueryWindowSubject(Value):
    """Whether the record supports one query's window."""

    check: Literal["window"] = "window"
    query: str


type QuestionSubject = Annotated[
    OutcomeSubject | QueryTargetSubject | QueryWindowSubject, Field(discriminator="check")
]
type QuestionAssessment = Assessment[QuestionSubject, str]


class QuestionCheckReport(Value):
    """The study question checked against the model and, once prepared, the record."""

    question_revision: GitOid
    data: DataRef[GitOid, int] | None
    findings: tuple[QuestionAssessment, ...]


class ModelCheckReport(Value):
    """Findings evaluated by the action against its pinned scientific inputs."""

    specification: tuple[SpecificationAssessment, ...]
    question: QuestionCheckReport | None = None
