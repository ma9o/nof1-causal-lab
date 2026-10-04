"""Recorded checks of a model and its selected observation design."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, computed_field

from nof1_causal_lab.artifacts.base import Value
from .checks import (
    Assessment,
    Evaluated,
    PredictiveAssessment,
    PredictiveCheckReason,
    SpecificationAssessment,
)
from .identity import ConstructRef, GitOid
from .posterior_diagnostics import PosteriorPredictiveChecks
from .predictive_provenance import PredictiveLawProvenance

type CheckGroup = Literal["specification", "identification", "compatibility", "question"]


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
    panel_revision: GitOid | None
    findings: tuple[QuestionAssessment, ...]


class EvaluatedPredictiveChecks(Value):
    """An evaluated run may retain failed and partially unavailable scientific evidence."""

    kind: Literal["evaluated"] = "evaluated"
    findings: tuple[PredictiveAssessment, ...]
    predictive_checks: PosteriorPredictiveChecks | None = None


class UnavailablePredictiveChecks(Value):
    """The run could not evaluate its scientific battery."""

    kind: Literal["unavailable"] = "unavailable"
    reason: PredictiveCheckReason
    detail: str | None = None


type ModelPredictiveEvaluation = Annotated[
    EvaluatedPredictiveChecks | UnavailablePredictiveChecks, Field(discriminator="kind")
]


class ModelPredictiveReport(Value):
    """One automatic, reproducible battery over the full model's current laws."""

    model_revision: GitOid
    panel_revision: GitOid | None
    draws: int = Field(ge=1)
    seed: int = Field(ge=0)
    law: PredictiveLawProvenance
    evaluation: ModelPredictiveEvaluation

    @computed_field
    @property
    def status(self) -> Literal["passed", "failed", "not_evaluated"]:
        evaluation = self.evaluation
        if isinstance(evaluation, UnavailablePredictiveChecks):
            return "not_evaluated"
        failed = any(
            isinstance(f, Evaluated) and f.outcome == "failed" for f in evaluation.findings
        ) or (
            evaluation.predictive_checks is not None
            and any(
                isinstance(f, Evaluated) and f.outcome != "passed"
                for f in evaluation.predictive_checks.per_variable_warnings
            )
        )
        return "failed" if failed else "passed"


class ModelCheckReport(Value):
    """Current-code findings selected by their consumed scientific inputs."""

    specification: tuple[SpecificationAssessment, ...]
    question: QuestionCheckReport | None = None
    predictive: ModelPredictiveReport | None = None
    reused: tuple[CheckGroup | Literal["predictive"], ...] = Field(default=())
