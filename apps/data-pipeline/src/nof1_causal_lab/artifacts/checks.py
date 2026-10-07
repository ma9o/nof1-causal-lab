"""Producer-specialized assessments, independent of authoring progression."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value

from .identity import ConstructId, EntityRef, IndicatorRef, ParameterRef

type PredictiveCheckReason = Literal[
    "MODEL_INCOMPLETE",
    "MODEL_NOT_EXECUTABLE",
    "NO_COMPATIBLE_PANEL",
    "INSUFFICIENT_OBSERVATION_TIMES",
    "SIMULATION_UNSUPPORTED",
    "ARCHIVED_MEASUREMENT_NOT_RETAINED",
]
type NotEvaluatedReason = (
    PredictiveCheckReason
    | Literal[
        "NONFINITE_EMISSION_MEAN",
        "INSUFFICIENT_TIMES",
        "NO_RELAXATION_TERM",
        "EDGE_CONTRASTS_EXPLICIT",
        "NO_OBSERVATION_SUPPORT",
        "NO_OBSERVATIONS",
        "STATIC_CONSTRUCT",
        "INSUFFICIENT_OBSERVATIONS",
        "ZERO_RESIDUAL_VARIANCE",
        "ZERO_OBSERVED_VARIANCE",
        "NONFINITE_PATHS",
        "NONFINITE_SIGNAL",
        "COMPARISON_INPUTS_MISSING",
        "INSUFFICIENT_CHAIN_SAMPLES",
        "NO_RETAINED_CHAINS",
        "NO_OUTCOME",
        "CONSTRUCT_UNDEFINED",
        "NO_PANEL",
        "STATE_NOT_RECORDED",
    ]
)


class Evaluated[Subject, Evidence](Value):
    """One producer's measured outcome and the evidence supporting it."""

    kind: Literal["evaluated"] = "evaluated"
    subject: Subject
    outcome: Literal["passed", "failed", "warning", "error"]
    evidence: Evidence


class NotEvaluated[Subject](Value):
    """An explicit absence of evaluation, with a closed producer reason."""

    kind: Literal["not_evaluated"] = "not_evaluated"
    subject: Subject
    reason: NotEvaluatedReason
    detail: str = ""


type Assessment[Subject, Evidence] = Annotated[
    Evaluated[Subject, Evidence] | NotEvaluated[Subject], Field(discriminator="kind")
]


class ConvergenceCriterion(StrEnum):
    """Convergence statistics assessed for each retained parameter coordinate."""

    R_HAT = "r_hat"
    ESS_BULK = "ess_bulk"
    ESS_TAIL = "ess_tail"


class NumericCriterionEvidence(Value):
    """A measured scalar and the producer's numerical acceptance region."""

    criterion: str
    value: float
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    note: str = ""
    display_value: str = ""
    band_label: str = ""


class PredictiveSubject(Value):
    """One named check and its stable target in a construct's scientific context."""

    check: str
    construct_id: ConstructId | None = None
    target: EntityRef | Literal["whole_model", "observations"]


type IndicatorCheck = Literal["calibration", "autocorrelation", "variance"]


class IndicatorCheckSubject(Value):
    """The indicator and criterion remain present when evaluation is unavailable."""

    target: IndicatorRef
    check: IndicatorCheck


class ConvergenceSubject(Value):
    """A convergence criterion on one stable scientific scalar."""

    parameter: ParameterRef
    criterion: ConvergenceCriterion
    label: str


type SpecificationAssessment = Assessment[str, str]
type PredictiveAssessment = Assessment[PredictiveSubject, tuple[NumericCriterionEvidence, ...]]
