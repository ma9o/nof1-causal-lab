"""Recorded checks of a model and its selected observation design."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value

from .checks import Evaluated, PredictiveAssessment, SpecificationReport
from .identity import GitOid
from .posterior_diagnostics import PosteriorPredictiveChecks
from .predictive_provenance import PredictiveLawProvenance
from .simulation import SimulationSpec

type CheckGroup = Literal["specification", "identification", "compatibility"]
type PredictiveCheckReason = Literal[
    "MODEL_INCOMPLETE",
    "MODEL_NOT_EXECUTABLE",
    "NO_COMPATIBLE_PANEL",
    "INSUFFICIENT_OBSERVATION_TIMES",
    "SIMULATION_UNSUPPORTED",
    "ARCHIVED_MEASUREMENT_NOT_RETAINED",
]


class ModelPredictiveReport(Value):
    """One automatic, reproducible battery over the full model's current laws."""

    input_key: str
    model_revision: GitOid
    panel_revision: GitOid | None
    status: Literal["passed", "failed", "not_evaluated"]
    reason: PredictiveCheckReason | None = None
    detail: str | None = None
    design: SimulationSpec | None = None
    draws: int = Field(ge=1)
    seed: int = Field(ge=0)
    law: PredictiveLawProvenance
    findings: tuple[PredictiveAssessment, ...] = ()
    predictive_checks: PosteriorPredictiveChecks | None = None

    def not_evaluated(self, reason: PredictiveCheckReason, detail: str | None = None) -> Self:
        return self.model_copy(
            update={"status": "not_evaluated", "reason": reason, "detail": detail}
        )

    def evaluated(
        self,
        design: SimulationSpec,
        findings: tuple[PredictiveAssessment, ...],
        predictive_checks: PosteriorPredictiveChecks | None = None,
    ) -> Self:
        failed = any(isinstance(f, Evaluated) and f.outcome == "failed" for f in findings) or (
            predictive_checks is not None
            and any(
                isinstance(f, Evaluated) and f.outcome != "passed"
                for f in predictive_checks.per_variable_warnings
            )
        )
        return self.model_copy(
            update={
                "status": "failed" if failed else "passed",
                "reason": None,
                "detail": None,
                "design": design,
                "findings": findings,
                "predictive_checks": predictive_checks,
            }
        )


class ModelCheckReport(Value):
    """Checks selected by their consumed inputs, retained with the study snapshot."""

    input_keys: Mapping[CheckGroup, str]
    specification: SpecificationReport
    predictive: ModelPredictiveReport | None = None
    reused: tuple[CheckGroup | Literal["predictive"], ...] = Field(default=())
