"""Recorded checks of a model and its selected observation design."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .checks import PredictiveCheckFinding, SpecificationReport  # noqa: TC001
from .identity import GitOid  # noqa: TC001
from .posterior_diagnostics import PosteriorPredictiveChecks  # noqa: TC001
from .predictive_provenance import PredictiveLawProvenance  # noqa: TC001
from .simulation import SimulationSpec  # noqa: TC001

type CheckGroup = Literal["specification", "identification", "compatibility"]
type PredictiveCheckReason = Literal[
    "MODEL_INCOMPLETE",
    "MODEL_NOT_EXECUTABLE",
    "NO_COMPATIBLE_PANEL",
    "INSUFFICIENT_OBSERVATION_TIMES",
    "SIMULATION_UNSUPPORTED",
]


class ModelPredictiveReport(BaseModel):
    """One automatic, reproducible battery over the full model's current laws."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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
    findings: tuple[PredictiveCheckFinding, ...] = ()
    predictive_checks: PosteriorPredictiveChecks | None = None


class ModelCheckReport(BaseModel):
    """Checks selected by their consumed inputs, retained with the study snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_keys: dict[CheckGroup, str]
    specification: SpecificationReport
    predictive: ModelPredictiveReport | None = None
    reused: tuple[CheckGroup | Literal["predictive"], ...] = Field(default=())
