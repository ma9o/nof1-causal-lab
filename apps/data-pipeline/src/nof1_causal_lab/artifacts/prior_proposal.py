"""Authored prior proposals with literature evidence and persisted density curves."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .distribution import DistributionSpec
from .evidence import LiteratureSource
from .identity import ParameterId  # noqa: TC001


class DensityPoint(BaseModel):
    """A density point stores one coordinate of a prior density curve for plotting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float
    y: float = Field(ge=0)


class PriorSource(LiteratureSource):
    """A prior source records literature evidence used to justify a parameter's prior
    distribution.
    """

    effect_size: str | None = Field(
        default=None, description="Reported effect size if available (e.g., 'r=0.3', 'β=0.2')"
    )
    study_interval_days: float | None = Field(
        default=None,
        description="Observation/measurement interval of this study in days (daily=1, weekly=7, monthly=30)",
    )


class PriorProposal(DistributionSpec):
    """A prior proposal specifies a parameter's prior distribution with its rationale and
    supporting evidence.
    """

    parameter_id: ParameterId = Field(description="Scientific parameter this prior describes")
    sources: list[PriorSource] = Field(
        default_factory=list, description="Literature sources supporting this prior"
    )
    reasoning: str = Field(
        description="Justification for the chosen prior distribution and parameters"
    )
    reference_interval_days: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Observation interval (in days) that the DT prior is expressed in. "
            "Sourced from the study's measurement schedule (e.g., 7 for a weekly study). "
            "Used for DT→CT conversion of dynamic priors "
            "(e.g. beta/dt for cross-lags, -log(rho)/dt for baseline persistence)."
        ),
    )
    density_points: list[DensityPoint] | None = Field(
        default=None,
        description=(
            "Pre-computed density curve points [{x, y}, ...] for frontend visualization. "
            "Computed by the pipeline before persistence so the frontend doesn't need "
            "to approximate the PDF client-side."
        ),
    )
