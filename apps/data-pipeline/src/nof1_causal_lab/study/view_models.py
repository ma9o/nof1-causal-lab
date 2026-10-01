"""Server-composed artifact views at a selected model revision."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, RootModel

from nof1_causal_lab.artifacts.effects import HistogramBin
from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId, ParameterId
from nof1_causal_lab.artifacts.measurements import ObservationRecord
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReport
from nof1_causal_lab.artifacts.validation_report import (
    IndicatorEmpiricalProfile,
    ValidationReportArtifact,
)


class ViewValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RawDataDateRange(ViewValue):
    """Observed date bounds of the uploaded table, when it contains a date column."""

    start: str
    end: str


class RawDataColumnDescription(ViewValue):
    """A stored column's physical type and authored interpretation."""

    name: str
    dtype: str
    description: str


class RawDataData(ViewValue):
    """Profile and representative rows from one uploaded table revision."""

    n_records: int
    n_columns: int
    date_range: RawDataDateRange
    sample: list[dict[str, str | None]]
    column_descriptions: list[RawDataColumnDescription]


class MeasurementsData(ViewValue):
    """Counts and representative observations read directly from one panel revision."""

    n_observations: int
    per_indicator_counts: dict[IndicatorId, int]
    combined_extractions_sample: list[ObservationRecord]


class LikelihoodDiagnostics(ViewValue):
    """Observed values and validation profile for one likelihood's pinned panel."""

    indicator_id: IndicatorId
    profile: IndicatorEmpiricalProfile | None
    histogram: list[HistogramBin]


class StateEquation(ViewValue):
    """A continuous-time state equation rendered from declared scientific mechanisms."""

    construct_id: ConstructId
    label: str
    latex: str


class DensityPoint(ViewValue):
    """A plotting coordinate evaluated from the native prior's log density."""

    x: float
    y: float = Field(ge=0)


class ModelDiagnostics(ViewValue):
    """Server-derived equations and comparisons with pinned observations."""

    confounder_equations: list[StateEquation] = Field(default_factory=list)
    state_equations: list[StateEquation] = Field(default_factory=list)
    observation_equations: dict[IndicatorId, str] = Field(default_factory=dict)
    likelihood_diagnostics: dict[IndicatorId, LikelihoodDiagnostics] = Field(default_factory=dict)
    prior_densities: dict[ParameterId, tuple[DensityPoint, ...]] = Field(default_factory=dict)


class ArtifactViewResponse(
    RootModel[
        RawDataData
        | ModelSpec
        | MeasurementsData
        | ValidationReportArtifact
        | ModelDiagnostics
        | InferenceReport
    ]
):
    """One available artifact projection returned by the model view endpoint."""
