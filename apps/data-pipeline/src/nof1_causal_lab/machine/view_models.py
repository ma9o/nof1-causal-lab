"""Server-composed artifact views at a selected model revision."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, RootModel

from nof1_causal_lab.artifacts.baseline_report import BaselineReportArtifact
from nof1_causal_lab.artifacts.causal_design import CausalDesign  # noqa: TC001
from nof1_causal_lab.artifacts.effects import HistogramBin  # noqa: TC001
from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId  # noqa: TC001
from nof1_causal_lab.artifacts.latent_structure import LatentStructureArtifact
from nof1_causal_lab.artifacts.measurements import ObservationRecord, WorkerStatus  # noqa: TC001
from nof1_causal_lab.artifacts.posterior import PosteriorArtifact
from nof1_causal_lab.artifacts.statistical_model_spec import (
    ParameterSpec,
    StatisticalModelSpecArtifact,
)
from nof1_causal_lab.artifacts.structural_plan import StructuralPlan  # noqa: TC001
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
    """Profile and representative rows from one uploaded table version."""

    n_records: int
    n_columns: int
    date_range: RawDataDateRange
    sample: list[dict[str, str | None]]
    column_descriptions: list[RawDataColumnDescription]


class MeasurementsData(ViewValue):
    """Worker outcomes and panel counts derived from the same extraction revision."""

    workers: list[WorkerStatus]
    per_indicator_counts: dict[IndicatorId, int]
    combined_extractions_sample: list[ObservationRecord]


class ModelSpecLikelihoodDiagnostics(ViewValue):
    """Observed values and validation profile for one likelihood's pinned panel."""

    indicator_id: IndicatorId
    profile: IndicatorEmpiricalProfile | None
    histogram: list[HistogramBin]
    prior_counts: list[float] | None = None
    prior_outside_fraction: float | None = None


class StateEquation(ViewValue):
    """A continuous-time state equation rendered from declared scientific mechanisms."""

    construct_id: ConstructId
    label: str
    latex: str


class StatisticalModelSpecData(StatisticalModelSpecArtifact):
    """A specification with observed likelihood diagnostics from its pinned inputs."""

    structural_plan: StructuralPlan | None = None
    state_equations: list[StateEquation] = Field(default_factory=list)
    parameters: list[ParameterSpec] = Field(default_factory=list)
    likelihood_diagnostics: dict[IndicatorId, ModelSpecLikelihoodDiagnostics] = Field(
        default_factory=dict
    )


class MeasurementStructureViewData(ViewValue):
    """Measurement definitions with their corresponding causal design and structural plan."""

    causal_design: CausalDesign
    structural_plan: StructuralPlan


class ArtifactViews(ViewValue):
    """Available artifact projections read from one committed model state."""

    raw_data: RawDataData | None = None
    latent_structure: LatentStructureArtifact | None = None
    measurement_structure: MeasurementStructureViewData | None = None
    measurements: MeasurementsData | None = None
    validation_report: ValidationReportArtifact | None = None
    statistical_model_spec: StatisticalModelSpecData | None = None
    posterior: PosteriorArtifact | None = None
    baseline_report: BaselineReportArtifact | None = None


class ArtifactViewResponse(
    RootModel[
        RawDataData
        | LatentStructureArtifact
        | MeasurementStructureViewData
        | MeasurementsData
        | ValidationReportArtifact
        | StatisticalModelSpecData
        | PosteriorArtifact
        | BaselineReportArtifact
    ]
):
    """One available artifact projection returned by the model view endpoint."""
