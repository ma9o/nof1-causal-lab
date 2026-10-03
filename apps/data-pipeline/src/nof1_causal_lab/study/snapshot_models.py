"""Aggregate reads for one committed model revision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.checks import SpecificationAssessment
from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
from nof1_causal_lab.artifacts.effects import HistogramBin
from nof1_causal_lab.artifacts.execution import StructuralItemDisposition
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import (
    ArtifactId,
    ConstructId,
    EdgeId,
    GitOid,
    GitRef,
    IndicatorId,
    ParameterId,
    ParameterRef,
)
from nof1_causal_lab.artifacts.model_checks import ModelPredictiveReport, QuestionCheckReport
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReportCore
from nof1_causal_lab.artifacts.posterior_diagnostics import (
    DensityCurve,
)
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
    ValidationReportArtifact,
)
from nof1_causal_lab.study.state import SourceValidity, StudyState, is_stale
from nof1_causal_lab.study.view_models import (
    MeasurementsData,
    RawDataData,
)


class FactSource(Value):
    """A fact source locates supporting content within an artifact revision and records its freshness."""

    ref: GitRef
    pointer: str = Field(pattern=r"^(?:/.*)?$")
    validity: SourceValidity


class Sourced[T](Value):
    """A sourced read pairs a canonical aggregate or derived finding with its artifact revision."""

    value: T
    source: FactSource


class FitSummary(Value):
    """A fit read contains the inference report summary and server-composed display findings.

    Per-draw diagnostics load separately from the inference report endpoint.
    """

    report: InferenceReportCore
    edge_estimates: Mapping[EdgeId, ParameterRef] = Field(default_factory=dict)
    decay_estimates: Mapping[ConstructId, ParameterRef] = Field(default_factory=dict)
    prior_densities: Mapping[ParameterId, DensityCurve] = Field(
        default_factory=dict,
        description=(
            "Conditioned input laws of the fitted parameters, on their posterior marginals' "
            "quantity scale; absent where the current compiler cannot place the input model."
        ),
    )


class ModelGraphView(Value):
    """Scientific entity identities selected for the graph at this authoring checkpoint."""

    construct_ids: tuple[ConstructId, ...] = ()
    edge_ids: tuple[EdgeId, ...] = ()
    dynamic_construct_ids: tuple[ConstructId, ...] = ()
    status: Mapping[ConstructId, Literal["observed", "marginalized", "blocking"]] = Field(
        default_factory=dict
    )


class ModelSnapshot(Value):
    """The canonical scientific definition with independently sourced inputs and findings."""

    question: Sourced[QuestionSpec] | None = None
    model: Sourced[ModelSpec] | None = None
    workspace_id: str = Field(min_length=1)
    branch: str = "main"
    commit_id: GitOid
    selected_seq: int = Field(ge=0)
    can_simulate: bool = False
    state: StudyState
    raw_data: Sourced[RawDataData] | None = None
    measurements: Sourced[MeasurementsData] | None = None
    metadata: Sourced[PreparedDataMetadata] | None = None
    profile: Sourced[DataProfileArtifact] | None = None

    identification: Sourced[IdentificationReport] | None = None
    dispositions: Sourced[tuple[StructuralItemDisposition, ...]] | None = None
    graph: ModelGraphView = Field(default_factory=ModelGraphView)
    entity_failures: Mapping[ConstructId | EdgeId | IndicatorId, tuple[str, ...]] = Field(
        default_factory=dict
    )
    validation_report: Sourced[ValidationReportArtifact] | None = None
    confounder_equations: Mapping[ConstructId, str] = Field(default_factory=dict)
    state_equations: Mapping[ConstructId, str] = Field(default_factory=dict)
    observation_equations: Mapping[IndicatorId, str] = Field(default_factory=dict)
    likelihood_diagnostics: Mapping[IndicatorId, tuple[HistogramBin, ...]] = Field(
        default_factory=dict
    )
    authoring_prior_densities: Mapping[ParameterId, DensityCurve] = Field(default_factory=dict)
    fit: Sourced[FitSummary] | None = None
    specification: Sourced[tuple[SpecificationAssessment, ...]] | None = None
    question_checks: Sourced[QuestionCheckReport] | None = None
    simulation: Sourced[SimulationReport] | None = None
    predictive: Sourced[ModelPredictiveReport] | None = None

    @model_validator(mode="after")
    def validate_ownership_and_sources(self) -> ModelSnapshot:
        sources: tuple[
            tuple[
                FactSource | None,
                ArtifactId | Literal["inference", "simulation", "specification", "predictive"],
            ],
            ...,
        ] = (
            (self.model.source if self.model is not None else None, "model"),
            (
                self.identification.source if self.identification is not None else None,
                "identification_report",
            ),
            (
                self.dispositions.source if self.dispositions is not None else None,
                "model",
            ),
            (self.raw_data.source if self.raw_data is not None else None, "raw_data"),
            (
                self.measurements.source if self.measurements is not None else None,
                "panel",
            ),
            (self.metadata.source if self.metadata is not None else None, "panel"),
            (self.profile.source if self.profile is not None else None, "data_profile"),
            (
                self.validation_report.source if self.validation_report is not None else None,
                "validation_report",
            ),
            (self.fit.source if self.fit is not None else None, "inference"),
            (
                self.specification.source if self.specification is not None else None,
                "specification",
            ),
            (
                self.simulation.source if self.simulation is not None else None,
                "simulation",
            ),
            (
                self.predictive.source if self.predictive is not None else None,
                "predictive",
            ),
        )
        for source, artifact_id in sources:
            if source is not None:
                self._validate_source(source, artifact_id)
        model = self.model.value if self.model else None
        constructs = {item.id for item in model.constructs} if model else set()
        edges = {item.id for item in model.edges} if model else set()
        indicators = {item.observation.id for item in model.indicators} if model else set()
        parameters = {item.id for item in model.parameters} if model else set()
        findings = self
        if (
            not set(findings.graph.dynamic_construct_ids) <= set(findings.graph.construct_ids)
            or not set(findings.graph.construct_ids) <= constructs
            or not set(findings.graph.edge_ids) <= edges
        ):
            raise ValueError("Graph view references an entity outside the snapshot")
        if findings.dispositions and any(
            item.target.id not in constructs | edges | indicators
            for item in findings.dispositions.value
        ):
            raise ValueError("Disposition owner does not exist in the snapshot")
        if not findings.graph.status.keys() <= constructs:
            raise ValueError("Graph status owner does not exist in the snapshot")
        if findings.identification:
            if model is None:
                raise ValueError("Identification requires its scientific model")
            findings.identification.value.validate_model(model)
        if self.measurements and self.metadata is None:
            raise ValueError("Prepared observations require their data-owned metadata")
        # Counts describe the stored table, including data-quality problems.
        # Undeclared variables are reported by preparation checks, never hidden here.
        if (
            findings.validation_report
            and not findings.validation_report.value.data.indicators.keys() <= indicators
        ):
            raise ValueError("Validation owner does not exist in the snapshot")
        if findings.fit:
            fit = findings.fit.value
            marginals = fit.report.posterior_marginals or []
            if any(item.subject.parameter_id not in parameters for item in marginals):
                raise ValueError("Posterior finding has no scientific parameter definition")
            if not fit.prior_densities.keys() <= parameters:
                raise ValueError("Prior curve has no scientific parameter definition")
        return self

    def _validate_source(
        self,
        source: FactSource,
        artifact_id: ArtifactId | Literal["inference", "simulation", "specification", "predictive"],
    ) -> None:
        from nof1_causal_lab.study.artifact_files import artifact_file_spec

        ref = source.ref
        if ref.workspace_id != self.workspace_id:
            raise ValueError("Fact source belongs to another study")
        if ref.path == "checks.json":
            if (
                artifact_id not in {"specification", "predictive"}
                or ref.revision != self.commit_id
                or source.pointer != f"/{artifact_id}"
            ):
                raise ValueError("Check source must identify this snapshot's recorded findings")
            panel = self.state.current.get("panel")
            expected = (
                "stale"
                if artifact_id == "predictive"
                and self.predictive is not None
                and self.predictive.value.panel_revision != (panel.revision if panel else None)
                else "fresh"
            )
            if source.validity != expected:
                raise ValueError("Check validity differs from its selected observation revision")
            return
        if ref.path == "logs/attempt.json":
            if artifact_id not in {"inference", "simulation", "specification"}:
                raise ValueError("Only operation findings refer to action logs")
            # The repository reader selects these logs through Git ancestry.
            return
        if artifact_id in {"inference", "simulation", "specification", "predictive"}:
            raise ValueError("Operation findings require an action log or recorded check")
        current = next(
            (record for key, record in self.state.current.items() if key == artifact_id),
            None,
        )
        if current is None or current.revision != ref.revision:
            raise ValueError("Fact source does not belong to the selected artifact tree")
        if (
            ref.path
            not in {
                **artifact_file_spec(current.artifact_id).parquet,
                **artifact_file_spec(current.artifact_id).json,
            }.values()
        ):
            raise ValueError("Fact source does not identify a declared artifact payload")
        state = self.state
        expected = "stale" if is_stale(state, current.artifact_id) else "fresh"
        if source.validity != expected:
            raise ValueError("Fact validity differs from its snapshot input references")
