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
    ConstructId,
    EdgeId,
    GitOid,
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
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.view_models import (
    MeasurementsData,
    RawDataData,
)


class FitSummary(Value):
    """A fit read contains the inference report summary and server-composed display findings.

    The completed action also carries the full inference report and per-draw diagnostics.
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
    """Scientific values selected from recorded action dependencies."""

    question: QuestionSpec | None = None
    model: ModelSpec | None = None
    workspace_id: str = Field(min_length=1)
    commit_id: GitOid
    selected_seq: int = Field(ge=0)
    can_simulate: bool = False
    state: StudyState
    raw_data: RawDataData | None = None
    measurements: MeasurementsData | None = None
    metadata: PreparedDataMetadata | None = None
    profile: DataProfileArtifact | None = None

    identification: IdentificationReport | None = None
    dispositions: tuple[StructuralItemDisposition, ...] | None = None
    graph: ModelGraphView = Field(default_factory=ModelGraphView)
    entity_failures: Mapping[ConstructId | EdgeId | IndicatorId, tuple[str, ...]] = Field(
        default_factory=dict
    )
    validation_report: ValidationReportArtifact | None = None
    confounder_equations: Mapping[ConstructId, str] = Field(default_factory=dict)
    state_equations: Mapping[ConstructId, str] = Field(default_factory=dict)
    observation_equations: Mapping[IndicatorId, str] = Field(default_factory=dict)
    likelihood_diagnostics: Mapping[IndicatorId, tuple[HistogramBin, ...]] = Field(
        default_factory=dict
    )
    authoring_prior_densities: Mapping[ParameterId, DensityCurve] = Field(default_factory=dict)
    fit: FitSummary | None = None
    specification: tuple[SpecificationAssessment, ...] | None = None
    question_checks: QuestionCheckReport | None = None
    simulation: SimulationReport | None = None
    predictive: ModelPredictiveReport | None = None

    @model_validator(mode="after")
    def validate_ownership(self) -> ModelSnapshot:
        """Require snapshot findings to reference model-owned entities."""
        model = self.model if self.model else None
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
            item.target.id not in constructs | edges | indicators for item in findings.dispositions
        ):
            raise ValueError("Disposition owner does not exist in the snapshot")
        if not findings.graph.status.keys() <= constructs:
            raise ValueError("Graph status owner does not exist in the snapshot")
        if findings.identification:
            if model is None:
                raise ValueError("Identification requires its scientific model")
            findings.identification.validate_model(model)
        if self.measurements and self.metadata is None:
            raise ValueError("Prepared observations require their data-owned metadata")
        # Counts describe the stored table, including data-quality problems.
        # Undeclared variables are reported by preparation checks, never hidden here.
        if (
            findings.validation_report
            and not findings.validation_report.data.indicators.keys() <= indicators
        ):
            raise ValueError("Validation owner does not exist in the snapshot")
        if findings.fit:
            fit = findings.fit
            marginals = fit.report.posterior_marginals or []
            if any(item.subject.parameter_id not in parameters for item in marginals):
                raise ValueError("Posterior finding has no scientific parameter definition")
            if not fit.prior_densities.keys() <= parameters:
                raise ValueError("Prior curve has no scientific parameter definition")
        return self
