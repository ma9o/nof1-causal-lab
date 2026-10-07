"""Aggregate reads for one committed model revision."""

from __future__ import annotations

from pydantic import Field, model_validator

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.checks import SpecificationAssessment
from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
from nof1_causal_lab.artifacts.identification import IdentificationReport
from nof1_causal_lab.artifacts.identity import (
    GitOid,
)
from nof1_causal_lab.artifacts.model_checks import QuestionCheckReport
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.posterior import InferenceReportCore
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.artifacts.simulation import SimulationReport
from nof1_causal_lab.artifacts.validation_report import (
    DataProfileArtifact,
    ValidationReportArtifact,
)
from nof1_causal_lab.study.state import StudyState


class ModelSnapshot(Value):
    """Scientific values selected from recorded action dependencies."""

    question: QuestionSpec | None = None
    model: ModelSpec | None = None
    workspace_id: str = Field(min_length=1)
    commit_id: GitOid
    selected_seq: int = Field(ge=0)
    state: StudyState
    metadata: PreparedDataMetadata | None = None
    profile: DataProfileArtifact | None = None

    identification: IdentificationReport | None = None
    validation_report: ValidationReportArtifact | None = None
    fit: InferenceReportCore | None = None
    specification: tuple[SpecificationAssessment, ...] | None = None
    question_checks: QuestionCheckReport | None = None
    simulation: SimulationReport | None = None

    @model_validator(mode="after")
    def validate_ownership(self) -> ModelSnapshot:
        """Require snapshot findings to reference model-owned entities."""
        model = self.model if self.model else None
        indicators = {item.observation.id for item in model.indicators} if model else set()
        parameters = {item.id for item in model.parameters} if model else set()
        findings = self
        if findings.identification:
            if model is None:
                raise ValueError("Identification requires its scientific model")
            findings.identification.validate_model(model)
        # Counts describe the stored table, including data-quality problems.
        # Undeclared variables are reported by preparation checks, never hidden here.
        if (
            findings.validation_report
            and not findings.validation_report.data.indicators.keys() <= indicators
        ):
            raise ValueError("Validation owner does not exist in the snapshot")
        if findings.fit:
            fit = findings.fit
            marginals = fit.posterior_marginals
            if any(item.subject.parameter_id not in parameters for item in marginals):
                raise ValueError("Posterior finding has no scientific parameter definition")
            if not fit.prior_densities.keys() <= parameters:
                raise ValueError("Prior curve has no scientific parameter definition")
        return self
