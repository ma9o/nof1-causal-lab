"""Aggregate reads for one committed model revision."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nof1_causal_lab.artifacts.checks import SpecificationReport  # noqa: TC001
from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata  # noqa: TC001
from nof1_causal_lab.artifacts.execution import StructuralItemDisposition  # noqa: TC001
from nof1_causal_lab.artifacts.identification import IdentificationReport  # noqa: TC001
from nof1_causal_lab.artifacts.identity import (  # noqa: TC001
    ConstructId,
    EdgeId,
    GitOid,
    GitRef,
)
from nof1_causal_lab.artifacts.model_checks import ModelPredictiveReport  # noqa: TC001
from nof1_causal_lab.artifacts.model_spec import ModelSpec  # noqa: TC001
from nof1_causal_lab.artifacts.posterior import InferenceReport  # noqa: TC001
from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorEstimate  # noqa: TC001
from nof1_causal_lab.artifacts.prior_predictive import PriorPredictiveResult  # noqa: TC001
from nof1_causal_lab.artifacts.simulation import SimulationReport  # noqa: TC001
from nof1_causal_lab.artifacts.validation_report import (  # noqa: TC001
    DataProfileArtifact,
    ValidationReportArtifact,
)
from nof1_causal_lab.machine.artifacts import EpisodeState  # noqa: TC001
from nof1_causal_lab.machine.execution import ArtifactFreshness, is_stale
from nof1_causal_lab.machine.view_models import (  # noqa: TC001
    MeasurementsData,
    ModelDiagnostics,
    RawDataData,
)


class SnapshotValue(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, json_schema_extra={"x-python-module": __name__}
    )


class SourceValidity(StrEnum):
    """Source validity records whether a fact still matches its pinned inputs."""

    FRESH = "fresh"
    STALE = "stale"


class FactSource(SnapshotValue):
    """A fact source locates supporting content within an artifact revision and records its freshness."""

    ref: GitRef
    pointer: str = Field(pattern=r"^(?:/.*)?$")
    validity: SourceValidity


class Sourced[T](SnapshotValue):
    """A sourced read pairs a canonical aggregate or derived finding with its artifact revision."""

    value: T
    source: FactSource


class FitSummary(SnapshotValue):
    """A fit read contains the inference log report and server-composed display findings."""

    report: InferenceReport
    edge_estimates: dict[EdgeId, PosteriorEstimate] = Field(default_factory=dict)
    decay_estimates: dict[ConstructId, PosteriorEstimate] = Field(default_factory=dict)


class SnapshotContext(SnapshotValue):
    """A snapshot context identifies the selected Git commit and its artifact versions."""

    workspace_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    commit_id: GitOid
    branch: str = "main"
    can_simulate: bool = False
    state: EpisodeState
    artifacts: list[ArtifactFreshness] = Field(default_factory=list)


class ModelData(SnapshotValue):
    """Observed evidence paired with its source versions."""

    raw_data: Sourced[RawDataData] | None = None
    measurements: Sourced[MeasurementsData] | None = None
    metadata: Sourced[PreparedDataMetadata] | None = None
    profile: Sourced[DataProfileArtifact] | None = None


class ModelGraphView(SnapshotValue):
    """Scientific entity identities selected for the graph at this authoring checkpoint."""

    construct_ids: tuple[ConstructId, ...] = ()
    edge_ids: tuple[EdgeId, ...] = ()
    status: dict[ConstructId, Literal["observed", "marginalized", "blocking"]] = Field(
        default_factory=dict
    )


class ModelFindings(SnapshotValue):
    """ModelSpec findings collect identification, validation, and fitted results with their input references."""

    identification: Sourced[IdentificationReport] | None = None
    dispositions: Sourced[tuple[StructuralItemDisposition, ...]] | None = None
    graph: ModelGraphView = Field(default_factory=ModelGraphView)
    validation_report: Sourced[ValidationReportArtifact] | None = None
    prior_predictive: Sourced[PriorPredictiveResult] | None = None
    diagnostics: ModelDiagnostics | None = None
    fit: Sourced[FitSummary] | None = None
    specification: Sourced[SpecificationReport] | None = None
    simulation: Sourced[SimulationReport] | None = None
    predictive: Sourced[ModelPredictiveReport] | None = None


class ModelSnapshot(SnapshotValue):
    """The canonical scientific definition with independently sourced inputs and findings."""

    model: Sourced[ModelSpec] | None = None
    context: SnapshotContext
    data: ModelData = Field(default_factory=ModelData)
    findings: ModelFindings = Field(default_factory=ModelFindings)

    @model_validator(mode="after")
    def validate_ownership_and_sources(self) -> ModelSnapshot:
        for read, artifact_id in (
            (self.model, "model"),
            (self.findings.identification, "identification_report"),
            (self.findings.dispositions, "model"),
            (self.data.raw_data, "raw_data"),
            (self.data.measurements, "panel"),
            (self.data.metadata, "panel"),
            (self.data.profile, "data_profile"),
            (self.findings.validation_report, "validation_report"),
            (self.findings.prior_predictive, "prior_predictive"),
            (self.findings.fit, "inference"),
            (self.findings.specification, "specification"),
            (self.findings.simulation, "simulation"),
            (self.findings.predictive, "predictive"),
        ):
            if read is not None:
                self._validate_source(read.source, artifact_id)
        model = self.model.value if self.model else None
        constructs = {item.id for item in model.constructs} if model else set()
        edges = {item.id for item in model.edges} if model else set()
        indicators = {item.id for item in model.indicators} if model else set()
        parameters = {item.id for item in model.parameters} if model else set()
        findings = self.findings
        if (
            not set(findings.graph.construct_ids) <= constructs
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
        if self.data.measurements and self.data.metadata is None:
            raise ValueError("Prepared observations require their data-owned metadata")
        # Counts describe the stored table, including data-quality problems.
        # Undeclared variables are reported by preparation checks, never hidden here.
        if (
            findings.validation_report
            and not findings.validation_report.value.indicators.keys() <= indicators
        ):
            raise ValueError("Validation owner does not exist in the snapshot")
        if findings.prior_predictive:
            predictive = findings.prior_predictive.value
            if any(item.construct_id not in constructs for item in predictive.diagnostics):
                raise ValueError("Prior-predictive check owner does not exist in the snapshot")
            if not predictive.samples.keys() <= indicators:
                raise ValueError(
                    "Prior-predictive observation owner does not exist in the snapshot"
                )
        if findings.fit:
            fit = findings.fit.value
            marginals = fit.report.posterior_marginals or []
            pairs = fit.report.posterior_pairs or []
            if any(item.subject.parameter_id not in parameters for item in marginals):
                raise ValueError("Posterior finding has no scientific parameter definition")
            if any(
                subject.parameter_id not in parameters
                for pair in pairs
                for subject in (pair.subject_x, pair.subject_y)
            ):
                raise ValueError("Posterior pair has no scientific parameter definition")
        return self

    def _validate_source(self, source: FactSource, artifact_id: str) -> None:
        from nof1_causal_lab.machine.artifact_files import artifact_file_spec

        ref = source.ref
        if ref.workspace_id != self.context.workspace_id:
            raise ValueError("Fact source belongs to another study")
        if ref.path == "checks.json":
            if (
                artifact_id not in {"specification", "predictive"}
                or ref.revision != self.context.commit_id
                or self.context.state.checks is None
                or source.pointer != f"/{artifact_id}"
            ):
                raise ValueError("Check source must identify this snapshot's recorded findings")
            predictive = self.context.state.checks.predictive
            panel = self.context.state.get("panel")
            expected = "stale" if artifact_id == "predictive" and predictive is not None and (
                predictive.panel_revision != (panel.revision if panel else None)
            ) else "fresh"
            if source.validity != expected:
                raise ValueError("Check validity differs from its selected observation revision")
            return
        if ref.path == "logs/transition.json":
            if artifact_id not in {"inference", "prior_predictive", "simulation", "specification"}:
                raise ValueError("Only operation findings refer to action logs")
            # The repository reader selects these logs through Git ancestry.
            return
        current = self.context.state.current.get(artifact_id)
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
        expected = "stale" if is_stale(self.context.state, current.artifact_id) else "fresh"
        if source.validity != expected:
            raise ValueError("Fact validity differs from its snapshot input references")
