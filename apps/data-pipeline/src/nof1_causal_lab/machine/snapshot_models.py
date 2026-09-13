"""Aggregate reads for one committed model revision."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nof1_causal_lab.artifacts.baseline_report import (  # noqa: TC001
    BaselineReportArtifact,
    SavedScenariosArtifact,
)
from nof1_causal_lab.artifacts.causal_design import IdentifiabilityStatus  # noqa: TC001
from nof1_causal_lab.artifacts.identity import (  # noqa: TC001
    ArtifactId,
    ArtifactRef,
    ConstructId,
    EdgeId,
    ModelRef,
)
from nof1_causal_lab.artifacts.latent_structure import LatentStructure  # noqa: TC001
from nof1_causal_lab.artifacts.measurement_structure import (
    MeasurementStructureArtifact,  # noqa: TC001
)
from nof1_causal_lab.artifacts.mechanism import ConstantDriftMechanism, NodePotentialMechanism
from nof1_causal_lab.artifacts.posterior import PosteriorArtifact  # noqa: TC001
from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorEstimate  # noqa: TC001
from nof1_causal_lab.artifacts.question import QuestionArtifact  # noqa: TC001
from nof1_causal_lab.artifacts.statistical_model_spec import (  # noqa: TC001
    ParameterSpec,
    StatisticalModelSpecArtifact,
)
from nof1_causal_lab.artifacts.structural_plan import StructuralItemDisposition  # noqa: TC001
from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact  # noqa: TC001
from nof1_causal_lab.machine.artifacts import EpisodeState  # noqa: TC001
from nof1_causal_lab.machine.moves import ArtifactFreshness, is_stale
from nof1_causal_lab.machine.view_models import MeasurementsData, RawDataData  # noqa: TC001


class SnapshotValue(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, json_schema_extra={"x-python-module": __name__}
    )


class SourceValidity(StrEnum):
    """Source validity records whether a fact still matches its pinned inputs."""

    FRESH = "fresh"
    STALE = "stale"


class FactSource(SnapshotValue):
    """A fact source locates supporting content within an artifact version and records its freshness."""

    artifact: ArtifactRef
    pointer: str = Field(pattern=r"^(?:/.*)?$")
    validity: SourceValidity


class Sourced[T](SnapshotValue):
    """A sourced read pairs a canonical aggregate or derived finding with its artifact version."""

    value: T
    source: FactSource


class FitSummary(SnapshotValue):
    """A fit read contains the canonical posterior and server-composed display findings."""

    posterior: PosteriorArtifact
    predictive_checks_passed: int = Field(ge=0)
    predictive_checks_total: int = Field(ge=0)
    edge_estimates: dict[EdgeId, PosteriorEstimate] = Field(default_factory=dict)
    decay_estimates: dict[ConstructId, PosteriorEstimate] = Field(default_factory=dict)


class ModelSnapshot(SnapshotValue):
    """A model snapshot batches independently sourced aggregates at one committed revision.

    Authored structure, measurement declarations, specification, and posterior retain their
    canonical hierarchy. Optional reads represent partial models; each source preserves its
    own version and freshness. Only cross-artifact ownership and provenance belong here.
    """

    model: ModelRef
    seq: int = Field(ge=0)
    state: EpisodeState
    question: Sourced[QuestionArtifact] | None = None
    latent_structure: Sourced[LatentStructure] | None = None
    measurement_structure: Sourced[MeasurementStructureArtifact] | None = None
    identification: Sourced[IdentifiabilityStatus] | None = None
    dispositions: Sourced[tuple[StructuralItemDisposition, ...]] | None = None
    graph_status: dict[ConstructId, Literal["observed", "marginalized", "blocking"]] = Field(
        default_factory=dict
    )
    raw_data: Sourced[RawDataData] | None = None
    measurements: Sourced[MeasurementsData] | None = None
    validation_report: Sourced[ValidationReportArtifact] | None = None
    specification: Sourced[StatisticalModelSpecArtifact] | None = None
    compiled_parameters: Sourced[tuple[ParameterSpec, ...]] | None = None
    fit: Sourced[FitSummary] | None = None
    baseline_report: Sourced[BaselineReportArtifact] | None = None
    saved_scenarios: Sourced[SavedScenariosArtifact] | None = None
    artifacts: list[ArtifactFreshness] = Field(default_factory=list)
    installed_at: dict[ArtifactId, int] = Field(default_factory=dict)
    retracted: list[ArtifactId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ownership_and_sources(self) -> ModelSnapshot:
        for read, artifact_id in (
            (self.question, "question"),
            (self.latent_structure, "latent_structure"),
            (self.measurement_structure, "measurement_structure"),
            (self.identification, "causal_design"),
            (self.dispositions, "structural_plan"),
            (self.raw_data, "raw_data"),
            (self.measurements, "panel"),
            (self.validation_report, "validation_report"),
            (self.specification, "statistical_model_spec"),
            (self.compiled_parameters, "compiled_ssm"),
            (self.fit, "posterior"),
            (self.baseline_report, "baseline_report"),
            (self.saved_scenarios, "saved_scenarios"),
        ):
            if read is not None:
                self._validate_source(read.source, artifact_id)

        latent = self.latent_structure.value if self.latent_structure else None
        constructs = {item.id for item in latent.constructs} if latent else set()
        edges = {item.id for item in latent.edges} if latent else set()
        measurement = self.measurement_structure.value if self.measurement_structure else None
        indicators = (
            {item.id for item in measurement.measurement_structure.indicators}
            if measurement
            else set()
        )
        if measurement:
            if any(
                item.construct_id not in constructs
                for item in measurement.measurement_structure.indicators
            ):
                raise ValueError("Indicator owner does not exist in the snapshot")
            if any(
                item.construct_id not in constructs
                for item in (*measurement.known_inputs, *measurement.scientific_only_constructs)
            ):
                raise ValueError("Measurement declaration owner does not exist in the snapshot")
        if self.dispositions and any(
            item.source_id not in constructs | edges | indicators
            for item in self.dispositions.value
        ):
            raise ValueError("Disposition owner does not exist in the snapshot")
        if not self.graph_status.keys() <= constructs:
            raise ValueError("Graph status owner does not exist in the snapshot")
        if (
            self.identification
            and not (
                self.identification.value.identifiable_treatments.keys()
                | self.identification.value.non_identifiable_treatments.keys()
            )
            <= constructs
        ):
            raise ValueError("Identification owner does not exist in the snapshot")
        if (
            self.measurements
            and not self.measurements.value.per_indicator_counts.keys() <= indicators
        ):
            raise ValueError("Observation owner does not exist in the snapshot")
        if (
            self.validation_report
            and not self.validation_report.value.indicators.keys() <= indicators
        ):
            raise ValueError("Validation owner does not exist in the snapshot")

        parameters = (
            {item.id for item in self.compiled_parameters.value}
            if self.compiled_parameters
            else set()
        )
        if self.specification:
            spec = self.specification.value
            for mechanism in spec.statistical_model_spec.mechanisms:
                if isinstance(mechanism, (NodePotentialMechanism, ConstantDriftMechanism)):
                    if mechanism.target_id not in constructs:
                        raise ValueError("Mechanism target does not exist in the snapshot")
                elif mechanism.edge_id not in edges:
                    raise ValueError("Mechanism edge does not exist in the snapshot")
            if any(
                item.indicator_id not in indicators
                for item in spec.statistical_model_spec.likelihoods
            ):
                raise ValueError("Likelihood owner does not exist in the snapshot")
            if any(
                item.construct_id not in constructs for item in spec.prior_predictive_diagnostics
            ):
                raise ValueError("Admission owner does not exist in the snapshot")
            if any(item.parameter_id not in parameters for item in spec.resolved_priors):
                raise ValueError("Prior finding has no compiled parameter definition")
            if spec.resolved_priors and not self.state.matches_inputs(
                "compiled_ssm", "statistical_model_spec"
            ):
                raise ValueError("Prior findings require the selected specification's compiler")
        if self.fit:
            fit = self.fit.value
            marginals = fit.posterior.posterior_marginals or []
            pairs = fit.posterior.posterior_pairs or []
            mcmc = fit.posterior.assessment.mcmc_diagnostics
            diagnostics = mcmc.per_parameter if mcmc else []
            if any(
                item.subject.parameter_id not in parameters for item in (*marginals, *diagnostics)
            ):
                raise ValueError("Posterior finding has no compiled parameter definition")
            if any(
                subject.parameter_id not in parameters
                for pair in pairs
                for subject in (pair.subject_x, pair.subject_y)
            ):
                raise ValueError("Posterior pair has no compiled parameter definition")
            if (
                marginals or diagnostics or pairs or fit.edge_estimates or fit.decay_estimates
            ) and not self.state.matches_inputs("posterior", "compiled_ssm"):
                raise ValueError("Posterior findings require the selected compiler version")
        return self

    def _validate_source(self, source: FactSource, artifact_id: str) -> None:
        ref = source.artifact
        current = self.state.get(ref.artifact_id)
        if ref.artifact_id != artifact_id or current is None or current.version != ref.version:
            raise ValueError("Fact source does not belong to the selected artifact snapshot")
        expected = "stale" if is_stale(self.state, ref.artifact_id) else "fresh"
        if source.validity != expected:
            raise ValueError("Fact validity differs from its snapshot provenance")
