"""Revision-pinned access to canonical aggregates and their compatible findings."""

from __future__ import annotations

from functools import cache, cached_property
from typing import TYPE_CHECKING, Literal, cast

from nof1_causal_lab.artifacts.causal_design import IdentifiabilityStatus
from nof1_causal_lab.artifacts.identity import ArtifactRef, ModelRef
from nof1_causal_lab.artifacts.mechanism import ConstantDriftMechanism, NodePotentialMechanism
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorEstimate
from nof1_causal_lab.machine.moves import freshness_report, is_stale
from nof1_causal_lab.machine.snapshot_models import (
    FactSource,
    FitSummary,
    ModelSnapshot,
    Sourced,
    SourceValidity,
)
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, replay_state
from nof1_causal_lab.machine.views import read_artifact_views, read_payload

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.baseline_report import (
        BaselineReportArtifact,
        SavedScenariosArtifact,
    )
    from nof1_causal_lab.artifacts.causal_design import CausalDesignArtifact
    from nof1_causal_lab.artifacts.compiled_ssm import CompiledSSMArtifact
    from nof1_causal_lab.artifacts.identity import ArtifactId, EntityRef
    from nof1_causal_lab.artifacts.latent_structure import (
        CausalEdge,
        Construct,
        LatentStructure,
        LatentStructureArtifact,
    )
    from nof1_causal_lab.artifacts.measurement_structure import (
        Indicator,
        MeasurementStructureArtifact,
    )
    from nof1_causal_lab.artifacts.posterior import PosteriorArtifact
    from nof1_causal_lab.artifacts.question import QuestionArtifact
    from nof1_causal_lab.artifacts.statistical_model_spec import (
        ParameterSpec,
        StatisticalModelSpecArtifact,
    )
    from nof1_causal_lab.artifacts.structural_plan import (
        StructuralItemDisposition,
        StructuralPlanArtifact,
    )


class SnapshotRevisionNotFound(ValueError):
    """The requested sequence does not identify a committed model revision."""


def read_revision(workspace_id: str, at_seq: int | None):
    """Read the journal once and select one immutable committed state."""
    records = EpisodeJournal(workspace_id).read_all()
    committed = {record.seq for record in records if record.status == "applied"}
    seq = max(committed, default=0) if at_seq is None else at_seq
    if seq != 0 and seq not in committed:
        raise SnapshotRevisionNotFound(f"Journal sequence {seq} is not a committed model revision")
    state = replay_state(record for record in records if record.seq <= seq)
    installed_at: dict[ArtifactId, int] = {}
    retracted = set()
    for record in records:
        if record.seq > seq or record.status != "applied":
            continue
        for entry in record.retracted:
            installed_at.pop(entry.artifact_id, None)
            retracted.add(entry.artifact_id)
        for info in record.produced:
            installed_at[info.artifact_id] = record.seq
            retracted.discard(info.artifact_id)
    return seq, state, installed_at, sorted(retracted)


class ModelReader:
    """Read canonical aggregates through one immutable committed state.

    Aggregate accessors load only their defining artifacts and required ownership inputs.
    The batch adds table-derived views without opening a second revision of "latest".
    Lookup indexes are private to this reader; they are not a second public domain model.
    """

    def __init__(self, workspace_id: str, *, at_seq: int | None = None):
        self.seq, self.state, self.installed_at, self.retracted = read_revision(
            workspace_id, at_seq
        )
        self.store = ArtifactStore(workspace_id)
        self.model = ModelRef(id=workspace_id)
        self.selected = cache(self._selected)

    def _selected(self, artifact_id: ArtifactId):
        return read_payload(
            self.store,
            ArtifactRef(artifact_id=artifact_id, version=self.state.current[artifact_id].version),
        )

    def source(self, artifact_id: ArtifactId, pointer: str) -> FactSource:
        return FactSource(
            artifact=ArtifactRef(
                artifact_id=artifact_id, version=self.state.current[artifact_id].version
            ),
            pointer=pointer,
            validity=SourceValidity.STALE
            if is_stale(self.state, artifact_id)
            else SourceValidity.FRESH,
        )

    def fact[T](self, value: T, artifact_id: ArtifactId, pointer: str) -> Sourced[T]:
        return Sourced(value=value, source=self.source(artifact_id, pointer))

    def latent_structure(self) -> Sourced[LatentStructure] | None:
        if not self.state.has("latent_structure"):
            return None
        latent = cast("LatentStructureArtifact", self.selected("latent_structure")).latent_structure
        return self.fact(latent, "latent_structure", "/latent_structure")

    def constructs(self) -> tuple[Construct, ...]:
        latent = self.latent_structure()
        return tuple(latent.value.constructs) if latent else ()

    def edges(self) -> tuple[CausalEdge, ...]:
        latent = self.latent_structure()
        return tuple(latent.value.edges) if latent else ()

    @cached_property
    def _construct_ids(self):
        return {item.id for item in self.constructs()}

    def measurement_structure(self) -> Sourced[MeasurementStructureArtifact] | None:
        if not self.state.has("measurement_structure"):
            return None
        measurement = cast("MeasurementStructureArtifact", self.selected("measurement_structure"))
        indicators = measurement.measurement_structure.indicators
        if not is_stale(self.state, "measurement_structure"):
            if any(item.construct_id not in self._construct_ids for item in indicators):
                raise ValueError("Indicator owner does not exist in the selected revision")
        else:
            # Keep the canonical hierarchy while removing relations to deleted current owners.
            # The immutable, unprojected artifact remains available through artifact inspection.
            measurement = measurement.model_copy(
                update={
                    "measurement_structure": measurement.measurement_structure.model_copy(
                        update={
                            "indicators": [
                                item
                                for item in indicators
                                if item.construct_id in self._construct_ids
                            ],
                        }
                    ),
                    "known_inputs": [
                        item
                        for item in measurement.known_inputs
                        if item.construct_id in self._construct_ids
                    ],
                    "scientific_only_constructs": [
                        item
                        for item in measurement.scientific_only_constructs
                        if item.construct_id in self._construct_ids
                    ],
                }
            )
        return self.fact(measurement, "measurement_structure", "")

    def indicators(self) -> tuple[Indicator, ...]:
        measurement = self.measurement_structure()
        return tuple(measurement.value.measurement_structure.indicators) if measurement else ()

    @cached_property
    def _indicator_ids(self):
        return {item.id for item in self.indicators()}

    def parameters(self, owner: EntityRef | None = None) -> tuple[ParameterSpec, ...]:
        if not self.state.has("compiled_ssm"):
            return ()
        compiled = cast("CompiledSSMArtifact", self.selected("compiled_ssm"))
        return tuple(item for item in compiled.parameters if owner is None or owner in item.owners)

    @cached_property
    def _parameter_ids(self):
        return {item.id for item in self.parameters()}

    def specification(self) -> Sourced[StatisticalModelSpecArtifact] | None:
        if not self.state.has("statistical_model_spec"):
            return None
        spec = cast("StatisticalModelSpecArtifact", self.selected("statistical_model_spec"))
        compatible = self.state.matches_inputs("compiled_ssm", "statistical_model_spec")
        edge_ids = {item.id for item in self.edges()}
        return self.fact(
            spec.model_copy(
                update={
                    "statistical_model_spec": spec.statistical_model_spec.model_copy(
                        update={
                            "mechanisms": [
                                item
                                for item in spec.statistical_model_spec.mechanisms
                                if (
                                    item.target_id in self._construct_ids
                                    if isinstance(
                                        item, (NodePotentialMechanism, ConstantDriftMechanism)
                                    )
                                    else item.edge_id in edge_ids
                                )
                            ],
                            "likelihoods": [
                                item
                                for item in spec.statistical_model_spec.likelihoods
                                if item.indicator_id in self._indicator_ids
                            ],
                        }
                    ),
                    "resolved_priors": [
                        item
                        for item in spec.resolved_priors
                        if compatible and item.parameter_id in self._parameter_ids
                    ],
                    "prior_predictive_diagnostics": [
                        item
                        for item in spec.prior_predictive_diagnostics
                        if item.construct_id in self._construct_ids
                    ],
                }
            ),
            "statistical_model_spec",
            "",
        )

    def posterior(self) -> Sourced[PosteriorArtifact] | None:
        if not self.state.has("posterior"):
            return None
        posterior = cast("PosteriorArtifact", self.selected("posterior"))
        compatible = self.state.matches_inputs("posterior", "compiled_ssm")
        mcmc = posterior.assessment.mcmc_diagnostics
        # Joint fit metadata and predictive checks remain inspectable when a compiler changes.
        # Parameter findings can only be attached to the compiler that defined their coordinates.
        return self.fact(
            posterior.model_copy(
                update={
                    "posterior_marginals": [
                        item
                        for item in posterior.posterior_marginals or []
                        if compatible and item.subject.parameter_id in self._parameter_ids
                    ],
                    "posterior_pairs": [
                        item
                        for item in posterior.posterior_pairs or []
                        if compatible
                        and item.subject_x.parameter_id in self._parameter_ids
                        and item.subject_y.parameter_id in self._parameter_ids
                    ],
                    "assessment": posterior.assessment.model_copy(
                        update={
                            "mcmc_diagnostics": mcmc.model_copy(
                                update={
                                    "per_parameter": [
                                        item
                                        for item in mcmc.per_parameter
                                        if compatible
                                        and item.subject.parameter_id in self._parameter_ids
                                    ],
                                }
                            )
                            if mcmc
                            else None,
                        }
                    ),
                }
            ),
            "posterior",
            "",
        )

    def identification(self) -> Sourced[IdentifiabilityStatus] | None:
        if not self.state.has("causal_design"):
            return None
        identification = cast(
            "CausalDesignArtifact", self.selected("causal_design")
        ).causal_design.identifiability
        if identification is None:
            return None
        return self.fact(
            IdentifiabilityStatus(
                identifiable_treatments={
                    cid: item
                    for cid, item in identification.identifiable_treatments.items()
                    if cid in self._construct_ids
                },
                non_identifiable_treatments={
                    cid: item
                    for cid, item in identification.non_identifiable_treatments.items()
                    if cid in self._construct_ids
                },
            ),
            "causal_design",
            "/causal_design/identifiability",
        )

    def dispositions(self) -> Sourced[tuple[StructuralItemDisposition, ...]] | None:
        if not self.state.has("structural_plan"):
            return None
        plan = cast("StructuralPlanArtifact", self.selected("structural_plan")).structural_plan
        owners = self._construct_ids | self._indicator_ids | {item.id for item in self.edges()}
        return self.fact(
            tuple(item for item in plan.dispositions if item.source_id in owners),
            "structural_plan",
            "/structural_plan/dispositions",
        )

    def fit(self) -> Sourced[FitSummary] | None:
        read = self.posterior()
        if read is None:
            return None
        posterior = read.value
        marginals = {}
        for item in posterior.posterior_marginals or []:
            marginals.setdefault(item.subject.parameter_id, []).append(item)
        edge_estimates, decay_estimates = {}, {}
        for parameter in self.parameters():
            findings = marginals.get(parameter.id, [])
            if len(findings) != 1:
                continue
            estimate = PosteriorEstimate.model_validate(findings[0], from_attributes=True)
            for owner in parameter.owners:
                if parameter.quantity == SiteKind.DYNAMICS_WEIGHT and owner.kind == "edge":
                    edge_estimates[owner.id] = estimate
                elif parameter.quantity == SiteKind.DYNAMICS_DECAY and owner.kind == "construct":
                    decay_estimates[owner.id] = estimate
        warnings = posterior.assessment.ppc.per_variable_warnings
        return self.fact(
            FitSummary(
                posterior=posterior,
                predictive_checks_passed=sum(item.passed for item in warnings),
                predictive_checks_total=len(warnings),
                edge_estimates=edge_estimates,
                decay_estimates=decay_estimates,
            ),
            "posterior",
            "",
        )

    def snapshot(self) -> ModelSnapshot:
        """Batch the aggregate reads and server-composed table facts at this revision."""
        identification, dispositions = self.identification(), self.dispositions()
        blocking = set()
        # Include confounders of historical treatments, even if that treatment was removed.
        if self.state.has("causal_design"):
            status = cast(
                "CausalDesignArtifact", self.selected("causal_design")
            ).causal_design.identifiability
            if status:
                for cid, finding in status.non_identifiable_treatments.items():
                    blocking.update([cid, *finding.confounders])
        disposition_by_id = (
            {item.source_id: item for item in dispositions.value} if dispositions else {}
        )
        graph_status: dict[str, Literal["observed", "marginalized", "blocking"]] = {
            cid: "blocking"
            if cid in blocking
            else "observed"
            if disposition_by_id[cid].disposition in {"retained_state", "known_input"}
            else "marginalized"
            for cid in self._construct_ids
            if cid in disposition_by_id
        }
        views = read_artifact_views(self.store, self.state, self.installed_at)
        measurements = views.measurements
        if measurements:
            measurements = measurements.model_copy(
                update={
                    "per_indicator_counts": {
                        iid: count
                        for iid, count in measurements.per_indicator_counts.items()
                        if iid in self._indicator_ids
                    },
                }
            )
        validation = views.validation_report
        if validation:
            validation = validation.model_copy(
                update={
                    "indicators": {
                        iid: audit
                        for iid, audit in validation.indicators.items()
                        if iid in self._indicator_ids
                    },
                }
            )
        report = None
        if self.state.has("baseline_report"):
            selected_report = cast("BaselineReportArtifact", self.selected("baseline_report"))
            report = self.fact(
                selected_report.model_copy(
                    update={
                        "intervention_results": [
                            item
                            for item in selected_report.intervention_results
                            if item.treatment_id in self._construct_ids
                        ],
                    }
                ),
                "baseline_report",
                "",
            )
        return ModelSnapshot(
            model=self.model,
            seq=self.seq,
            state=self.state,
            question=self.fact(cast("QuestionArtifact", self.selected("question")), "question", "")
            if self.state.has("question")
            else None,
            latent_structure=self.latent_structure(),
            measurement_structure=self.measurement_structure(),
            identification=identification,
            dispositions=dispositions,
            graph_status=graph_status,
            raw_data=self.fact(views.raw_data, "raw_data", "") if views.raw_data else None,
            measurements=self.fact(measurements, "panel", "") if measurements else None,
            validation_report=self.fact(validation, "validation_report", "")
            if validation
            else None,
            specification=self.specification(),
            compiled_parameters=self.fact(self.parameters(), "compiled_ssm", "/parameters")
            if self.state.has("compiled_ssm")
            else None,
            fit=self.fit(),
            baseline_report=report,
            saved_scenarios=self.fact(
                cast("SavedScenariosArtifact", self.selected("saved_scenarios")),
                "saved_scenarios",
                "",
            )
            if self.state.has("saved_scenarios")
            else None,
            artifacts=freshness_report(self.state),
            installed_at=self.installed_at,
            retracted=self.retracted,
        )


def read_model_snapshot(workspace_id: str, *, at_seq: int | None = None) -> ModelSnapshot:
    """Read the UI batch through a single pinned accessor transaction."""
    return ModelReader(workspace_id, at_seq=at_seq).snapshot()
