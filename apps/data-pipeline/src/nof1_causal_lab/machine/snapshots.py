"""Revision-pinned access to canonical aggregates and their compatible findings."""

from __future__ import annotations

from functools import cache, cached_property
from typing import TYPE_CHECKING, Literal, cast

from nof1_causal_lab.artifacts.identification import IdentificationReport  # noqa: TC001
from nof1_causal_lab.artifacts.identity import ArtifactRef, TransitionRef
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorEstimate
from nof1_causal_lab.machine.moves import freshness_report, is_stale
from nof1_causal_lab.machine.snapshot_models import (
    FactSource,
    FitSummary,
    ModelData,
    ModelFindings,
    ModelSnapshot,
    SnapshotContext,
    Sourced,
    SourceValidity,
)
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal, replay_state
from nof1_causal_lab.machine.views import read_artifact_views, read_payload

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
    from nof1_causal_lab.artifacts.execution import (
        StructuralItemDisposition,
    )
    from nof1_causal_lab.artifacts.identity import ArtifactId, ConstructId, EntityRef
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.artifacts.prior_predictive import PriorPredictiveResult


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
        self.workspace_id = workspace_id
        self.selected = cache(self._selected)

    def _selected(self, artifact_id: ArtifactId):
        return read_payload(
            self.store,
            ArtifactRef(artifact_id=artifact_id, version=self.state.current[artifact_id].version),
        )

    def source(self, artifact_id: ArtifactId, pointer: str) -> FactSource:
        return FactSource(
            ref=ArtifactRef(
                artifact_id=artifact_id, version=self.state.current[artifact_id].version
            ),
            pointer=pointer,
            validity=SourceValidity.STALE
            if is_stale(self.state, artifact_id)
            else SourceValidity.FRESH,
        )

    def fact[T](self, value: T, artifact_id: ArtifactId, pointer: str) -> Sourced[T]:
        return Sourced(value=value, source=self.source(artifact_id, pointer))

    @cached_property
    def model(self) -> ModelSpec | None:
        return cast("ModelSpec", self.selected("model")) if self.state.has("model") else None

    def constructs(self) -> tuple[ConstructSpec, ...]:
        return self.model.constructs if self.model else ()

    def edges(self) -> tuple[CausalEdgeSpec, ...]:
        return self.model.edges if self.model else ()

    def indicators(self) -> tuple[IndicatorSpec, ...]:
        return self.model.indicators if self.model else ()

    def parameters(self, owner: EntityRef | None = None) -> tuple[ParameterSpec, ...]:
        if self.model is None:
            return ()
        return self.model.parameters if owner is None else self.model.parameters_for(owner.id)

    @cached_property
    def _construct_ids(self):
        return {item.id for item in self.constructs()}

    @cached_property
    def _indicator_ids(self):
        return {item.id for item in self.indicators()}

    def inference_report(self) -> Sourced[InferenceReport] | None:
        from nof1_causal_lab.artifacts.posterior import InferenceReport
        from nof1_causal_lab.machine.inference import (
            inference_report_is_current,
            inference_report_record,
        )

        if not self.state.has("model"):
            return None
        record = inference_report_record(
            (
                record
                for record in EpisodeJournal(self.store.workspace_id).read_all()
                if record.seq <= self.seq
            ),
            self.state,
        )
        if record is None:
            return None
        report = InferenceReport.model_validate(record.diagnostics["report"])
        current = inference_report_is_current(record, self.state)
        return Sourced(
            value=report
            if current
            else report.model_copy(update={"posterior_marginals": [], "posterior_pairs": []}),
            source=FactSource(
                ref=TransitionRef(seq=record.seq),
                pointer="/diagnostics/report",
                validity=SourceValidity.FRESH if current else SourceValidity.STALE,
            ),
        )

    def prior_predictive(self) -> Sourced[PriorPredictiveResult] | None:
        from nof1_causal_lab.artifacts.prior_predictive import PriorPredictiveResult
        from nof1_causal_lab.machine.model_spec_results import (
            model_spec_is_current,
            model_spec_record,
        )

        if self.model is None:
            return None
        record = model_spec_record(
            record
            for record in EpisodeJournal(self.workspace_id).read_all()
            if record.seq <= self.seq
        )
        if record is None or (payload := record.diagnostics.get("prior_predictive")) is None:
            return None
        result = PriorPredictiveResult.model_validate(payload)
        return Sourced(
            value=result.model_copy(
                update={
                    "samples": {
                        iid: samples
                        for iid, samples in result.samples.items()
                        if iid in self._indicator_ids
                    },
                    "diagnostics": [
                        item
                        for item in result.diagnostics
                        if item.construct_id in self._construct_ids
                    ],
                }
            ),
            source=FactSource(
                ref=TransitionRef(seq=record.seq),
                pointer="/diagnostics/prior_predictive",
                validity=SourceValidity.FRESH
                if model_spec_is_current(record, self.state, self.store)
                else SourceValidity.STALE,
            ),
        )

    def identification(self) -> Sourced[IdentificationReport] | None:
        if not self.state.has("identification_report"):
            return None
        report = cast("IdentificationReport", self.selected("identification_report"))
        if self.model is None:
            raise ValueError("Identification requires its scientific model")
        report.validate_model(self.model)
        return self.fact(report, "identification_report", "")

    def dispositions(self) -> Sourced[tuple[StructuralItemDisposition, ...]] | None:
        if self.model is None or self.model.measurement_clock is None or not self.model.indicators:
            return None
        owners = self._construct_ids | self._indicator_ids | {item.id for item in self.edges()}
        return self.fact(
            tuple(item for item in self.model.structural_dispositions if item.target.id in owners),
            "model",
            "",
        )

    def fit(self) -> Sourced[FitSummary] | None:
        read = self.inference_report()
        if read is None:
            return None
        posterior = read.value
        marginals = {}
        for item in posterior.posterior_marginals or []:
            marginals.setdefault(item.subject.parameter_id, []).append(item)
        edge_estimates, decay_estimates = {}, {}
        model = self.model
        assert model is not None
        for parameter in self.parameters():
            findings = marginals.get(parameter.id, [])
            if len(findings) != 1:
                continue
            estimate = PosteriorEstimate.model_validate(findings[0], from_attributes=True)
            for owner in model.parameter_context(parameter.id).owners:
                if (
                    model.parameter_context(parameter.id).quantity == SiteKind.DYNAMICS_WEIGHT
                    and owner.kind == "edge"
                ):
                    edge_estimates[owner.id] = estimate
                elif (
                    model.parameter_context(parameter.id).quantity == SiteKind.DYNAMICS_DECAY
                    and owner.kind == "construct"
                ):
                    decay_estimates[owner.id] = estimate
        warnings = posterior.ppc.per_variable_warnings
        return Sourced(
            value=FitSummary(
                report=posterior,
                predictive_checks_passed=sum(item.passed for item in warnings),
                predictive_checks_total=len(warnings),
                edge_estimates=edge_estimates,
                decay_estimates=decay_estimates,
            ),
            source=read.source,
        )

    def snapshot(self) -> ModelSnapshot:
        """Batch the aggregate reads and server-composed table facts at this revision."""
        from nof1_causal_lab.machine.inference import inference_is_current

        identification, dispositions = self.identification(), self.dispositions()
        blocking = set()
        if identification:
            for cid, finding in identification.value.non_identifiable.items():
                blocking.update([cid, *finding.confounders])
        disposition_by_id = (
            {item.target.id: item for item in dispositions.value} if dispositions else {}
        )
        graph_status: dict[ConstructId, Literal["observed", "marginalized", "blocking"]] = {
            cid: "blocking"
            if cid in blocking or disposition_by_id[cid].disposition == "unsupported"
            else "observed"
            if disposition_by_id[cid].disposition == "retained_state"
            else "marginalized"
            for cid in self._construct_ids
            if cid in disposition_by_id
        }
        views = read_artifact_views(self.store, self.state, at_seq=self.seq)
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
        return ModelSnapshot(
            model=self.fact(self.model, "model", "") if self.model else None,
            context=SnapshotContext(
                workspace_id=self.workspace_id,
                seq=self.seq,
                can_simulate=bool(
                    self.model
                    and self.model.distributions
                    and self.model.time_points
                    and inference_is_current(self.state)
                    and identification
                    and identification.value.estimable_treatments
                ),
                state=self.state,
                artifacts=freshness_report(self.state),
                installed_at=self.installed_at,
                retracted=self.retracted,
            ),
            data=ModelData(
                raw_data=self.fact(views.raw_data, "raw_data", "") if views.raw_data else None,
                measurements=self.fact(measurements, "panel", "") if measurements else None,
            ),
            findings=ModelFindings(
                identification=identification,
                dispositions=dispositions,
                graph_status=graph_status,
                validation_report=self.fact(validation, "validation_report", "")
                if validation
                else None,
                prior_predictive=self.prior_predictive(),
                diagnostics=views.model_diagnostics,
                fit=self.fit(),
            ),
        )
