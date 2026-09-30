"""Revision-pinned access to canonical aggregates and their compatible findings."""

from __future__ import annotations

from functools import cache, cached_property
from typing import TYPE_CHECKING, Literal, cast

from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.identification import IdentificationReport  # noqa: TC001
from nof1_causal_lab.artifacts.identity import GitOid, GitRef
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorEstimate
from nof1_causal_lab.machine.artifact_files import artifact_file_spec, parquet_filename
from nof1_causal_lab.machine.execution import freshness_report, is_stale
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.snapshot_models import (
    FactSource,
    FitSummary,
    ModelData,
    ModelFindings,
    ModelSnapshot,
    SnapshotContext,
    SnapshotState,
    Sourced,
    SourceValidity,
)
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.views import (
    measurements_view,
    model_diagnostics_view,
    raw_data_view,
    read_payload,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl

    from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
    from nof1_causal_lab.artifacts.execution import (
        StructuralItemDisposition,
    )
    from nof1_causal_lab.artifacts.identity import ArtifactId, ConstructId, EntityRef, ParameterId
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.artifacts.prior_predictive import PriorPredictiveResult
    from nof1_causal_lab.artifacts.validation_report import (
        DataProfileArtifact,
        ValidationReportArtifact,
    )
    from nof1_causal_lab.machine.view_models import MeasurementsData, ModelDiagnostics, RawDataData


class SnapshotRevisionNotFound(ValueError):
    """The requested sequence does not identify a committed model revision."""


class ModelReader:
    """Read canonical aggregates through one immutable committed state.

    Aggregate accessors load only their defining artifacts and required ownership inputs.
    The batch adds table-derived views without opening a second revision of "latest".
    Lookup indexes are private to this reader; they are not a second public domain model.
    """

    def __init__(self, workspace_id: str, *, at: GitOid | None = None, branch: str = "main"):
        self.repository = StudyRepository(workspace_id)
        try:
            self.commit_id = self.repository.resolve(branch=branch, at=at)
        except (ValueError, KeyError) as exc:
            raise SnapshotRevisionNotFound(str(exc)) from exc
        self.branch = branch
        self.store = ArtifactStore(workspace_id)
        self.workspace_id = workspace_id
        self.selected = cache(self._selected)

    @cached_property
    def state(self):
        return self.repository.state(self.commit_id)

    @cached_property
    def records(self):
        return self.repository.records(self.commit_id)

    @cached_property
    def seq(self) -> int:
        return self.records[-1].seq if self.records else 0

    @cached_property
    def retracted(self) -> set[ArtifactId]:
        return {
            item.artifact_id
            for record in self.records
            for item in record.retracted
            if not self.state.has(item.artifact_id)
        }

    def _selected(self, artifact_id: ArtifactId):
        return read_payload(
            self.store,
            artifact_id,
            self.state.current[artifact_id].revision,
        )

    def source(
        self, artifact_id: ArtifactId, pointer: str, *, filename: str | None = None
    ) -> FactSource:
        return FactSource(
            ref=GitRef(
                workspace_id=self.workspace_id,
                revision=self.state.current[artifact_id].revision,
                path=filename
                or next(
                    iter(
                        {
                            **artifact_file_spec(artifact_id).parquet,
                            **artifact_file_spec(artifact_id).json,
                        }.values()
                    )
                ),
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

    @cached_property
    def _panel(self) -> pl.DataFrame | None:
        if not self.state.has("panel"):
            return None
        return self.store.read_parquet_file(
            "panel", self.state.current["panel"].revision, parquet_filename("panel", "panel")
        )

    @cached_property
    def raw_data(self) -> Sourced[RawDataData] | None:
        if not self.state.has("raw_data"):
            return None
        return self.fact(
            raw_data_view(
                self.store.read_parquet_table(
                    "raw_data",
                    self.state.current["raw_data"].revision,
                    parquet_filename("raw_data", "raw"),
                )
            ),
            "raw_data",
            "",
        )

    @cached_property
    def measurements(self) -> Sourced[MeasurementsData] | None:
        if self._panel is None:
            return None
        return self.fact(
            measurements_view(self._panel, set(self._panel["indicator_id"].to_list())), "panel", ""
        )

    @cached_property
    def data_metadata(self):
        if not self.state.has("panel"):
            return None
        from nof1_causal_lab.actions.data_checks import read_data_metadata

        return Sourced(
            value=read_data_metadata(self.store, self.state.current["panel"].revision),
            source=self.source("panel", "", filename="metadata.json"),
        )

    @cached_property
    def data_profile(self):
        if not self.state.has("data_profile"):
            return None
        return self.fact(
            cast("DataProfileArtifact", self.selected("data_profile")), "data_profile", ""
        )

    @cached_property
    def validation_report(self) -> Sourced[ValidationReportArtifact] | None:
        if not self.state.has("validation_report"):
            return None
        report = cast("ValidationReportArtifact", self.selected("validation_report"))
        return self.fact(
            report.model_copy(
                update={
                    "indicators": {
                        iid: audit
                        for iid, audit in report.indicators.items()
                        if iid in self._indicator_ids
                    }
                }
            ),
            "validation_report",
            "",
        )

    @cached_property
    def diagnostics(self) -> ModelDiagnostics | None:
        if self.model is None:
            return None
        compatible = self.state.matches_inputs("validation_report", "panel", "model")
        validation = self.validation_report if compatible else None
        predictive = self.prior_predictive if compatible else None
        return model_diagnostics_view(
            self.model,
            panel=self._panel if compatible else None,
            validation=validation.value if validation else None,
            predictive=predictive.value
            if predictive and predictive.source.validity == SourceValidity.FRESH
            else None,
        )

    def artifact_view(self, name: str):
        """Select one projection without evaluating unrelated view builders."""
        match name:
            case "model":
                return self.model
            case "model_diagnostics":
                return self.diagnostics
            case "raw_data":
                finding = self.raw_data
            case "measurements":
                finding = self.measurements
            case "validation_report":
                finding = self.validation_report
            case "prior_predictive":
                finding = self.prior_predictive
            case "inference_report":
                finding = self.inference_report
            case _:
                raise KeyError(name)
        return finding.value if finding is not None else None

    @cached_property
    def inference_report(self) -> Sourced[InferenceReport] | None:
        from nof1_causal_lab.artifacts.posterior import InferenceReport
        from nof1_causal_lab.machine.inference import (
            inference_report_is_current,
            inference_report_record,
            scientific_inference_report,
        )

        if not self.state.has("model"):
            return None
        record = inference_report_record(
            self.records,
            self.state,
        )
        if record is None:
            return None
        assert self.model is not None
        report = scientific_inference_report(
            self.model, InferenceReport.model_validate(record.diagnostics["report"])
        )
        current = inference_report_is_current(record, self.state)
        return Sourced(
            value=report,
            source=FactSource(
                ref=GitRef(
                    workspace_id=self.workspace_id,
                    revision=record.commit_id,
                    path="logs/transition.json",
                ),
                pointer="/diagnostics/report",
                validity=SourceValidity.FRESH if current else SourceValidity.STALE,
            ),
        )

    @cached_property
    def prior_predictive(self) -> Sourced[PriorPredictiveResult] | None:
        from nof1_causal_lab.artifacts.prior_predictive import PriorPredictiveResult
        from nof1_causal_lab.machine.model_spec_results import (
            model_spec_is_current,
            model_spec_record,
        )

        if self.model is None:
            return None
        record = model_spec_record(self.records)
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
                ref=GitRef(
                    workspace_id=self.workspace_id,
                    revision=record.commit_id,
                    path="logs/transition.json",
                ),
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
        read = self.inference_report
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
        return Sourced(
            value=FitSummary(
                report=posterior.summary(),
                edge_estimates=edge_estimates,
                decay_estimates=decay_estimates,
                prior_densities=self.fit_prior_densities(marginals.keys()),
            ),
            source=read.source,
        )

    def fit_prior_densities(self, fitted: Iterable[ParameterId]):
        """Curves of the input laws the fit conditioned, where it reports posteriors."""
        from nof1_causal_lab.compilation_errors import IncompleteModelError
        from nof1_causal_lab.machine.inference import inference_report_record
        from nof1_causal_lab.machine.prior_views import quantity_prior_densities
        from nof1_causal_lab.machine.store import read_model

        record = inference_report_record(self.records, self.state)
        assert record is not None
        try:
            curves = quantity_prior_densities(
                read_model(self.store, record.diagnostics["input_pins"]["model"])
            )
        except IncompleteModelError:
            return {}  # The current compiler places no laws of an input it cannot execute.
        return {identity: curves[identity] for identity in fitted if curves.get(identity)}

    def simulation(self):
        """Return the most recent explicit simulation with its own input revisions."""
        from nof1_causal_lab.artifacts.simulation import SimulationReport

        for record in reversed(self.records):
            if record.operation_id != "simulate" or record.status != "applied":
                continue
            report = TypeAdapter(SimulationReport).validate_python(record.diagnostics["report"])
            pins: dict[ArtifactId, GitOid] = {"model": report.model.revision}
            current = all(
                self.state.has(aid) and self.state.current[aid].revision == revision
                for aid, revision in pins.items()
            )
            return Sourced(
                value=report,
                source=FactSource(
                    ref=GitRef(
                        workspace_id=self.workspace_id,
                        revision=record.commit_id,
                        path="logs/transition.json",
                    ),
                    pointer="/diagnostics/report",
                    validity=SourceValidity.FRESH if current else SourceValidity.STALE,
                ),
            )
        return None

    def check_finding[T](
        self, value: T | None, pointer: str, *, validity: SourceValidity = SourceValidity.FRESH
    ) -> Sourced[T] | None:
        """Source a check committed with this snapshot, without recomputation."""
        if value is None:
            return None
        return Sourced(
            value=value,
            source=FactSource(
                ref=GitRef(
                    workspace_id=self.workspace_id, revision=self.commit_id, path="checks.json"
                ),
                pointer=pointer,
                validity=validity,
            ),
        )

    def snapshot(self) -> ModelSnapshot:
        """Batch the aggregate reads and server-composed table facts at this revision."""
        from nof1_causal_lab.machine.snapshot_models import ModelGraphView
        from nof1_causal_lab.models.model_structure import model_graph_entities

        identification, dispositions = self.identification(), self.dispositions()
        graph_constructs, graph_edges = model_graph_entities(self.model) if self.model else ((), ())
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
        can_simulate = False
        if self.model is not None:
            from nof1_causal_lab.models.ssm.predictive.parameters import validate_simulation_laws
            from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
                _ensure_gaussian_process_diffusion,
            )

            try:
                self.model.check_execution()
                validate_simulation_laws(self.model)
                _ensure_gaussian_process_diffusion(self.model)
            except ValueError:
                pass  # An incomplete or unsupported scientific model has no forward generator.
            else:
                can_simulate = True
        return ModelSnapshot(
            model=self.fact(self.model, "model", "") if self.model else None,
            context=SnapshotContext(
                workspace_id=self.workspace_id,
                seq=self.seq,
                commit_id=self.commit_id,
                branch=self.branch,
                can_simulate=can_simulate,
                state=SnapshotState(current=self.state.current),
                artifacts=[
                    item.model_copy(update={"retracted": item.artifact_id in self.retracted})
                    for item in freshness_report(self.state)
                ],
            ),
            data=ModelData(
                raw_data=self.raw_data,
                measurements=self.measurements,
                metadata=self.data_metadata,
                profile=self.data_profile,
            ),
            findings=ModelFindings(
                identification=identification,
                dispositions=dispositions,
                graph=ModelGraphView(
                    construct_ids=tuple(item.id for item in graph_constructs),
                    edge_ids=tuple(item.id for item in graph_edges),
                    dynamic_construct_ids=tuple(
                        item.id for item in graph_constructs if item.is_dynamic
                    ),
                    status=graph_status,
                ),
                validation_report=self.validation_report,
                prior_predictive=self.prior_predictive,
                diagnostics=self.diagnostics,
                fit=self.fit(),
                simulation=self.simulation(),
                specification=self.check_finding(
                    self.state.checks.specification if self.state.checks else None,
                    "/specification",
                ),
                predictive=self.check_finding(
                    self.state.checks.predictive if self.state.checks else None,
                    "/predictive",
                    validity=SourceValidity.STALE
                    if self.state.checks is not None
                    and self.state.checks.predictive is not None
                    and self.state.checks.predictive.panel_revision
                    != (self.state.current["panel"].revision if self.state.has("panel") else None)
                    else SourceValidity.FRESH,
                ),
            ),
        )
