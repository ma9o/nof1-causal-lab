"""Revision-pinned access to canonical aggregates and their compatible findings."""

from __future__ import annotations

from datetime import date, datetime
from functools import cache, cached_property
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import polars as pl

from nof1_causal_lab.artifacts.availability import Available, Unavailable
from nof1_causal_lab.artifacts.identity import GitOid, GitRef, ParameterRef
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.model_parameters import execution_parameters
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.numpyro_json import distribution_shape
from nof1_causal_lab.study.artifact_files import artifact_file_spec, parquet_filename
from nof1_causal_lab.study.equations import (
    confounder_equations,
    observation_equations,
    state_equations,
)
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.prior_views import prior_density
from nof1_causal_lab.study.records import Applied, StudyRevision
from nof1_causal_lab.study.snapshot_models import (
    FactSource,
    FitSummary,
    ModelSnapshot,
    Sourced,
)
from nof1_causal_lab.study.state import SourceValidity, StudyState, is_stale
from nof1_causal_lab.study.store import ArtifactStore, observation_sample, read_payload
from nof1_causal_lab.study.views import (
    entity_failures,
    likelihood_histograms,
    measurements_view,
    raw_data_view,
)
from nof1_causal_lab.study.visual_models import (
    ObservationHistory,
    ParameterDrawColumn,
    ParameterDraws,
    SimulationPaths,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
    from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
    from nof1_causal_lab.artifacts.execution import (
        StructuralItemDisposition,
    )
    from nof1_causal_lab.artifacts.identification import IdentificationReport
    from nof1_causal_lab.artifacts.identity import (
        ArtifactId,
        ConstructId,
        EntityRef,
        IndicatorId,
        ParameterId,
    )
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.artifacts.posterior_diagnostics import DensityCurve, PPCOverlay
    from nof1_causal_lab.artifacts.question import QuestionSpec
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.artifacts.validation_report import (
        DataProfileArtifact,
        ValidationReportArtifact,
    )
    from nof1_causal_lab.study.view_models import (
        MeasurementsData,
        RawDataData,
    )


class ModelReader:
    """Read canonical aggregates through one immutable committed state.

    Aggregate accessors load only their defining artifacts and required ownership inputs.
    The batch adds table-derived views without opening a second revision of "latest".
    Lookup indexes are private to this reader; they are not a second public domain model.
    """

    def __init__(
        self, workspace_id: str, *, at: GitOid
    ) -> None:
        self.repository = StudyRepository(workspace_id)
        self.commit_id = self.repository.resolve(at=at)
        self.store = ArtifactStore(workspace_id)
        self.workspace_id = workspace_id
        self.selected = cache(self._selected)

    @cached_property
    def state(self) -> StudyState:
        from nof1_causal_lab.study.state import apply_effects

        checkpoint = self.repository.state(self.commit_id)
        if not self.records:
            return checkpoint
        attempt = self.records[-1].record.attempt
        if attempt.request is None:
            return checkpoint
        inputs = self.repository.input_state(attempt.request)
        context = checkpoint.with_artifacts(tuple(inputs.current.values()))
        if not isinstance(attempt.outcome, Applied):
            raise RuntimeError("A result reader requires an applied call")
        return apply_effects(context, attempt.outcome.effects.produced, attempt.outcome.effects.retracted)

    @cached_property
    def records(self) -> list[StudyRevision]:
        return self.repository.records(self.commit_id)

    @cached_property
    def seq(self) -> int:
        return self.records[-1].record.seq if self.records else 0

    def _selected(self, artifact_id: ArtifactId) -> object:
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
                            **artifact_file_spec(artifact_id).parquet_files,
                            **artifact_file_spec(artifact_id).json_files,
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
    def question(self) -> QuestionSpec | None:
        return (
            cast("QuestionSpec", self.selected("question")) if self.state.has("question") else None
        )

    @cached_property
    def model(self) -> ModelSpec | None:
        return cast("ModelSpec", self.selected("model")) if self.state.has("model") else None

    def scoped(self, model: ModelSpec) -> StructuralSelection:
        """The study question's outcome scopes any of its models' execution."""
        assert self.question is not None, "set_question roots every lineage"
        return StructuralSelection.for_question(model, self.question)

    @cached_property
    def selection(self) -> StructuralSelection | None:
        return self.scoped(self.model) if self.model is not None else None

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
    def _construct_ids(self) -> frozenset[ConstructId]:
        return frozenset(item.id for item in self.constructs())

    @cached_property
    def _indicator_ids(self) -> frozenset[IndicatorId]:
        return frozenset(item.observation.id for item in self.indicators())

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
        table = self.store.read_parquet_table(
            "raw_data", self.state.current["raw_data"].revision, parquet_filename("raw_data", "raw")
        )
        frame = pl.DataFrame(table)

        dates: list[str] = []
        for candidate in ("timestamp", "date", "time", "datetime"):
            if candidate not in frame.columns:
                continue
            for value in frame[candidate].drop_nulls():
                if isinstance(value, (date, datetime)):
                    dates.append(value.isoformat()[:10])
                elif isinstance(value, str):
                    dates.append(datetime.fromisoformat(value).date().isoformat())
            if dates:
                break
        from nof1_causal_lab.study.view_models import RawDataDateRange

        return self.fact(
            raw_data_view(
                table,
                RawDataDateRange(start=min(dates), end=max(dates)) if dates else None,
            ),
            "raw_data",
            "",
        )

    @cached_property
    def measurements(self) -> Sourced[MeasurementsData] | None:
        if self._panel is None:
            return None
        return self.fact(
            measurements_view(
                self._panel,
                set(self._panel["indicator_id"].to_list()),
                observation_sample(self._panel),
            ),
            "panel",
            "",
        )

    @cached_property
    def data_metadata(self) -> Sourced[PreparedDataMetadata] | None:
        if not self.state.has("panel"):
            return None
        from nof1_causal_lab.study.lineage import read_data_metadata

        return Sourced(
            value=read_data_metadata(self.store, self.state.current["panel"].revision),
            source=self.source("panel", "", filename="metadata.json"),
        )

    @cached_property
    def checks(self) -> tuple[ModelCheckReport, IdentificationReport, ValidationReportArtifact | None] | None:
        from nof1_causal_lab.actions.model_checks import read_model_checks

        if not self.state.has("model"):
            return None
        action = "fit" if self.records and self.records[-1].record.attempt.action == "fit" else "edit_model"
        return read_model_checks(self.workspace_id, self.state, action=action)

    @cached_property
    def data_profile(self) -> Sourced[DataProfileArtifact] | None:
        if not self.state.has("panel"):
            return None
        from nof1_causal_lab.actions.data_checks import read_data_profile

        return self.fact(read_data_profile(self.store, self.state.current["panel"].revision), "panel", "")

    @cached_property
    def validation_report(self) -> Sourced[ValidationReportArtifact] | None:
        if self.checks is None or self.checks[2] is None:
            return None
        return self.fact(self.checks[2].for_indicators(frozenset(self._indicator_ids)), "panel", "")

    @cached_property
    def inference_report(self) -> Sourced[InferenceReport] | None:
        from nof1_causal_lab.study.lineage import (
            inference_report_is_current,
            inference_report_record,
        )

        if not self.state.has("model"):
            return None
        record = inference_report_record(
            self.records,
            self.state,
        )
        if record is None:
            return None
        assert record.record.attempt.action == "fit"
        assert isinstance(record.record.attempt.outcome, Applied)
        result = record.record.attempt.outcome.result
        assert result is not None
        from nof1_causal_lab.actions.fit import read_inference_report

        report = read_inference_report(self.store, self.state.current["model"].revision, result.evidence)
        current = inference_report_is_current(result, self.state)
        return Sourced(
            value=report,
            source=FactSource(
                ref=GitRef(
                    workspace_id=self.workspace_id,
                    revision=record.commit_id,
                    path="logs/attempt.json",
                ),
                pointer="/attempt/outcome/result/evidence",
                validity=SourceValidity.FRESH if current else SourceValidity.STALE,
            ),
        )

    def identification(self) -> Sourced[IdentificationReport] | None:
        if self.selection is None:
            return None
        from nof1_causal_lab.actions.model_checks import read_identification

        return self.fact(read_identification(self.store, self.selection), "model", "")

    def dispositions(self) -> Sourced[tuple[StructuralItemDisposition, ...]] | None:
        selection = self.selection
        if (
            selection is None
            or selection.model.measurement_clock is None
            or not selection.model.indicators
        ):
            return None
        owners = self._construct_ids | self._indicator_ids | {item.id for item in self.edges()}
        return self.fact(
            tuple(item for item in selection.structural_dispositions if item.target.id in owners),
            "model",
            "",
        )

    def fit(self) -> Sourced[FitSummary] | None:

        read = self.inference_report
        if read is None:
            return None
        posterior = read.value.core
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
            estimate = findings[0]
            for owner in model.parameter_context(parameter.id).owners:
                if (
                    model.parameter_context(parameter.id).quantity == SiteKind.DYNAMICS_WEIGHT
                    and owner.kind == "edge"
                ):
                    edge_estimates[owner.id] = estimate.subject
                elif (
                    model.parameter_context(parameter.id).quantity == SiteKind.DYNAMICS_DECAY
                    and owner.kind == "construct"
                ):
                    decay_estimates[owner.id] = estimate.subject
        return Sourced(
            value=FitSummary(
                report=posterior,
                edge_estimates=edge_estimates,
                decay_estimates=decay_estimates,
                prior_densities=self.fit_prior_densities(marginals.keys()),
            ),
            source=read.source,
        )

    def fit_prior_densities(self, fitted: Iterable[ParameterId]) -> dict[ParameterId, DensityCurve]:
        """Curves of the input laws the fit conditioned, where it reports posteriors."""
        from nof1_causal_lab.study.lineage import inference_report_record
        from nof1_causal_lab.study.store import read_model

        record = inference_report_record(self.records, self.state)
        assert record is not None
        assert record.record.attempt.action == "fit"
        assert isinstance(record.record.attempt.outcome, Applied)
        result = record.record.attempt.outcome.result
        assert result is not None
        curves = quantity_prior_densities(self.scoped(read_model(self.store, result.model.revision)))
        return {
            identity: curve
            for identity in fitted
            if (curve := curves.get(identity)) is not None and curve.x
        }

    def simulation(self) -> Sourced[SimulationReport] | None:
        """Return the most recent explicit simulation with its own input revisions."""

        for record in reversed(self.records):
            if record.record.attempt.action != "simulate" or not isinstance(
                record.record.attempt.outcome, Applied
            ):
                continue
            from nof1_causal_lab.actions.simulate import read_simulation_report

            evidence = record.record.attempt.outcome.result.evidence
            report = read_simulation_report(self.store, evidence, self.repository.state(record.commit_id).current["question"].revision)
            pins: dict[ArtifactId, GitOid] = {"model": report.evidence.model.revision}
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
                        path="logs/attempt.json",
                    ),
                    pointer="/attempt/outcome/result/evidence",
                    validity=SourceValidity.FRESH if current else SourceValidity.STALE,
                ),
            )
        return None

    def check_finding[T](self, value: T | None) -> Sourced[T] | None:
        """Point derived findings at their supporting scientific input."""
        return self.fact(value, "model", "") if value is not None else None

    def snapshot(self) -> ModelSnapshot:
        """Batch the aggregate reads and server-composed table facts at this revision."""
        from nof1_causal_lab.models.model_structure import model_graph_entities
        from nof1_causal_lab.study.snapshot_models import ModelGraphView

        identification, dispositions = self.identification(), self.dispositions()
        fit = self.fit()
        graph_constructs, graph_edges = (
            model_graph_entities(self.selection) if self.selection else ((), ())
        )
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
        if self.selection is not None:
            from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel, compile_model
            from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
                forward_simulation_supported,
            )

            compiled = compile_model(self.selection)
            can_simulate = isinstance(compiled, CompiledModel) and forward_simulation_supported(
                compiled
            )
        checks = self.checks[0] if self.checks is not None else None
        predictive = self.check_finding(checks.predictive if checks else None)
        return ModelSnapshot(
            question=self.fact(self.question, "question", "") if self.question else None,
            model=self.fact(self.model, "model", "") if self.model else None,
            workspace_id=self.workspace_id,
            commit_id=self.commit_id,
            selected_seq=self.seq,
            state=self.state,
            can_simulate=can_simulate,
            raw_data=self.raw_data,
            measurements=self.measurements,
            metadata=self.data_metadata,
            profile=self.data_profile,
            entity_failures=entity_failures(
                self.model,
                fit,
                predictive,
                identification,
                self.validation_report or self.data_profile,
            ),
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
            confounder_equations=confounder_equations(self.selection)
            if self.selection
            and self.selection.model.measurement_clock is not None
            and self.selection.model.indicators
            else {},
            state_equations=state_equations(self.selection)
            if self.selection
            and self.selection.model.measurement_clock is not None
            and self.selection.model.indicators
            else {},
            observation_equations=observation_equations(self.model) if self.model else {},
            likelihood_diagnostics=likelihood_histograms(
                self.selection,
                self._panel if self.validation_report is not None else None,
            )
            if self.selection
            else {},
            authoring_prior_densities={
                parameter.id: prior_density(law)
                for parameter in self.parameters()
                if self.model
                and (law := self.model.distribution_for(parameter.id)) is not None
                and distribution_shape(law) == ((), ())
            },
            fit=fit,
            simulation=self.simulation(),
            specification=self.check_finding(checks.specification if checks else None),
            question_checks=self.check_finding(checks.question if checks else None),
            predictive=predictive,
        )

    def observation_history(self, indicator_id: IndicatorId) -> ObservationHistory | None:
        metadata = self.data_metadata
        if metadata is None:
            return None
        variable = next((v for v in metadata.value.variables if v.id == indicator_id), None)
        if variable is None:
            return None
        panel = (
            self.store.read_parquet_file(
                "panel", self.state.current["panel"].revision, "panel.parquet"
            )
            .filter(pl.col("indicator_id") == indicator_id)
            .sort("anchor_time")
        )
        from nof1_causal_lab.study.visuals import observation_history

        return observation_history(metadata.value, variable, panel)

    def predictive_history(self, indicator_id: IndicatorId) -> PPCOverlay | None:
        """Return the saved overlay with its producer-owned schedule and scale."""
        check = self.checks[0].predictive if self.checks is not None else None
        if (
            check is None
            or check.evaluation.kind != "evaluated"
            or check.evaluation.predictive_checks is None
        ):
            return None
        return next(
            (
                item
                for item in check.evaluation.predictive_checks.overlays
                if item.indicator_id == indicator_id
            ),
            None,
        )

    def simulation_paths(self, *, start: int, count: int) -> SimulationPaths | None:
        """Read a contiguous page of original paired paths."""
        from nof1_causal_lab.study.visuals import recorded_simulation_paths

        saved = self.simulation()
        if saved is None:
            return None
        report = saved.value
        if start >= report.evidence.draws:
            raise StudyLookupError("Draw page starts past the saved simulation")
        from nof1_causal_lab.study.store import read_model

        observations = self.store.read_array(report.evidence.observations)
        mask = self.store.read_array(report.evidence.observation_layout.mask)
        reference = (
            self.store.read_array(report.evidence.reference_latent_paths)
            if report.evidence.reference_latent_paths is not None
            else None
        )
        reference_observations = (
            self.store.read_array(report.evidence.reference_observations)
            if report.evidence.reference_observations is not None
            else None
        )
        # Hydrate categorical emissions against their declared codebook at the storage edge.
        for index, variable in enumerate(report.evidence.observation_layout.variables):
            levels = (
                ("0", "1")
                if variable.measurement_dtype == "binary"
                else variable.ordinal_levels or variable.categorical_levels
            )
            if levels is None:
                continue
            for buffer in (observations, reference_observations):
                if buffer is None:
                    continue
                channel = buffer[:, :, index]
                codes = channel[mask[:, :, index] & np.isfinite(channel)]
                if not np.all((codes == np.floor(codes)) & (codes >= 0) & (codes < len(levels))):
                    raise StudyLookupError(
                        "Saved simulation emissions differ from their declared category codes"
                    )
        return recorded_simulation_paths(
            report,
            read_model(self.store, report.evidence.model.revision),
            self.store.read_array(report.evidence.latent_paths),
            observations,
            mask,
            reference,
            reference_observations,
            start=start,
            count=count,
        )

    def parameter_draws(self) -> ParameterDraws:
        """Read atoms by their production coordinates and labels, without compiling."""
        from nof1_causal_lab.numpyro_json import empirical_atoms
        from nof1_causal_lab.study.lineage import law_provenance
        from nof1_causal_lab.study.visuals import empirical_points

        model = self.model
        if model is None:
            return Unavailable(reason="No model at this revision.")
        provenance = law_provenance(self.store, self.state.current["model"], model, None)
        if provenance.kind != "fitted":
            return Unavailable(reason="This revision has no complete retained joint posterior.")
        columns = []
        for identity, layout in sorted(model.law_layouts.items()):
            atoms = empirical_atoms(model.distributions[identity])
            for parameter, elements in layout.parameters:
                for element in elements:
                    values = atoms[:, layout.parameter_columns[element]]
                    columns.append(ParameterDrawColumn(
                        label=layout.labels[element],
                        subject=ParameterRef(parameter_id=parameter, element_id=element),
                        values=tuple(float(value) for value in values),
                        empirical=empirical_points(values),
                    ))
        return Available[tuple[ParameterDrawColumn, ...]](value=tuple(columns))


def quantity_prior_densities(
    selection: StructuralSelection,
) -> dict[ParameterId, DensityCurve]:
    """Resolve native quantity laws at the reader boundary before projecting curves."""
    from nof1_causal_lab.models.ssm.compile.prior_compilation import quantity_parameter_law
    from nof1_causal_lab.numpyro_json import distribution_shape
    from nof1_causal_lab.study.prior_views import prior_density

    model = selection.model
    return {
        parameter.id: prior_density(quantity_parameter_law(model, parameter)[0])
        for parameter in execution_parameters(selection)
        if parameter.distribution is not None
        and not any(distribution_shape(model.distributions[parameter.distribution]))
    }
