"""Revision-pinned access to canonical aggregates and their compatible findings."""

from __future__ import annotations

from datetime import date, datetime
from functools import cache, cached_property
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import polars as pl

from nof1_causal_lab.actions.io import EditModelOutput
from nof1_causal_lab.artifacts.availability import Available, Unavailable
from nof1_causal_lab.artifacts.identity import GitOid, ParameterRef
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.model_parameters import execution_parameters
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.numpyro_json import distribution_shape
from nof1_causal_lab.study.artifact_files import parquet_filename
from nof1_causal_lab.study.equations import (
    confounder_equations,
    observation_equations,
    state_equations,
)
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.prior_views import prior_density
from nof1_causal_lab.study.records import Applied, StudyRevision
from nof1_causal_lab.study.snapshot_models import FitSummary, ModelSnapshot
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
    from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.artifacts.posterior import InferenceReport
    from nof1_causal_lab.artifacts.posterior_diagnostics import DensityCurve, PPCOverlay
    from nof1_causal_lab.artifacts.question import QuestionSpec
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.artifacts.validation_report import (
        DataProfileArtifact,
        ValidationReportArtifact,
    )
    from nof1_causal_lab.study.data import DataHistory
    from nof1_causal_lab.study.state import StudyState
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

    def __init__(self, workspace_id: str, *, at: GitOid, state: StudyState | None = None) -> None:
        """Pin reads to a successful study commit and optionally an explicitly composed input state."""
        self.repository = StudyRepository(workspace_id)
        self.commit_id = self.repository.resolve(at=at)
        self.store = ArtifactStore(workspace_id)
        self.workspace_id = workspace_id
        self._state = state
        self.selected = cache(self._selected)

    @cached_property
    def state(self) -> StudyState:
        """Artifact and data selections reconstructed for the pinned action, including its input context."""
        from nof1_causal_lab.study.state import apply_effects

        if self._state is not None:
            return self._state
        checkpoint = self.repository.state(self.commit_id)
        if not self.records:
            return checkpoint
        attempt = self.records[-1].record.attempt
        if attempt.request is None:
            return checkpoint
        inputs = self.repository.input_state(attempt.request)
        context = checkpoint.with_artifacts(tuple(inputs.current.values()))
        if attempt.action in {"fit", "edit_model"}:
            context = context.revised(data=inputs.data)
        if not isinstance(attempt.outcome, Applied):
            raise RuntimeError("A result reader requires an applied call")
        return apply_effects(
            context, attempt.outcome.effects.produced, attempt.outcome.effects.retracted
        )

    @cached_property
    def records(self) -> list[StudyRevision]:
        """Retained journal records reachable from the pinned study commit."""
        return self.repository.records(self.commit_id)

    @cached_property
    def seq(self) -> int:
        """Sequence number at the selected checkpoint, or zero at the study root."""
        return self.records[-1].record.seq if self.records else 0

    def _selected(self, artifact_id: ArtifactId) -> object:
        return read_payload(
            self.store,
            artifact_id,
            self.state.current[artifact_id].revision,
        )

    @cached_property
    def question(self) -> QuestionSpec | None:
        """Selected authored study question, or ``None`` before a question is available."""
        return (
            cast("QuestionSpec", self.selected("question")) if self.state.has("question") else None
        )

    @cached_property
    def model(self) -> ModelSpec | None:
        """Selected scientific model definition, or ``None`` before a model is available."""
        return cast("ModelSpec", self.selected("model")) if self.state.has("model") else None

    def scoped(self, model: ModelSpec) -> StructuralSelection:
        """The study question's outcome scopes any of its models' execution."""
        assert self.question is not None, "edit_question roots every lineage"
        return StructuralSelection.for_question(model, self.question)

    @cached_property
    def selection(self) -> StructuralSelection | None:
        """Selected model scoped to the study question's outcome, when a model is available."""
        return self.scoped(self.model) if self.model is not None else None

    def constructs(self) -> tuple[ConstructSpec, ...]:
        """Return the selected model's constructs, or an empty tuple without a model."""
        return self.model.constructs if self.model else ()

    def edges(self) -> tuple[CausalEdgeSpec, ...]:
        """Return the selected model's causal edges, or an empty tuple without a model."""
        return self.model.edges if self.model else ()

    def indicators(self) -> tuple[IndicatorSpec, ...]:
        """Return the selected model's observation indicators, or an empty tuple without a model."""
        return self.model.indicators if self.model else ()

    def parameters(self, owner: EntityRef | None = None) -> tuple[ParameterSpec, ...]:
        """Return model parameters, optionally restricted to those owned by a scientific entity."""
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
    def data_history(self) -> DataHistory | None:
        """Exact observation history selected for this read, or ``None`` without a data selection."""
        from nof1_causal_lab.study.data import read_data_history

        return (
            read_data_history(self.store, self.state.data) if self.state.data is not None else None
        )

    @cached_property
    def _panel(self) -> pl.DataFrame | None:
        return (
            self.data_history.observations.recorded.frame if self.data_history is not None else None
        )

    @cached_property
    def raw_data(self) -> RawDataData | None:
        """Source table dimensions, sample rows, and available date bounds from the selected artifact."""
        if not self.state.has("raw_data") or self.data_metadata is None:
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

        return raw_data_view(
            table,
            RawDataDateRange(start=min(dates), end=max(dates)) if dates else None,
        )

    @cached_property
    def measurements(self) -> MeasurementsData | None:
        """Selected panel's observation counts and sample rows, or ``None`` without a panel."""
        if self._panel is None:
            return None
        return measurements_view(
            self._panel,
            set(self._panel["indicator_id"].to_list()),
            observation_sample(self._panel),
        )

    @cached_property
    def data_metadata(self) -> PreparedDataMetadata | None:
        """Preparation recipe and provenance for uploaded data, absent for histories without metadata."""
        history = self.data_history
        if history is None or history.metadata is None:
            return None
        return history.metadata

    @cached_property
    def _model_check_record(self) -> StudyRevision | None:
        return self._report_record("model")

    def _report_record(self, artifact_id: ArtifactId) -> StudyRevision | None:
        selected = self.state.get(artifact_id)
        if selected is None:
            return None
        return next(
            (
                record
                for record in reversed(self.records)
                if isinstance(record.record.attempt.outcome, Applied)
                and any(
                    item.artifact_id == artifact_id and item.revision == selected.revision
                    for item in record.record.attempt.outcome.effects.produced
                )
            ),
            None,
        )

    @cached_property
    def checks(
        self,
    ) -> tuple[ModelCheckReport, IdentificationReport, ValidationReportArtifact | None] | None:
        """Retained model, identification, and optional data-validation reports for the model producer."""
        from nof1_causal_lab.artifacts.identification import IdentificationReport
        from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
        from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact

        record = self._model_check_record
        if record is None:
            return None
        checks = self.repository.read_report(record.commit_id, "checks", ModelCheckReport)
        if checks is None:
            return None
        identification = self.repository.read_report(
            record.commit_id, "identification", IdentificationReport
        )
        assert identification is not None, "Model checks must retain their identification report"
        validation = self.repository.read_report(
            record.commit_id, "validation", ValidationReportArtifact
        )
        return checks, identification, validation

    @cached_property
    def data_profile(self) -> DataProfileArtifact | None:
        """Newest retained empirical profile matching the exact selected observation history."""
        if self.data_history is None:
            return None
        from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact

        for record in reversed(self.records):
            profile = self.repository.read_report(
                record.commit_id, "data-profile", DataProfileArtifact
            )
            if profile is None:
                continue
            attempt = record.record.attempt
            if (
                attempt.action == "fit"
                and isinstance(attempt.outcome, Applied)
                and attempt.outcome.result is not None
            ):
                matches = attempt.outcome.result.data == self.state.data
            elif attempt.action == "prepare_data" and isinstance(attempt.outcome, Applied):
                from nof1_causal_lab.study.data import panel_revision

                matches = self.data_history.metadata is not None and any(
                    info.artifact_id == "panel"
                    and info.revision
                    == panel_revision(self.store, self.data_history.source.revision)
                    for info in attempt.outcome.effects.produced
                )
            else:
                matches = False
            if matches:
                return profile
        return None

    @cached_property
    def validation_report(self) -> ValidationReportArtifact | None:
        """Model-data validation for the selected history, restricted to the model's indicators."""
        if (
            self.checks is None
            or self.checks[2] is None
            or self.checks[0].question is None
            or self.state.data is None
            or self.checks[0].question.data != self.state.data
        ):
            return None
        return self.checks[2].for_indicators(frozenset(self._indicator_ids))

    @cached_property
    def inference_report(self) -> InferenceReport | None:
        """Retained inference report for the model owned by its recorded fit."""
        from nof1_causal_lab.study.lineage import (
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
        from nof1_causal_lab.artifacts.posterior import InferenceReport

        report = self.repository.read_report(record.commit_id, "inference", InferenceReport)
        if report is None:
            return None
        return report

    def identification(self) -> IdentificationReport | None:
        """Return model identification findings, or ``None`` without retained checks."""
        if self.checks is None:
            return None
        return self.checks[1]

    def dispositions(self) -> tuple[StructuralItemDisposition, ...] | None:
        """Return execution dispositions for model-owned entities when measurement structure is available."""
        selection = self.selection
        if (
            selection is None
            or selection.model.measurement_clock is None
            or not selection.model.indicators
        ):
            return None
        owners = self._construct_ids | self._indicator_ids | {item.id for item in self.edges()}
        return tuple(item for item in selection.structural_dispositions if item.target.id in owners)

    def fit(self) -> FitSummary | None:
        """Compose fit summaries and quantity-scale prior curves from the retained inference report."""
        read = self.inference_report
        if read is None:
            return None
        posterior = read.core
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
        return FitSummary(
            report=posterior,
            edge_estimates=edge_estimates,
            decay_estimates=decay_estimates,
            prior_densities=self.fit_prior_densities(marginals.keys()),
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
        curves = quantity_prior_densities(
            self.scoped(read_model(self.store, result.model.revision))
        )
        return {
            identity: curve
            for identity in fitted
            if (curve := curves.get(identity)) is not None and curve.x
        }

    def simulation(self) -> SimulationReport | None:
        """Return the most recent explicit simulation with its own input revisions."""
        for record in reversed(self.records):
            if record.record.attempt.action != "simulate" or not isinstance(
                record.record.attempt.outcome, Applied
            ):
                continue
            from nof1_causal_lab.artifacts.simulation import SimulationReport

            report = self.repository.read_report(record.commit_id, "simulation", SimulationReport)
            if report is None:
                return None
            return report
        return None

    def model_output(self, fit: FitSummary | None = None) -> EditModelOutput:
        """Project only the produced model and its own checks, without loading other actions."""
        from nof1_causal_lab.models.model_structure import model_graph_entities
        from nof1_causal_lab.study.snapshot_models import ModelGraphView

        identification, dispositions = self.identification(), self.dispositions()
        model = self.model
        assert model is not None, "A model producer must retain its model"
        graph_constructs, graph_edges = (
            model_graph_entities(self.selection) if self.selection else ((), ())
        )
        blocking = set()
        if identification:
            for cid, finding in identification.non_identifiable.items():
                blocking.update([cid, *finding.confounders])
        disposition_by_id = {item.target.id: item for item in dispositions} if dispositions else {}
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
        predictive = checks.predictive if checks else None
        return EditModelOutput(
            model=model,
            checks=checks,
            predictive_overlays={
                indicator.observation.id: overlay
                for indicator in self.indicators()
                if (overlay := self.predictive_history(indicator.observation.id)) is not None
            },
            can_simulate=can_simulate,
            entity_failures=entity_failures(
                self.model,
                fit,
                predictive,
                identification,
                self.validation_report,
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
            specification=checks.specification if checks else None,
            question_checks=checks.question if checks else None,
            predictive=predictive,
        )

    def snapshot(self) -> ModelSnapshot:
        """Compose the retained aggregate for local readers and fixture exports."""
        from nof1_causal_lab.study.snapshot_models import ModelGraphView

        fit = self.fit()
        model = self.model_output(fit) if self.model is not None else None
        return ModelSnapshot(
            question=self.question if self.question else None,
            model=model.model if model is not None else None,
            workspace_id=self.workspace_id,
            commit_id=self.commit_id,
            selected_seq=self.seq,
            state=self.state,
            raw_data=self.raw_data,
            measurements=self.measurements,
            metadata=self.data_metadata,
            profile=self.data_profile,
            fit=fit,
            simulation=self.simulation(),
            can_simulate=model.can_simulate if model is not None else False,
            identification=model.identification if model is not None else None,
            dispositions=model.dispositions if model is not None else None,
            graph=model.graph if model is not None else ModelGraphView(),
            entity_failures=model.entity_failures if model is not None else {},
            validation_report=model.validation_report if model is not None else None,
            confounder_equations=model.confounder_equations if model is not None else {},
            state_equations=model.state_equations if model is not None else {},
            observation_equations=model.observation_equations if model is not None else {},
            likelihood_diagnostics=model.likelihood_diagnostics if model is not None else {},
            authoring_prior_densities=model.authoring_prior_densities if model is not None else {},
            specification=model.specification if model is not None else None,
            question_checks=model.question_checks if model is not None else None,
            predictive=model.predictive if model is not None else None,
        )

    def observation_history(self, indicator_id: IndicatorId) -> ObservationHistory | None:
        """Read one indicator's complete, time-ordered observations and measurement supports."""
        history = self.data_history
        if history is None:
            return None
        variable = next((v for v in history.variables if v.id == indicator_id), None)
        if variable is None:
            return None
        panel = history.observations.recorded.frame.filter(
            pl.col("indicator_id") == indicator_id
        ).sort("anchor_time")
        from nof1_causal_lab.study.visuals import observation_history

        return observation_history(history.time_origin, variable, panel)

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
        report = saved
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
                    columns.append(
                        ParameterDrawColumn(
                            label=layout.labels[element],
                            subject=ParameterRef(parameter_id=parameter, element_id=element),
                            values=tuple(float(value) for value in values),
                            empirical=empirical_points(values),
                        )
                    )
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
