"""Revision-pinned access to canonical aggregates and their compatible findings."""

from __future__ import annotations

from functools import cache, cached_property
from typing import TYPE_CHECKING, cast

from nof1_causal_lab.actions.io import EditModelOutput, FitOutput, PrepareDataOutput, SimulateOutput
from nof1_causal_lab.artifacts.availability import Unavailable
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, StudyRevision
from nof1_causal_lab.study.snapshot_models import FitSummary, ModelGraphView, ModelSnapshot
from nof1_causal_lab.study.store import ArtifactStore, read_payload

if TYPE_CHECKING:
    import polars as pl

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
    )
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.artifacts.posterior import InferenceReport
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
    from nof1_causal_lab.study.visual_models import (
        ObservationHistory,
        ParameterDraws,
        PathSeries,
        SimulationPaths,
    )


class ModelReader:
    """Compose saved action results through their pinned scientific inputs."""

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
    def fit_result(self) -> FitOutput | None:
        """The saved fit that produced the selected model."""
        from nof1_causal_lab.study.lineage import inference_report_record

        record = inference_report_record(self.records, self.state)
        if record is None:
            return None
        assert isinstance(record.record.attempt.outcome, Applied)
        return self.store.read_report(record.record.attempt.outcome.result, FitOutput)

    @cached_property
    def fit_for_selected_data(self) -> FitOutput | None:
        """Select model/data findings only while their conditioning history is selected."""
        from nof1_causal_lab.artifacts.data_ref import DataRef
        from nof1_causal_lab.study.lineage import inference_report_record

        record = inference_report_record(self.records, self.state)
        if record is None:
            return None
        attempt = record.record.attempt
        assert attempt.action == "fit"
        assert attempt.request is not None
        selected = DataRef[GitOid, int](
            revision=attempt.request.input.data_ref,
            replicate_index=attempt.request.input.replicate_index,
        )
        return self.fit_result if selected == self.state.data else None

    def model_output(self) -> EditModelOutput | None:
        """Follow model inputs to the edit that owns structural and authoring findings."""
        selected = self.state.get("model")
        while selected is not None:
            record = next(
                (
                    item
                    for item in reversed(self.records)
                    if item.record.attempt.action == "edit_model"
                    and isinstance(item.record.attempt.outcome, Applied)
                    and any(
                        artifact.revision == selected.revision
                        for artifact in item.record.attempt.outcome.effects.produced
                    )
                ),
                None,
            )
            if record is not None:
                assert isinstance(record.record.attempt.outcome, Applied)
                return self.store.read_report(record.record.attempt.outcome.result, EditModelOutput)
            parent = selected.derived_from.get("model")
            selected = self.store.read_meta("model", parent) if parent is not None else None
        return None

    @cached_property
    def prepared_result(self) -> PrepareDataOutput | None:
        """The preparation result owning the selected panel, with no lookup in unrelated calls."""
        if self.state.data is None:
            return None
        from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
        from nof1_causal_lab.study.data import panel_revision, read_data_source

        if not isinstance(
            read_data_source(self.store, self.state.data.revision), PreparedDataMetadata
        ):
            return None
        panel = panel_revision(self.store, self.state.data.revision)
        record = next(
            (
                item
                for item in reversed(self.records)
                if item.record.attempt.action == "prepare_data"
                and isinstance(item.record.attempt.outcome, Applied)
                and any(
                    info.artifact_id == "panel" and info.revision == panel
                    for info in item.record.attempt.outcome.effects.produced
                )
            ),
            None,
        )
        if record is None:
            return None
        assert isinstance(record.record.attempt.outcome, Applied)
        return self.store.read_report(record.record.attempt.outcome.result, PrepareDataOutput)

    @property
    def raw_data(self) -> RawDataData | None:
        """Uploaded-table summary retained by the selected preparation."""
        return self.prepared_result.raw_data if self.prepared_result is not None else None

    @property
    def measurements(self) -> MeasurementsData | None:
        """Observation counts retained by the selected preparation."""
        return self.prepared_result.measurements if self.prepared_result is not None else None

    @property
    def data_metadata(self) -> PreparedDataMetadata | None:
        """Preparation recipe and calendar coordinates retained with its result."""
        return self.prepared_result.metadata if self.prepared_result is not None else None

    @property
    def data_profile(self) -> DataProfileArtifact | None:
        """Data-quality findings owned by the selected preparation."""
        return self.prepared_result.profile if self.prepared_result is not None else None

    @property
    def inference_report(self) -> InferenceReport | None:
        """The complete diagnostic report saved by the owning fit."""
        return self.fit_result.inference_report if self.fit_result is not None else None

    @property
    def validation_report(self) -> ValidationReportArtifact | None:
        """Model/data findings retained by the owning fit."""
        return (
            self.fit_for_selected_data.validation_report
            if self.fit_for_selected_data is not None
            else None
        )

    def identification(self) -> IdentificationReport | None:
        """Identification findings retained by the model edit."""
        result = self.model_output()
        return result.identification if result is not None else None

    def dispositions(self) -> tuple[StructuralItemDisposition, ...] | None:
        """Structural dispositions retained by the model edit."""
        result = self.model_output()
        return result.dispositions if result is not None else None

    def fit(self) -> FitSummary | None:
        """Select stored report and display values without recomputing summaries."""
        result = self.fit_result
        if result is None or result.inference_report is None:
            return None
        return FitSummary(
            report=result.inference_report.core,
            edge_estimates=result.edge_estimates,
            decay_estimates=result.decay_estimates,
            prior_densities=result.prior_densities,
        )

    @cached_property
    def simulation_result(self) -> SimulateOutput | None:
        """The latest explicit simulation in the selected action history."""
        for record in reversed(self.records):
            attempt = record.record.attempt
            if attempt.action == "simulate" and isinstance(attempt.outcome, Applied):
                return self.store.read_report(attempt.outcome.result, SimulateOutput)
        return None

    def simulation(self) -> SimulationReport | None:
        """Findings saved by the selected simulation."""
        return self.simulation_result.report if self.simulation_result is not None else None

    def observation_history(self, indicator_id: IndicatorId) -> ObservationHistory | None:
        """Select one saved observation series from the exact input history."""
        if self.state.data is None:
            return None
        if self.prepared_result is not None:
            return self.prepared_result.data.get(indicator_id)
        attempt = self.repository.record(self.state.data.revision).record.attempt
        assert attempt.action == "simulate"
        assert isinstance(attempt.outcome, Applied)
        result = self.store.read_report(attempt.outcome.result, SimulateOutput)
        return result.data[self.state.data.replicate_index].get(indicator_id)

    def simulation_paths(self, *, start: int, count: int) -> SimulationPaths | None:
        """Select a page of already stored trajectories without recomputing their findings."""
        result = self.simulation_result
        if result is None or result.paths is None:
            return None
        paths = result.paths
        if start >= paths.total_draws:
            from nof1_causal_lab.study.errors import StudyLookupError

            raise StudyLookupError("Draw page starts past the saved simulation")
        stop = min(start + count, paths.total_draws)

        def page(series: PathSeries) -> PathSeries:
            return series.revised(
                action=series.action[start:stop], reference=series.reference[start:stop]
            )

        return paths.revised(
            start=start,
            count=stop - start,
            states={key: page(value) for key, value in paths.states.items()},
            indicators={key: page(value) for key, value in paths.indicators.items()},
            effect=page(paths.effect) if paths.effect is not None else None,
        )

    def parameter_draws(self) -> ParameterDraws:
        """The parameter coordinates and draws saved by the fit."""
        return (
            self.fit_result.parameter_draws
            if self.fit_result is not None
            else Unavailable(reason="No fit at this revision.")
        )

    def snapshot(self) -> ModelSnapshot:
        """Compose stored values by their action ownership for the workbench."""
        model = self.model_output()
        fit = self.fit_for_selected_data
        return ModelSnapshot(
            workspace_id=self.workspace_id,
            commit_id=self.commit_id,
            selected_seq=self.seq,
            state=self.state,
            question=self.question,
            model=self.model,
            raw_data=self.raw_data,
            measurements=self.measurements,
            metadata=self.data_metadata,
            profile=self.data_profile,
            can_simulate=model.can_simulate if model else False,
            identification=model.identification if model else None,
            dispositions=model.dispositions if model else None,
            graph=model.graph if model else ModelGraphView(),
            entity_failures={
                identity: (
                    *((model.entity_failures if model else {}).get(identity, ())),
                    *((fit.entity_failures if fit else {}).get(identity, ())),
                )
                for identity in (
                    (model.entity_failures.keys() if model else set())
                    | (fit.entity_failures.keys() if fit else set())
                )
            },
            validation_report=self.validation_report,
            confounder_equations=model.confounder_equations if model else {},
            state_equations=model.state_equations if model else {},
            observation_equations=model.observation_equations if model else {},
            likelihood_diagnostics=fit.likelihood_diagnostics if fit else {},
            authoring_prior_densities=model.authoring_prior_densities if model else {},
            fit=self.fit(),
            specification=model.checks.specification if model and model.checks else None,
            question_checks=fit.question_checks
            if fit
            else model.checks.question
            if model and model.checks
            else None,
            simulation=self.simulation(),
        )
