"""Complete action-owned scientific results before atomic publication."""

from __future__ import annotations

from datetime import date, datetime
from functools import cache, cached_property
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import polars as pl

from nof1_causal_lab.actions.io import (
    DataDiffOutput,
    EditModelOutput,
    EditQuestionOutput,
    FitOutput,
    ModelDiffOutput,
    PrepareDataOutput,
    SimulateOutput,
)
from nof1_causal_lab.artifacts.arrays import ArrayVector
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
from nof1_causal_lab.study.records import ActionAttempt, Applied, StagedActionAttempt
from nof1_causal_lab.study.snapshot_models import FitSummary
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
    PathSeries,
    SimulationPaths,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nof1_causal_lab.actions.effects import ActionEffects
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
    from nof1_causal_lab.artifacts.posterior_diagnostics import DensityCurve, TraceSeries
    from nof1_causal_lab.artifacts.question import QuestionSpec
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.artifacts.validation_report import (
        DataProfileArtifact,
        ValidationReportArtifact,
    )
    from nof1_causal_lab.study.data import DataHistory
    from nof1_causal_lab.study.results import ActionOutput
    from nof1_causal_lab.study.view_models import (
        MeasurementsData,
        RawDataData,
    )


from nof1_causal_lab.artifacts.data_ref import DataRef
from nof1_causal_lab.artifacts.posterior import ModelFitResult
from nof1_causal_lab.study.state import StudyState, apply_effects


class OutputBuilder:
    """Read one execution's inputs and computed findings to construct its final result."""

    def __init__(self, workspace_id: str, attempt: StagedActionAttempt, state: StudyState) -> None:
        """Bind only the current action's staged artifacts and reports."""
        self.repository = StudyRepository(workspace_id)
        self.store = ArtifactStore(workspace_id)
        self.workspace_id = workspace_id
        assert isinstance(attempt.outcome, Applied)
        self.applied = attempt.outcome
        self.state = state
        self.selected = cache(self._selected)

    @cached_property
    def checks(
        self,
    ) -> tuple[ModelCheckReport, IdentificationReport, ValidationReportArtifact | None] | None:
        """Checks computed by this action, without searching another call's reports."""
        from nof1_causal_lab.artifacts.identification import IdentificationReport
        from nof1_causal_lab.artifacts.model_checks import ModelCheckReport
        from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact

        reports = self.applied.effects.reports
        if "checks" not in reports:
            return None
        return (
            self.store.read_report(reports["checks"], ModelCheckReport),
            self.store.read_report(reports["identification"], IdentificationReport),
            self.store.read_report(reports["validation"], ValidationReportArtifact)
            if "validation" in reports
            else None,
        )

    @cached_property
    def data_profile(self) -> DataProfileArtifact | None:
        """Data profile owned by this preparation action."""
        from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact

        ref = self.applied.effects.reports.get("data-profile")
        return self.store.read_report(ref, DataProfileArtifact) if ref is not None else None

    @cached_property
    def inference_report(self) -> InferenceReport | None:
        """The report computed by this fit's numerical execution."""
        from nof1_causal_lab.artifacts.posterior import InferenceReport

        ref = self.applied.effects.reports.get("inference")
        if ref is None:
            return None
        report = self.store.read_report(ref, InferenceReport)
        result = self.applied.result
        assert isinstance(result, ModelFitResult)
        from nof1_causal_lab.numpyro_json import empirical_atoms

        model = self.model
        assert model is not None
        evidence = result.evidence
        chains = evidence.num_chains

        def trace(value: TraceSeries) -> TraceSeries:
            layout = model.law_layouts[evidence.distribution]
            atoms = empirical_atoms(model.distributions[evidence.distribution])
            atoms_ref = self.store.write_array(atoms)
            assert chains is not None
            size = len(atoms) // chains
            return value.revised(
                chains=tuple(
                    ArrayVector(
                        array_ref=atoms_ref,
                        indices=(None, layout.parameter_columns[value.subject.element_id]),
                        start=chain * size,
                        stop=(chain + 1) * size,
                    )
                    for chain in range(chains)
                )
            )

        def rows(ref: str | None) -> tuple[ArrayVector, ...] | None:
            return (
                tuple(
                    ArrayVector(array_ref=ref, indices=(row, None))
                    for row in range(self.store.read_array(ref).shape[0])
                )
                if ref is not None
                else None
            )

        return report.revised(
            detail=report.detail.revised(
                trace_data=tuple(trace(value) for value in report.detail.trace_data),
                divergent=evidence.chain_extra_fields.get("diverging"),
                initial_latent_delta=rows(evidence.initial_latent_delta),
                final_latent_delta=rows(evidence.final_latent_delta),
            )
        )

    def fit_prior_densities(self, fitted: Iterable[ParameterId]) -> dict[ParameterId, DensityCurve]:
        """Input laws on the quantity scale used for this fit's parameter summaries."""
        from nof1_causal_lab.study.store import read_model

        result = self.applied.result
        assert isinstance(result, ModelFitResult)
        parent = result.model.revision
        curves = quantity_prior_densities(self.scoped(read_model(self.store, parent)))
        return {
            identity: curve
            for identity in fitted
            if (curve := curves.get(identity)) is not None and curve.x
        }

    def simulation(self) -> SimulationReport | None:
        """The findings computed for this simulation's retained draws."""
        from nof1_causal_lab.artifacts.simulation import SimulationReport

        ref = self.applied.effects.reports.get("simulation")
        return self.store.read_report(ref, SimulationReport) if ref is not None else None

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

    def model_output(self) -> EditModelOutput:
        """Project only the produced model and its own checks, without loading other actions."""
        from nof1_causal_lab.models.model_structure import model_graph_entities
        from nof1_causal_lab.study.action_arrays import array_references, result_arrays
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
        return EditModelOutput(
            model=model,
            arrays=result_arrays(self.store, array_references(model.model_dump(mode="json"))),
            checks=checks,
            can_simulate=can_simulate,
            entity_failures=entity_failures(
                self.model,
                None,
                None,
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
            authoring_prior_densities={
                parameter.id: prior_density(law)
                for parameter in self.parameters()
                if self.model
                and (law := self.model.distribution_for(parameter.id)) is not None
                and distribution_shape(law) == ((), ())
            },
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
        from nof1_causal_lab.study.visuals import empirical_points

        model = self.model
        if model is None:
            return Unavailable(reason="No model at this revision.")
        columns = []
        for identity, layout in sorted(model.law_layouts.items()):
            atoms = empirical_atoms(model.distributions[identity])
            atoms_ref = self.store.write_array(atoms)
            for parameter, elements in layout.parameters:
                for element in elements:
                    values = atoms[:, layout.parameter_columns[element]]
                    columns.append(
                        ParameterDrawColumn(
                            label=layout.labels[element],
                            subject=ParameterRef(parameter_id=parameter, element_id=element),
                            values=ArrayVector(
                                array_ref=atoms_ref,
                                indices=(None, layout.parameter_columns[element]),
                            ),
                            empirical=empirical_points(values),
                        )
                    )
        return Available[tuple[ParameterDrawColumn, ...]](value=tuple(columns))


def quantity_prior_densities(
    selection: StructuralSelection,
) -> dict[ParameterId, DensityCurve]:
    """Resolve the fit's input quantity laws before retaining their density curves."""
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


def build_output(
    workspace_id: str, attempt: StagedActionAttempt, state: StudyState
) -> ActionOutput:
    """Complete one action's result using only its inputs and staged execution evidence."""
    from nof1_causal_lab.study.action_arrays import array_references, result_arrays
    from nof1_causal_lab.study.visuals import simulation_observation_histories

    assert isinstance(attempt.outcome, Applied)
    store = ArtifactStore(workspace_id)
    effects = attempt.outcome.effects
    if attempt.action == "data_diff":
        return store.read_report(effects.reports["data-diff"], DataDiffOutput)
    if attempt.action == "model_diff":
        return store.read_report(effects.reports["model-diff"], ModelDiffOutput)
    reader = OutputBuilder(workspace_id, attempt, state)
    match attempt.action:
        case "edit_question":
            assert reader.question is not None
            return EditQuestionOutput(question=reader.question)
        case "edit_model":
            return reader.model_output()
        case "prepare_data":
            history = reader.data_history
            return PrepareDataOutput(
                raw_data=reader.raw_data,
                measurements=reader.measurements,
                metadata=reader.data_metadata,
                profile=reader.data_profile,
                data={
                    variable.id: value
                    for variable in (history.variables if history is not None else ())
                    if (value := reader.observation_history(variable.id)) is not None
                },
            )
        case "fit":
            assert reader.model is not None
            result = attempt.outcome.result
            assert result is None or isinstance(result, ModelFitResult)
            summary = reader.fit()
            draws = reader.parameter_draws()
            return FitOutput(
                model=reader.model,
                inference=result,
                entity_failures=entity_failures(
                    reader.model, summary, None, None, reader.validation_report
                ),
                inference_report=reader.inference_report,
                validation_report=reader.validation_report,
                question_checks=reader.checks[0].question if reader.checks else None,
                likelihood_diagnostics=likelihood_histograms(reader.selection, reader._panel)
                if reader.selection is not None
                else {},
                edge_estimates=summary.edge_estimates if summary is not None else {},
                decay_estimates=summary.decay_estimates if summary is not None else {},
                prior_densities=summary.prior_densities if summary is not None else {},
                parameter_draws=draws,
                arrays=result_arrays(
                    store,
                    (
                        *result.evidence.array_references,
                        *array_references(reader.model.model_dump(mode="json")),
                        *array_references(draws.model_dump(mode="json")),
                    ),
                )
                if result is not None
                else {},
            )
        case "simulate":
            report = reader.simulation()
            assert report is not None
            evidence = report.evidence
            references = (
                *evidence.parameter_draws.values(),
                evidence.latent_paths,
                evidence.observations,
                evidence.observation_layout.mask,
                evidence.observation_layout.support_start_times,
                evidence.observation_layout.support_end_times,
                *(
                    (evidence.reference_latent_paths,)
                    if evidence.reference_latent_paths is not None
                    else ()
                ),
                *(
                    (evidence.reference_observations,)
                    if evidence.reference_observations is not None
                    else ()
                ),
            )
            histories = simulation_observation_histories(
                evidence,
                store.read_array(evidence.observations),
                store.read_array(evidence.observation_layout.mask),
                store.read_array(evidence.observation_layout.support_start_times),
                store.read_array(evidence.observation_layout.support_end_times),
            )
            paths = reader.simulation_paths(start=0, count=evidence.draws)
            assert paths is not None

            def vector(ref: str, draw: int, column: int, *, observed: bool = False) -> ArrayVector:
                return ArrayVector(
                    array_ref=ref,
                    indices=(draw, None, column),
                    mask=ArrayVector(
                        array_ref=evidence.observation_layout.mask, indices=(draw, None, column)
                    )
                    if observed
                    else None,
                )

            def series(
                value: PathSeries,
                ref: str,
                reference: str | None,
                column: int,
                *,
                observed: bool = False,
            ) -> PathSeries:
                return value.revised(
                    action=tuple(
                        path.revised(values=vector(ref, path.draw, column, observed=observed))
                        for path in value.action
                    ),
                    reference=tuple(
                        path.revised(values=vector(reference, path.draw, column, observed=observed))
                        for path in value.reference
                    )
                    if reference is not None
                    else (),
                )

            return SimulateOutput(
                report=report,
                data=tuple(
                    {
                        variable.id: history[variable.id].revised(
                            values=vector(evidence.observations, draw, column, observed=True),
                            support_start=ArrayVector(
                                array_ref=evidence.observation_layout.support_start_times,
                                indices=(None, column),
                            ),
                            support_end=ArrayVector(
                                array_ref=evidence.observation_layout.support_end_times,
                                indices=(None, column),
                            ),
                        )
                        for column, variable in enumerate(evidence.observation_layout.variables)
                    }
                    for draw, history in enumerate(histories)
                ),
                paths=paths.revised(
                    states={
                        identity: series(
                            paths.states[identity],
                            evidence.latent_paths,
                            evidence.reference_latent_paths,
                            column,
                        )
                        for column, identity in enumerate(evidence.state_ids)
                    },
                    indicators={
                        variable.id: series(
                            paths.indicators[variable.id],
                            evidence.observations,
                            evidence.reference_observations,
                            column,
                            observed=True,
                        )
                        for column, variable in enumerate(evidence.observation_layout.variables)
                    },
                ),
                arrays=result_arrays(store, references),
            )


def complete_attempt(workspace_id: str, attempt: StagedActionAttempt) -> ActionAttempt:
    """Finish a staged execution and retain the one result its publication will own."""
    from nof1_causal_lab.study.records import failed_attempt, retained_attempt

    assert attempt.request is not None
    if not isinstance(attempt.outcome, Applied):
        return failed_attempt(attempt.request, attempt.outcome)
    inputs = StudyRepository(workspace_id).input_state(attempt.request)
    state = staged_state(inputs, attempt.outcome.effects)
    result = build_output(workspace_id, attempt, state)
    store = ArtifactStore(workspace_id)
    identity = store.write_result(result)
    produced = tuple(
        store.result_artifact(info, identity) for info in attempt.outcome.effects.produced
    )
    return retained_attempt(
        attempt, identity, attempt.outcome.effects.revised(produced=produced, reports={})
    )


def staged_state(inputs: StudyState, effects: ActionEffects) -> StudyState:
    """Resolve the execution's selected artifacts and its newly prepared history."""
    state = apply_effects(inputs, effects.produced, effects.retracted)
    panel = next((item for item in effects.produced if item.artifact_id == "panel"), None)
    return (
        state.revised(data=DataRef[GitOid, int](revision=panel.revision, replicate_index=0))
        if panel is not None
        else state
    )
