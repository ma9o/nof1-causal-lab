"""Revision-pinned access to canonical aggregates and their compatible findings."""

from __future__ import annotations

from datetime import date, datetime
from functools import cache, cached_property
from typing import TYPE_CHECKING, Literal, cast

import jax
import numpy as np
import numpyro.distributions as dist
import polars as pl

from nof1_causal_lab.artifacts.expressions import expression_coefficients, expression_states
from nof1_causal_lab.artifacts.identity import GitOid, GitRef, ParameterRef
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.model_parameters import execution_parameters
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model
from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponentSpec
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
    MechanismCurves,
    MechanismViewRequest,
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
        self, workspace_id: str, *, at: GitOid | None = None, branch: str = "main"
    ) -> None:
        self.repository = StudyRepository(workspace_id)
        self.commit_id = self.repository.resolve(branch=branch, at=at)
        self.branch = branch
        self.store = ArtifactStore(workspace_id)
        self.workspace_id = workspace_id
        self.selected = cache(self._selected)

    @cached_property
    def state(self) -> StudyState:
        return self.repository.state(self.commit_id)

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
    def data_profile(self) -> Sourced[DataProfileArtifact] | None:
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
            report.for_indicators(frozenset(self._indicator_ids)),
            "validation_report",
            "",
        )

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
        report = result.report
        current = inference_report_is_current(result, self.state)
        return Sourced(
            value=report,
            source=FactSource(
                ref=GitRef(
                    workspace_id=self.workspace_id,
                    revision=record.commit_id,
                    path="logs/attempt.json",
                ),
                pointer="/attempt/outcome/result/report",
                validity=SourceValidity.FRESH if current else SourceValidity.STALE,
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
        curves = quantity_prior_densities(
            self.scoped(read_model(self.store, record.record.attempt.outcome.result.model.revision))
        )
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
            report = record.record.attempt.outcome.result.report
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
                        path="logs/attempt.json",
                    ),
                    pointer="/attempt/outcome/result/report",
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
        predictive = self.check_finding(
            self.state.checks.predictive if self.state.checks else None,
            "/predictive",
            validity=SourceValidity.STALE
            if self.state.checks is not None
            and self.state.checks.predictive is not None
            and self.state.checks.predictive.panel_revision
            != (self.state.current["panel"].revision if self.state.has("panel") else None)
            else SourceValidity.FRESH,
        )
        checks = self.state.checks
        current_panel = self.state.current["panel"].revision if self.state.has("panel") else None
        return ModelSnapshot(
            question=self.fact(self.question, "question", "") if self.question else None,
            model=self.fact(self.model, "model", "") if self.model else None,
            workspace_id=self.workspace_id,
            branch=self.branch,
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
                self._panel
                if self.state.matches_inputs("validation_report", "panel", "model")
                else None,
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
            specification=self.check_finding(
                self.state.checks.specification if self.state.checks else None,
                "/specification",
            ),
            question_checks=self.check_finding(
                checks.question if checks else None,
                "/question",
                validity=SourceValidity.STALE
                if checks is not None
                and checks.question is not None
                and checks.question.panel_revision != current_panel
                else SourceValidity.FRESH,
            ),
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
        check = self.state.checks.predictive if self.state.checks else None
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
        if start >= report.draws:
            raise StudyLookupError("Draw page starts past the saved simulation")
        from nof1_causal_lab.study.store import read_model

        observations = self.store.read_array(report.observations)
        mask = self.store.read_array(report.observation_layout.mask)
        reference = (
            self.store.read_array(report.reference_latent_paths)
            if report.reference_latent_paths is not None
            else None
        )
        reference_observations = (
            self.store.read_array(report.reference_observations)
            if report.reference_observations is not None
            else None
        )
        # Hydrate categorical emissions against their declared codebook at the storage edge.
        for index, variable in enumerate(report.observation_layout.variables):
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
            read_model(self.store, report.model.revision),
            self.store.read_array(report.latent_paths),
            observations,
            mask,
            reference,
            reference_observations,
            start=start,
            count=count,
        )

    def parameter_draws(self) -> ParameterDraws:
        """Read the fitted joint law instead of the report's small selection of pair plots."""
        from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
        from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
        from nof1_causal_lab.numpyro_json import empirical_atoms
        from nof1_causal_lab.study.lineage import law_provenance
        from nof1_causal_lab.study.visuals import empirical_points

        selection = self.selection
        if selection is None:
            return ParameterDraws(columns=(), unavailable_reason="No model at this revision.")
        model = selection.model
        provenance = law_provenance(self.store, self.state.current["model"], model, None)
        if provenance.kind != "fitted":
            return ParameterDraws(
                columns=(),
                unavailable_reason="This revision has no complete retained joint posterior. Recorded summary plots cannot recover missing draws.",
            )
        bindings, _ = parameter_bindings(compile_executable_model(selection))
        columns = []
        for identity in sorted(
            {
                parameter.distribution
                for parameter in execution_parameters(selection)
                if parameter.distribution
            }
        ):
            members = [
                b for b in bindings if model.parameter(b.parameter_id).distribution == identity
            ]
            layout = JointLawLayout.from_bindings(
                members,
                parameters=[b.parameter_id for b in members],
                constructs=[c.id for c in model.constructs if c.distribution == identity],
                time_points=model.time_points,
            )
            atoms = empirical_atoms(model.distributions[identity])
            for binding in members:
                for element, label in binding.elements.items():
                    columns.append(
                        ParameterDrawColumn(
                            label=label,
                            subject=ParameterRef(
                                parameter_id=binding.parameter_id, element_id=element
                            ),
                            values=tuple(
                                float(v) for v in atoms[:, layout.parameter_columns[element]]
                            ),
                            empirical=empirical_points(atoms[:, layout.parameter_columns[element]]),
                        )
                    )
        return ParameterDraws(columns=tuple(columns))

    def mechanism_curves(self, request: MechanismViewRequest) -> MechanismCurves:
        model = self.model
        if model is None:
            raise StudyLookupError("No model at this revision")
        edge = next((item for item in model.edges if item.id == request.owner_id), None)
        construct = next((item for item in model.constructs if item.id == request.owner_id), None)
        if edge is not None:
            mechanisms, target, default_axis = edge.mechanisms, edge.effect, edge.cause.id
        elif construct is not None:
            mechanisms, target, default_axis = construct.dynamics, construct, construct.id
        else:
            raise StudyLookupError("Unknown mechanism owner")
        if not mechanisms:
            raise StudyLookupError("No dynamics are declared for this entity")
        dependencies = {
            identity for m in mechanisms for identity in expression_states(m.expression)
        }
        dependencies.add(default_axis)
        axis = request.axis or default_axis
        if axis not in dependencies:
            raise StudyLookupError("The response axis must be a state in this mechanism")
        if request.moderator is not None and (
            request.moderator not in dependencies or request.moderator == axis
        ):
            raise StudyLookupError("The moderator must be another state in this mechanism")
        if request.held.keys() - dependencies or axis in request.held:
            raise StudyLookupError("Held values must name other states in this mechanism")
        held = {
            identity: request.held.get(identity, 0.0)
            for identity in sorted(dependencies - {axis, request.moderator})
        }
        ids = tuple(item.id for item in model.constructs)
        parameters, law, total = _mechanism_parameters(
            self.scoped(model),
            {
                operand.value
                for mechanism in mechanisms
                for operand in expression_coefficients(mechanism.expression)
                if isinstance(operand.value, str)
            },
        )
        if request.start >= total:
            raise StudyLookupError("Draw page starts past this law")
        components = [
            ExpressionComponentSpec(
                target=ids.index(target.id),
                source=ids.index(edge.cause.id) if edge is not None else None,
                kind=mechanism.kind,
                expression=mechanism.expression,
                state_ids=ids,
            ).build()
            for mechanism in mechanisms
        ]

        from nof1_causal_lab.study.mechanism_views import mechanism_curves

        return mechanism_curves(
            request,
            tuple(components),
            tuple(m.kind for m in mechanisms),
            ids,
            target.id,
            target.name,
            axis,
            {identity: model.get_construct(identity).name for identity in sorted(dependencies)},
            held,
            parameters,
            law,
            total,
        )


def _mechanism_parameters(
    selection: StructuralSelection, identities: set[ParameterId]
) -> tuple[dict[ParameterId, np.ndarray], Literal["retained", "sampled", "fixed"], int]:
    """Preserve joint atoms, or sample native current laws with a reproducible plot seed."""
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.compile.prior_compilation import quantity_parameter_law
    from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
    from nof1_causal_lab.numpyro_json import empirical_atoms, materialize_distribution

    model = selection.model
    parameters = [model.parameter(identity) for identity in sorted(identities)]
    if any(p.distribution is None for p in parameters):
        raise StudyLookupError("Assign probability laws to this mechanism's parameters first")
    if not parameters:
        return {}, "fixed", 1
    laws = {p.distribution for p in parameters if p.distribution is not None}
    native = {
        identity: materialize_distribution(model.distributions[identity]) for identity in laws
    }
    only = next(iter(native.values())) if len(native) == 1 else None
    retained = (
        isinstance(only, dist.MixtureSameFamily)
        and isinstance(only.component_distribution, dist.Delta)
        and np.all(
            np.asarray(only.mixing_distribution.probs)
            == np.asarray(only.mixing_distribution.probs)[0]
        )
    )
    total = len(empirical_atoms(only)) if retained and only is not None else 128
    values = {}
    for index, identity in enumerate(sorted(laws)):
        law = native[identity]
        members = [p for p in parameters if p.distribution == identity]
        key = jax.random.fold_in(jax.random.PRNGKey(0), index)
        if not law.batch_shape and not law.event_shape:
            for member_index, parameter in enumerate(members):
                compiled, _ = quantity_parameter_law(model, parameter)
                values[parameter.id] = np.asarray(
                    compiled.sample(jax.random.fold_in(key, member_index), (total,))
                )
        else:
            bindings, _ = parameter_bindings(compile_executable_model(selection))
            by_id = {b.parameter_id: b for b in bindings}
            layout = JointLawLayout.from_bindings(
                bindings,
                parameters=[
                    p.id for p in execution_parameters(selection) if p.distribution == identity
                ],
                constructs=[c.id for c in model.constructs if c.distribution == identity],
                time_points=model.time_points,
            )
            draws = empirical_atoms(law) if retained else np.asarray(law.sample(key, (total,)))
            for parameter in members:
                coordinates = by_id[parameter.id].coordinates
                if len(coordinates) != 1:
                    raise ValueError(
                        "A scalar drift coefficient requires one scientific coordinate"
                    )
                column = layout.parameter_columns[next(iter(coordinates))]
                values[parameter.id] = draws[:, column]
    return values, "retained" if retained else "sampled", total


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
