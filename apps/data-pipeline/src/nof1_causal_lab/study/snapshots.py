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
from nof1_causal_lab.artifacts.predictive_provenance import FittedLawProvenance, MixedLawProvenance
from nof1_causal_lab.models.model_parameters import execution_parameters
from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model
from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponentSpec
from nof1_causal_lab.study.artifact_files import artifact_file_spec, parquet_filename
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, FitAttempt, SimulateAttempt, StudyRevision
from nof1_causal_lab.study.snapshot_models import (
    FactSource,
    FitSummary,
    ModelData,
    ModelFindings,
    ModelSnapshot,
    SnapshotContext,
    Sourced,
    SourceValidity,
)
from nof1_causal_lab.study.state import StudyState, is_stale
from nof1_causal_lab.study.store import ArtifactStore, observation_sample, read_payload
from nof1_causal_lab.study.views import (
    entity_failures,
    measurements_view,
    model_diagnostics_view,
    raw_data_view,
)
from nof1_causal_lab.study.visual_models import (
    MechanismCurves,
    MechanismViewRequest,
    ObservationHistory,
    ParameterDrawColumn,
    ParameterDraws,
    PredictiveHistory,
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
    from nof1_causal_lab.artifacts.simulation import SimulationReport
    from nof1_causal_lab.artifacts.validation_report import (
        DataProfileArtifact,
        ValidationReportArtifact,
    )
    from nof1_causal_lab.study.view_models import (
        DensityPoint,
        MeasurementsData,
        ModelDiagnostics,
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
    def diagnostics(self) -> ModelDiagnostics | None:
        if self.model is None:
            return None
        compatible = self.state.matches_inputs("validation_report", "panel", "model")
        validation = self.validation_report if compatible else None
        return model_diagnostics_view(
            self.model,
            panel=self._panel if compatible else None,
            validation=validation.value if validation else None,
        )

    def artifact_view(
        self, name: str
    ) -> (
        ModelSpec
        | ModelDiagnostics
        | RawDataData
        | MeasurementsData
        | ValidationReportArtifact
        | InferenceReport
        | None
    ):
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
            case "inference_report":
                finding = self.inference_report
            case _:
                raise StudyLookupError(f"Unknown artifact view: {name}")
        return finding.value if finding is not None else None

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
        assert isinstance(record.record.attempt, FitAttempt)
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
            estimate = findings[0]
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
                report=posterior,
                edge_estimates=edge_estimates,
                decay_estimates=decay_estimates,
                prior_densities=self.fit_prior_densities(marginals.keys()),
            ),
            source=read.source,
        )

    def fit_prior_densities(
        self, fitted: Iterable[ParameterId]
    ) -> dict[ParameterId, tuple[DensityPoint, ...]]:
        """Curves of the input laws the fit conditioned, where it reports posteriors."""
        from nof1_causal_lab.study.lineage import inference_report_record
        from nof1_causal_lab.study.store import read_model

        record = inference_report_record(self.records, self.state)
        assert record is not None
        assert isinstance(record.record.attempt, FitAttempt)
        assert isinstance(record.record.attempt.outcome, Applied)
        curves = quantity_prior_densities(
            read_model(self.store, record.record.attempt.outcome.result.model.revision)
        )
        return {identity: curves[identity] for identity in fitted if curves.get(identity)}

    def simulation(self) -> Sourced[SimulationReport] | None:
        """Return the most recent explicit simulation with its own input revisions."""

        for record in reversed(self.records):
            if not isinstance(record.record.attempt, SimulateAttempt) or not isinstance(
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
            from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel, compile_model
            from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
                forward_simulation_supported,
            )

            compiled = compile_model(self.model)
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
        return ModelSnapshot(
            model=self.fact(self.model, "model", "") if self.model else None,
            context=SnapshotContext(
                workspace_id=self.workspace_id,
                seq=self.seq,
                commit_id=self.commit_id,
                branch=self.branch,
                can_simulate=can_simulate,
                current=self.state.current,
            ),
            data=ModelData(
                raw_data=self.raw_data,
                measurements=self.measurements,
                metadata=self.data_metadata,
                profile=self.data_profile,
            ),
            findings=ModelFindings(
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
                diagnostics=self.diagnostics,
                fit=fit,
                simulation=self.simulation(),
                specification=self.check_finding(
                    self.state.checks.specification if self.state.checks else None,
                    "/specification",
                ),
                predictive=predictive,
            ),
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

    def predictive_history(self, indicator_id: IndicatorId) -> PredictiveHistory | None:
        """Recover the saved check's exact schedule from its pinned inputs, without prediction."""
        from nof1_causal_lab.models.ssm.observation_support import (
            augment_wide_data_with_support_boundaries,
        )
        from nof1_causal_lab.models.ssm.runtime import project_observation_data
        from nof1_causal_lab.study.lineage import fitted_law_report, read_data_metadata
        from nof1_causal_lab.study.store import read_model

        check = self.state.checks.predictive if self.state.checks else None
        if check is None or check.predictive_checks is None or check.panel_revision is None:
            return None
        overlay = next(
            (
                item
                for item in check.predictive_checks.overlays
                if item.indicator_id == indicator_id
            ),
            None,
        )
        if overlay is None:
            return None
        model = read_model(self.store, check.model_revision)
        origin = read_data_metadata(self.store, check.panel_revision).time_origin
        if isinstance(check.law, (FittedLawProvenance, MixedLawProvenance)):
            origin = fitted_law_report(
                self.repository.attempts(), check.law.fitted_model_revision
            ).time_origin
        panel = self.store.read_parquet_file("panel", check.panel_revision, "panel.parquet")
        compiled = compile_executable_model(model)
        wide, rows = project_observation_data(panel, model_spec=compiled, time_origin=origin)
        wide = augment_wide_data_with_support_boundaries(rows, wide, time_origin=origin)
        times = tuple(float(value) for value in wide["time"])
        if len(times) != len(overlay.observed):
            raise ValueError(
                "Saved predictive series do not match their pinned observation schedule"
            )
        likelihood = next(
            law
            for indicator, law in model.iter_likelihoods()
            if indicator.observation.id == indicator_id
        )
        return PredictiveHistory(
            times=times, time_origin=origin, standardized=likelihood.standardized, overlay=overlay
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
        stop = min(start + count, report.draws)
        latent = self.store.read_array(report.latent_paths)[start:stop]
        observed = self.store.read_array(report.observations)[start:stop]
        mask = self.store.read_array(report.observation_layout.mask)[start:stop]
        reference = (
            self.store.read_array(report.reference_latent_paths)[start:stop]
            if report.reference_latent_paths is not None
            else None
        )
        reference_observed = (
            self.store.read_array(report.reference_observations)[start:stop]
            if report.reference_observations is not None
            else None
        )
        effect = None
        if report.causal_result is not None:
            if reference is None:
                raise ValueError("A causal simulation requires its retained reference paths")
            outcome = report.state_ids.index(report.causal_result.outcome)
            effect = (
                report.causal_result.labels[report.causal_result.outcome],
                latent[:, :, outcome] - reference[:, :, outcome],
            )
        return recorded_simulation_paths(
            report, latent, observed, mask, reference, reference_observed, effect, start=start
        )

    def parameter_draws(self) -> ParameterDraws:
        """Read the fitted joint law instead of the report's small selection of pair plots."""
        from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
        from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
        from nof1_causal_lab.numpyro_json import empirical_atoms
        from nof1_causal_lab.study.lineage import law_provenance
        from nof1_causal_lab.study.visuals import empirical_points

        model = self.model
        if model is None:
            return ParameterDraws(columns=(), unavailable_reason="No model at this revision.")
        provenance = law_provenance(self.store, self.state.current["model"], model, None)
        if provenance.kind != "fitted":
            return ParameterDraws(
                columns=(),
                unavailable_reason="This revision has no complete retained joint posterior. Recorded summary plots cannot recover missing draws.",
            )
        bindings, _ = parameter_bindings(compile_executable_model(model))
        columns = []
        for identity in sorted(
            {
                parameter.distribution
                for parameter in execution_parameters(model)
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
            model,
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
    model: ModelSpec, identities: set[ParameterId]
) -> tuple[dict[ParameterId, np.ndarray], Literal["retained", "sampled", "fixed"], int]:
    """Preserve joint atoms, or sample native current laws with a reproducible plot seed."""
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.compile.prior_compilation import quantity_parameter_law
    from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
    from nof1_causal_lab.numpyro_json import empirical_atoms, materialize_distribution

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
            bindings, _ = parameter_bindings(compile_executable_model(model))
            by_id = {b.parameter_id: b for b in bindings}
            layout = JointLawLayout.from_bindings(
                bindings,
                parameters=[
                    p.id for p in execution_parameters(model) if p.distribution == identity
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


def quantity_prior_densities(model: ModelSpec) -> dict[ParameterId, tuple[DensityPoint, ...]]:
    """Resolve native quantity laws at the reader boundary before projecting curves."""
    from nof1_causal_lab.models.ssm.compile.prior_compilation import quantity_parameter_law
    from nof1_causal_lab.numpyro_json import distribution_shape
    from nof1_causal_lab.study.prior_views import prior_density

    return {
        parameter.id: prior_density(quantity_parameter_law(model, parameter)[0])
        for parameter in execution_parameters(model)
        if parameter.distribution is not None
        and not any(distribution_shape(model.distributions[parameter.distribution]))
    }
