"""Explicit nonlinear replication and measurements from current model uncertainty."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.availability import NotApplicable, Unavailable
from nof1_causal_lab.artifacts.simulation import FitReliability, SimulationEvidence, SimulationObservationLayout, SimulationReport
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.predictive.simulation import (
    generate_simulation_batch,
    measure_simulation_batch,
)
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    import polars as pl

    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.simulation import SimulationSpec
    from nof1_causal_lab.study.store import ArtifactStore
    from nof1_causal_lab.models.model_structure import StructuralSelection


def simulate(
    selection: StructuralSelection,
    design: SimulationSpec,
    *,
    revision: GitRef,
    write_array: Callable[[np.ndarray], str],
    time_origin: datetime,
    origin_panel_revision: GitOid | None = None,
    input_data: pl.DataFrame | None = None,
) -> SimulationEvidence | ObservationPreflightFailure:
    """Generate current model histories; data_diff compares the saved observations separately."""
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model

    model = selection.model
    compiled = compile_executable_model(selection)
    from nof1_causal_lab.models.ssm.runtime import replay_input_events

    start, end = design.start_day(time_origin), design.end_day(time_origin)
    assignments = design.assignments(time_origin)
    retained_times = next(
        (law.layout.time_points for law in compiled.laws if law.layout.constructs), ()
    )
    history_start = (
        retained_times[int(np.searchsorted(retained_times, start, side="right")) - 1]
        if retained_times and start >= retained_times[0]
        else 0.0
    )
    input_events = replay_input_events(
        compiled,
        input_data,
        time_origin=time_origin,
        start=history_start,
        end=end,
    )
    if isinstance(input_events, ObservationPreflightFailure):
        return input_events
    batch = generate_simulation_batch(
        compiled,
        start=start,
        end=end,
        assignments=assignments,
        input_events=input_events,
        time_origin=time_origin,
    )
    support = batch.measurement_design.observation_support
    if support is None:
        raise ValueError("Simulation must retain its observation support")
    prediction = batch.prediction
    state_ids = tuple(numeric.state_ids(compiled))
    variables = tuple(
        model.indicator(observation.id).observation.resolved(observation.observation_window)
        for observation in compiled.observations
    )
    return SimulationEvidence(
        model=revision,
        design=design,
        time_origin=time_origin,
        times=batch.times,
        draws=prediction.n_draws,
        seed=batch.measurement_design.seed,
        origin_panel_revision=origin_panel_revision,
        state_ids=state_ids,
        parameter_draws={
            name: write_array(np.asarray(value)) for name, value in prediction.parameters.items()
        },
        latent_paths=write_array(np.asarray(prediction.trajectory.latents)),
        observations=write_array(np.asarray(prediction.trajectory.observations)),
        observation_layout=SimulationObservationLayout(
            variables=variables,
            support_start_times=write_array(np.asarray(support.support_start_times)),
            support_end_times=write_array(np.asarray(support.support_end_times)),
            mask=write_array(np.asarray(prediction.trajectory.observations_mask)),
        ),
        reference_latent_paths=write_array(np.asarray(prediction.reference.latents))
        if prediction.reference is not None
        else None,
        reference_observations=write_array(np.asarray(prediction.reference.observations))
        if prediction.reference is not None
        else None,
    )


def read_simulation_report(
    store: ArtifactStore, evidence: SimulationEvidence, question_revision: GitOid
) -> SimulationReport:
    """Measure saved histories with current code; never sample parameters, paths or emissions."""
    from nof1_causal_lab.actions.scenarios import summarize_causal_simulation
    from nof1_causal_lab.artifacts.predictive_provenance import FittedLawProvenance, MixedLawProvenance
    from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model
    from nof1_causal_lab.models.ssm.inference.convergence import convergence_failures
    from nof1_causal_lab.models.ssm.observation_support import recorded_observation_support
    from nof1_causal_lab.models.ssm.predictive.simulation import SimulationBatch
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import _predictive_models
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
    from nof1_causal_lab.models.ssm.simulation_checks import DesignInfo
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.lineage import fitted_law_report, law_provenance
    from nof1_causal_lab.study.records import inference_record
    from nof1_causal_lab.study.store import cached_value, read_model, read_question
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    def render() -> SimulationReport:
        model = read_model(store, evidence.model.revision)
        records = StudyRepository(store.workspace_id).attempts()
        law = law_provenance(store, store.read_meta("model", evidence.model.revision), model, None)
        reliability: FitReliability = "unknown" if law.kind == "unknown" else "not_fitted"
        if isinstance(law, (FittedLawProvenance, MixedLawProvenance)):
            report = fitted_law_report(store, records, law.fitted_model_revision)
            reliability = "unknown" if report.inference_diagnostics is None else "unconverged" if convergence_failures(report.convergence) else "converged"
        causal = Unavailable(reason="Causal effect certification is pending.") if evidence.design.interventions else NotApplicable(reason="No intervention was requested.")
        selection = StructuralSelection.for_question(model, read_question(store, question_revision))
        try:
            compiled = compile_executable_model(selection)
        except (AggregatedCompileError, IncompleteModelError) as exc:
            return SimulationReport(evidence=evidence, law=law, fit_reliability=reliability, causal=Unavailable(reason=str(exc)) if evidence.design.interventions else causal)
        times = jnp.asarray(evidence.times)
        parameters = {name: jnp.asarray(store.read_array(ref)) for name, ref in evidence.parameter_draws.items()}
        latents = jnp.asarray(store.read_array(evidence.latent_paths))
        observations = jnp.asarray(store.read_array(evidence.observations))
        mask = jnp.asarray(store.read_array(evidence.observation_layout.mask))
        support = recorded_observation_support(np.asarray(times), evidence.observation_layout.variables, store.read_array(evidence.observation_layout.support_start_times), store.read_array(evidence.observation_layout.support_end_times))
        if isinstance(support, ObservationPreflightFailure):
            raise ValueError(f"Recorded simulation support is corrupt: {support.message}")
        from nof1_causal_lab.models.ssm.execution.observation_operator import compile_observation_operator
        operator = compile_observation_operator(support)
        native = _predictive_models(compiled, parameters, times)

        def responses(native_model, path):
            observation = native_model.observation_model
            predictors = jax.vmap(observation.linear_predictor)(path)
            response = jax.vmap(lambda value: observation.at_predictor(value).response)(predictors)
            if operator is not None:
                response, _ = operator.project_response_trajectory(response)
            return predictors, response

        predictors, means = eqx.filter_vmap(responses)(native, latents)
        trajectory = PredictiveTrajectory(latents, predictors, observations, mask, jnp.where(mask, means, jnp.nan))
        prediction = PredictiveDraws(parameters, trajectory)
        variables = evidence.observation_layout.indicator_ids
        batch = SimulationBatch(evidence.times, prediction, None, DesignInfo(
            t_grid=times, manifest_ids=variables,
            obs_index_by_indicator={identity: np.flatnonzero(np.asarray(mask[:, :, index]).any(axis=0)) for index, identity in enumerate(variables)},
            values_by_indicator={identity: np.asarray([]) for identity in variables},
            n_draws=evidence.draws, seed=evidence.seed, observation_support=support,
        ), time_origin=evidence.time_origin)
        findings, _ = measure_simulation_batch(compiled, batch, clock=time.monotonic)
        return summarize_causal_simulation(selection, SimulationReport(evidence=evidence, law=law, findings=findings, fit_reliability=reliability, causal=causal), store=store, inference=inference_record(records, evidence.model.revision))

    value, _ = cached_value(store.workspace_id, ("simulation-report", question_revision, evidence.model_dump_json(round_trip=True)), TypeAdapter(SimulationReport), render)
    return value
