"""Explicit nonlinear replication and measurements from current model uncertainty."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import numpy as np

from nof1_causal_lab.artifacts.arrays import NumericalArray
from nof1_causal_lab.artifacts.availability import NotApplicable, Unavailable
from nof1_causal_lab.artifacts.simulation import (
    FitReliability,
    PairedArmSimulation,
    SimulationArm,
    SimulationEvidence,
    SimulationObservationLayout,
    SimulationReport,
    SingleArmSimulation,
)
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.predictive.simulation import (
    generate_simulation_batch,
    measure_simulation_batch,
)
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure

if TYPE_CHECKING:
    import dynestyx as dsx
    from jax import Array

    from nof1_causal_lab.artifacts.identity import GitOid, GitRef
    from nof1_causal_lab.artifacts.simulation import SimulationSpec
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.execution.dynamical_model import HeterogeneousObservation
    from nof1_causal_lab.study.store import ArtifactStore


def simulate(
    selection: StructuralSelection,
    design: SimulationSpec,
    *,
    revision: GitRef,
) -> SimulationEvidence | ObservationPreflightFailure:
    """Generate current model histories; data_diff compares the saved observations separately."""
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model

    compiled = compile_executable_model(selection)
    time_origin = next(
        (
            law.layout.time_origin
            for law in compiled.laws
            if law.layout.constructs and not isinstance(law.layout.time_origin, str)
        ),
        design.start_instant,
    )
    start, end = design.start_day(time_origin), design.end_day(time_origin)
    assignments = design.assignments(time_origin)
    batch = generate_simulation_batch(
        compiled,
        start=start,
        end=end,
        assignments=assignments,
        time_origin=time_origin,
    )
    if isinstance(batch, ObservationPreflightFailure):
        return batch
    support = batch.measurement_design.observation_support
    if support is None:
        raise ValueError("Simulation must retain its observation support")
    prediction = batch.prediction
    state_ids = tuple(numeric.state_ids(compiled))
    variables = tuple(observation.observation for observation in compiled.observations)
    action = SimulationArm(
        latent_paths=NumericalArray.from_numpy(np.asarray(prediction.trajectory.latents)),
        observations=NumericalArray.from_numpy(np.asarray(prediction.trajectory.observations)),
    )
    return SimulationEvidence(
        model=revision,
        design=design,
        time_origin=time_origin,
        times=batch.times,
        draws=prediction.n_draws,
        seed=batch.measurement_design.seed,
        state_ids=state_ids,
        parameter_draws={
            name: NumericalArray.from_numpy(np.asarray(value)) for name, value in prediction.parameters.items()
        },
        arms=SingleArmSimulation(action=action)
        if prediction.reference is None
        else PairedArmSimulation(
            action=action,
            reference=SimulationArm(
                latent_paths=NumericalArray.from_numpy(np.asarray(prediction.reference.latents)),
                observations=NumericalArray.from_numpy(np.asarray(prediction.reference.observations)),
            ),
        ),
        observation_layout=SimulationObservationLayout(
            variables=variables,
            support_start_times=NumericalArray.from_numpy(np.asarray(support.support_start_times)),
            support_end_times=NumericalArray.from_numpy(np.asarray(support.support_end_times)),
            mask=NumericalArray.from_numpy(np.asarray(prediction.trajectory.observations_mask)),
        ),
    )


def read_simulation_report(
    store: ArtifactStore, evidence: SimulationEvidence, question_revision: GitOid
) -> SimulationReport:
    """Evaluate this simulation's findings from its already generated histories."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from nof1_causal_lab.actions.scenarios import summarize_causal_simulation
    from nof1_causal_lab.actions.simulation_summaries import simulation_summary
    from nof1_causal_lab.artifacts.predictive_provenance import (
        FittedLawProvenance,
        MixedLawProvenance,
    )
    from nof1_causal_lab.compilation_errors import AggregatedCompileError, IncompleteModelError
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model
    from nof1_causal_lab.models.ssm.inference.convergence import convergence_failures
    from nof1_causal_lab.models.ssm.observation_support import recorded_observation_support
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import _predictive_models
    from nof1_causal_lab.models.ssm.predictive.simulation import SimulationBatch
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
    from nof1_causal_lab.models.ssm.simulation_checks import DesignInfo
    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.lineage import fitted_law_report, law_provenance
    from nof1_causal_lab.study.records import inference_record
    from nof1_causal_lab.study.store import read_model, read_question

    def render() -> SimulationReport:
        model = read_model(store, evidence.model.revision)
        records = StudyRepository(store.workspace_id).attempts()
        law = law_provenance(store, store.read_meta("model", evidence.model.revision), model, None)
        reliability: FitReliability = "unknown" if law.kind == "unknown" else "not_fitted"
        if isinstance(law, (FittedLawProvenance, MixedLawProvenance)):
            report = fitted_law_report(store, records, law.fitted_model_revision)
            reliability = (
                "unknown"
                if report.inference_diagnostics is None
                else "unconverged"
                if convergence_failures(report.convergence)
                else "converged"
            )
        causal = (
            Unavailable(reason="Causal effect certification is pending.")
            if evidence.design.interventions
            else NotApplicable(reason="No intervention was requested.")
        )
        latents = evidence.arms.action.latent_paths.values
        observations = evidence.arms.action.observations.values
        mask = evidence.observation_layout.mask.values
        reference = (
            evidence.arms.reference.latent_paths.values
            if isinstance(evidence.arms, PairedArmSimulation)
            else None
        )
        reference_observations = (
            evidence.arms.reference.observations.values
            if isinstance(evidence.arms, PairedArmSimulation)
            else None
        )
        for index, variable in enumerate(evidence.observation_layout.variables):
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
                    raise ValueError(
                        "Saved simulation emissions differ from their declared category codes"
                    )
        summary = simulation_summary(
            evidence, latents, observations, mask, reference, reference_observations
        )
        selection = StructuralSelection.for_question(model, read_question(store, question_revision))
        try:
            compiled = compile_executable_model(selection)
        except (AggregatedCompileError, IncompleteModelError) as exc:
            return SimulationReport(
                evidence=evidence,
                summary=summary,
                law=law,
                fit_reliability=reliability,
                causal=Unavailable(reason=str(exc)) if evidence.design.interventions else causal,
            )
        times = jnp.asarray(evidence.times)
        parameters = {
            name: jnp.asarray(ref.values)
            for name, ref in evidence.parameter_draws.items()
        }
        support = recorded_observation_support(
            np.asarray(times),
            evidence.observation_layout.variables,
            evidence.observation_layout.support_start_times.values,
            evidence.observation_layout.support_end_times.values,
        )
        if isinstance(support, ObservationPreflightFailure):
            raise ValueError(f"Recorded simulation support is corrupt: {support.message}")
        from nof1_causal_lab.models.ssm.execution.observation_operator import (
            compile_observation_operator,
        )

        operator = compile_observation_operator(
            support,
            held_channels=tuple(
                index
                for index, observation in enumerate(compiled.observations)
                if compiled.states[observation.state_index].is_input
            ),
        )
        native = _predictive_models(compiled, parameters, times)

        def responses(native_model: dsx.DynamicalModel, path: Array) -> tuple[Array, Array]:
            observation = cast("HeterogeneousObservation", native_model.observation_model)
            predictors = jax.vmap(observation.linear_predictor)(path)
            response = jax.vmap(lambda value: observation.at_predictor(value).response)(predictors)
            response, _ = operator.project_response_trajectory(response)
            return predictors, response

        predictors, means = eqx.filter_vmap(responses)(native, jnp.asarray(latents))
        trajectory = PredictiveTrajectory(
            jnp.asarray(latents),
            predictors,
            jnp.asarray(observations),
            jnp.asarray(mask),
            jnp.where(mask, means, jnp.nan),
        )
        prediction = PredictiveDraws(parameters, trajectory)
        variables = evidence.observation_layout.indicator_ids
        batch = SimulationBatch.from_draws(
            prediction,
            DesignInfo(
                t_grid=times,
                manifest_ids=variables,
                obs_index_by_indicator={
                    identity: np.flatnonzero(np.asarray(mask[:, :, index]).any(axis=0))
                    for index, identity in enumerate(variables)
                },
                values_by_indicator={identity: np.asarray([]) for identity in variables},
                n_draws=evidence.draws,
                seed=evidence.seed,
                observation_support=support,
            ),
            time_origin=evidence.time_origin,
        )
        findings, _ = measure_simulation_batch(compiled, batch, clock=time.monotonic)
        return summarize_causal_simulation(
            selection,
            SimulationReport(
                evidence=evidence,
                summary=summary,
                law=law,
                findings=findings,
                fit_reliability=reliability,
                causal=causal,
            ),
            store=store,
            inference=inference_record(records, evidence.model.revision),
        )

    return render()
