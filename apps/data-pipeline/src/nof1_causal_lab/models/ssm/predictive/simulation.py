"""Forward histories from current joint laws, with separate reusable measurements."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as random
import numpy as np

from nof1_causal_lab.artifacts.checks import (
    NotEvaluated,
    PredictiveAssessment,
    PredictiveSubject,
)
from nof1_causal_lab.artifacts.expressions import hill_applications
from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, IndicatorRef
from nof1_causal_lab.models.posterior_predictive import measure_predictive_checks
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.counterfactual.orchestration import ResolvedIntervention
from nof1_causal_lab.models.ssm.predictive.parameters import sample_model_laws
from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
    predictive_keys,
    simulate_latent_histories,
    simulate_predictive_draws,
)
from nof1_causal_lab.models.ssm.runtime import BoundPanel
from nof1_causal_lab.models.ssm.simulation_checks import (
    ConstructSimulationTarget,
    DesignInfo,
    measure_construct_simulation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorPredictiveChecks
    from nof1_causal_lab.artifacts.simulation import SimulationSpec
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws


SIMULATION_DRAWS = 100
SIMULATION_SEED = 0


@dataclass(frozen=True)
class SimulationBatch:
    """One generated batch and its resolved observation coordinates."""

    times: tuple[float, ...]
    prediction: PredictiveDraws
    observations: jnp.ndarray | None
    measurement_design: DesignInfo


def _time_grid(model: CompiledModel, design: SimulationSpec, start: float) -> np.ndarray:
    times = {
        float(value)
        for value in np.linspace(
            start, design.end, max(1, ceil((design.end - start) / model.clock_days)) + 1
        )
    }
    times.update(event.time for event in design.interventions)
    return np.asarray(sorted(times))


def generate_simulation_batch(
    source: CompiledModel | BoundPanel,
    design: SimulationSpec,
    *,
    input_events: tuple[ResolvedIntervention, ...] = (),
    times: np.ndarray | jnp.ndarray | None = None,
    draws: int = SIMULATION_DRAWS,
    seed: int = SIMULATION_SEED,
) -> SimulationBatch:
    """Sample current laws once and always generate nonlinear stochastic paths and emissions.

    Internal check callers may supply a prepared observation grid and execution budget.
    Dated requests derive their grid from the model clock and intervention boundaries.
    """
    if isinstance(source, BoundPanel):
        model = source.model
        times = source.times
        observations = source.values
        support = source.observation_support
        input_events = source.input_events
    else:
        model = source
        observations = None
        support = None
    indicator_ids = tuple(numeric.observation_ids(model))
    state_ids = tuple(numeric.state_ids(model))
    laws = sample_model_laws(model, draws=draws, key=predictive_keys(seed).parameters)
    time_points: tuple[float, ...] = next(
        (law.layout.time_points for law in model.laws if law.layout.constructs), ()
    )
    current_time = next(reversed(time_points)) if laws.latent_paths is not None else 0.0
    start = current_time if design.start is None else design.start
    # Resolve the omitted start before applying the same window validation.
    design = design.starting_at(start)
    grid = _time_grid(model, design, start) if times is None else np.asarray(times)
    if len(grid) < 2 or grid[0] != start or grid[-1] != design.end or np.any(np.diff(grid) <= 0):
        raise ValueError("The prepared simulation grid must increase from start through end")
    grid = np.asarray(
        sorted(
            {
                *grid,
                *(
                    event.spec.time
                    for event in input_events
                    if start < event.spec.time < design.end
                ),
            }
        )
    )
    initial = None
    if laws.latent_paths is None:
        if start < 0:
            raise ValueError("Simulation cannot start before the initial law at model day zero")
        if start > 0:
            history_events = tuple(event for event in input_events if event.spec.time <= start)
            history_grid = sorted(
                {
                    0.0,
                    start,
                    *(event.spec.time for event in history_events if 0 < event.spec.time < start),
                }
            )
            history, _, _ = simulate_latent_histories(
                model,
                laws.parameters,
                jnp.asarray(history_grid),
                random.fold_in(predictive_keys(seed).latents, 1),
                None,
                (),
                history_events,
            )
            initial = history[:, -1, :]
    if laws.latent_paths is not None:
        index = int(np.searchsorted(time_points, start, side="right")) - 1
        if index < 0:
            raise ValueError("Simulation cannot start before the model's first retained state")
        initial = jnp.zeros((draws, len(state_ids)), dtype=laws.latent_paths.dtype)
        initial = initial.at[
            :, jnp.asarray([state_ids.index(identity) for identity in laws.state_ids])
        ].set(laws.latent_paths[:, index, :])
        anchor = time_points[index]
        if anchor < start:
            # Advance a jointly drawn state to an unrepresented start time; never interpolate
            # retained paths or condition on a second independently sampled state.
            history_events = tuple(event for event in input_events if event.spec.time <= start)
            history_grid = sorted(
                {
                    anchor,
                    start,
                    *(
                        event.spec.time
                        for event in history_events
                        if anchor < event.spec.time < start
                    ),
                }
            )
            history, _, _ = simulate_latent_histories(
                model,
                laws.parameters,
                jnp.asarray(history_grid),
                random.fold_in(predictive_keys(seed).latents, 1),
                initial,
                (),
                history_events,
            )
            initial = history[:, -1, :]
    from nof1_causal_lab.models.ssm.observation_support import simulation_observation_support

    if support is None:
        support = simulation_observation_support(model, grid)
    interventions = []
    for event in design.interventions:
        if event.target not in state_ids:
            raise ValueError(f"Intervention target is not a model state: {event.target}")
        interventions.append(ResolvedIntervention(index=state_ids.index(event.target), spec=event))
    prediction = simulate_predictive_draws(
        model,
        laws.parameters,
        jnp.asarray(grid),
        seed=seed,
        initial_states=initial,
        interventions=tuple(interventions),
        input_events=input_events,
        observation_support=support,
        observation_mask=None if observations is None else ~jnp.isnan(observations),
    )
    indices: dict[str, np.ndarray] = {
        identity: np.flatnonzero(
            np.asarray(prediction.trajectory.observations_mask[:, :, i]).any(axis=0)
        )
        if observations is None
        else np.flatnonzero(np.isfinite(observations[:, i]))
        for i, identity in enumerate(indicator_ids)
    }
    values: dict[str, np.ndarray] = {
        identity: np.asarray([])
        if observations is None
        else np.asarray(observations[indices[identity], i])
        for i, identity in enumerate(indicator_ids)
    }
    return SimulationBatch(
        tuple(float(t) for t in grid),
        prediction,
        observations,
        DesignInfo(
            t_grid=jnp.asarray(grid),
            manifest_ids=indicator_ids,
            obs_index_by_indicator=indices,
            values_by_indicator=values,
            n_draws=draws,
            seed=seed,
            observation_support=support,
        ),
    )


def measure_simulation_batch(
    model: CompiledModel,
    batch: SimulationBatch,
    *,
    groups: tuple[str, ...] = ("dynamics", "measurement"),
    edge_contrasts: bool = False,
    clock: Callable[[], float],
) -> tuple[tuple[PredictiveAssessment, ...], PosteriorPredictiveChecks | None]:
    """All selected reducers consume the same generated paths."""
    prediction = batch.prediction
    observations = batch.observations
    measurement_design = batch.measurement_design
    indicator_ids = tuple(numeric.observation_ids(model))
    state_ids = tuple(numeric.state_ids(model))
    targets = {
        **{state.name: ConstructRef(id=state.id) for state in model.states},
        **{observation.id: IndicatorRef(id=observation.id) for observation in model.observations},
        **{
            f"{model.states[source].name}->{state.name}": EdgeRef(id=identity)
            for state in model.states
            for source, identity in state.incoming_edges
        },
    }
    findings: list[PredictiveAssessment] = []
    components = model.dynamics.spec.components if "dynamics" in groups else ()
    for state_index, identity in enumerate(
        state_ids if set(groups) & {"dynamics", "measurement"} else ()
    ):
        if model.states[state_index].is_input:
            continue
        incoming = [
            component
            for component in components
            if component.target == state_index and component.edge_owned
        ]
        parents = sorted({source for component in incoming for source in component.sources})
        hills = {
            source
            for component in incoming
            if any(hill_applications(component.expression))
            for source in component.sources
        }
        target = ConstructSimulationTarget(
            construct=model.states[state_index],
            edge_parents=tuple(model.states[source].name for source in parents),
            hill_parents=tuple(model.states[source].name for source in sorted(hills)),
        )
        measured, _ = measure_construct_simulation(
            model,
            prediction,
            measurement_design,
            target,
            dynamics="dynamics" in groups,
            measurement="measurement" in groups,
            edge_contrasts=edge_contrasts,
            clock=clock,
        )
        findings.extend(result.finding(identity, targets[result.target]) for result in measured)
    modeled_columns = [
        index
        for index, identity in enumerate(indicator_ids)
        if not model.states[model.observations[index].state_index].is_input
    ]
    nonfinite = bool(
        np.any(
            np.asarray(prediction.trajectory.observations_mask[:, :, modeled_columns])
            & ~np.isfinite(prediction.trajectory.observations[:, :, modeled_columns])
        )
    )
    checks = (
        None
        if observations is None or "data_comparison" not in groups or nonfinite
        else measure_predictive_checks(
            prediction.trajectory.observations[:, :, modeled_columns],
            observations[:, modeled_columns],
            tuple(indicator_ids[index] for index in modeled_columns),
        )
    )
    if "data_comparison" in groups and checks is None:
        findings.append(
            NotEvaluated(
                subject=PredictiveSubject(check="data_comparison", target="observations"),
                reason="NONFINITE_PATHS" if nonfinite else "COMPARISON_INPUTS_MISSING",
                detail="Predictive observations contain non-finite values."
                if nonfinite
                else "Observed data are required to assess predictive calibration.",
            )
        )
    return tuple(findings), checks
