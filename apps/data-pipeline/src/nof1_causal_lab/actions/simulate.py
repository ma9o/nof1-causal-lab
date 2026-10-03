"""Explicit nonlinear replication and measurements from current model uncertainty."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.simulation import SimulationObservationLayout, SimulationReport
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
    from nof1_causal_lab.artifacts.simulation import FitReliability, SimulationSpec
    from nof1_causal_lab.models.model_structure import StructuralSelection


def simulate(
    selection: StructuralSelection,
    design: SimulationSpec,
    *,
    revision: GitRef,
    write_array: Callable[[np.ndarray], str],
    time_origin: datetime,
    fit_reliability: FitReliability = "not_fitted",
    input_data: pl.DataFrame | None = None,
) -> SimulationReport | ObservationPreflightFailure:
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
    findings, _ = measure_simulation_batch(compiled, batch, clock=time.monotonic)
    support = batch.measurement_design.observation_support
    if support is None:
        raise ValueError("Simulation must retain its observation support")
    prediction = batch.prediction
    state_ids = tuple(numeric.state_ids(compiled))
    indicator_ids = tuple(numeric.observation_ids(compiled))
    variables = tuple(
        model.indicator(identity).observation.resolved(support.observation_windows[index])
        for index, identity in enumerate(indicator_ids)
    )
    return SimulationReport(
        model=revision,
        design=design,
        time_origin=time_origin,
        assignments=assignments,
        times=batch.times,
        draws=prediction.n_draws,
        seed=batch.measurement_design.seed,
        fit_reliability=fit_reliability,
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
        findings=tuple(findings),
        reference_latent_paths=write_array(np.asarray(prediction.reference.latents))
        if prediction.reference is not None
        else None,
        reference_observations=write_array(np.asarray(prediction.reference.observations))
        if prediction.reference is not None
        else None,
    )
