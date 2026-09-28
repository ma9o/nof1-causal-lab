"""Explicit nonlinear replication and measurements from current model uncertainty."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.observations import ObservationSpec
from nof1_causal_lab.artifacts.simulation import SimulationObservationLayout, SimulationReport
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.predictive.simulation import (
    generate_simulation_batch,
    measure_simulation_batch,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.identity import GitRef
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.simulation import SimulationSpec


def simulate(
    model: ModelSpec,
    design: SimulationSpec,
    *,
    revision: GitRef,
    write_array: Callable[[np.ndarray], str],
) -> SimulationReport:
    """Generate current model histories; measurements consume the same batch."""
    batch = generate_simulation_batch(model, design)
    findings, _ = measure_simulation_batch(model, batch)
    support = batch.measurement_design.observation_support
    if support is None:
        raise ValueError("Simulation must retain its observation support")
    prediction = batch.prediction
    state_ids = tuple(numeric.state_ids(model))
    indicator_ids = tuple(numeric.observation_ids(model))
    return SimulationReport(
        model=revision,
        design=design,
        times=batch.times,
        draws=prediction.n_draws,
        seed=batch.measurement_design.seed,
        state_ids=state_ids,
        indicator_ids=indicator_ids,
        parameter_draws={
            name: write_array(np.asarray(value)) for name, value in prediction.parameters.items()
        },
        latent_paths=write_array(np.asarray(prediction.trajectory.latents)),
        observations=write_array(np.asarray(prediction.trajectory.observations)),
        observation_layout=SimulationObservationLayout(
            variables=tuple(
                ObservationSpec.model_validate({
                    **model.indicator(identity).model_dump(include=set(ObservationSpec.model_fields)),
                    "observation_window": support.observation_windows[index],
                })
                for index, identity in enumerate(indicator_ids)
            ),
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
