"""Explicit data schemas accompanying test-owned observations."""

import numpy as np

from nof1_causal_lab.artifacts.data_preparation import (
    DataPreparationSpec,
    DataVariableSpec,
    FileSourceRef,
    PreparedDataMetadata,
)
from nof1_causal_lab.artifacts.observations import ObservationSpec
from nof1_causal_lab.artifacts.simulation import SimulationObservationLayout


def metadata_for_model(model):
    preparation = DataPreparationSpec(
        default_window=model.measurement_clock,
        variables=tuple(
            DataVariableSpec.model_validate(
                {
                    **indicator.model_dump(include=set(ObservationSpec.model_fields)),
                    "how_to_measure": "Read " + indicator.name,
                }
            )
            for indicator in model.indicators
        ),
    )
    return PreparedDataMetadata(
        source=FileSourceRef(files=("observations.csv",)),
        variables=preparation.observation_schema(),
        preparation=preparation,
    )


def simulation_layout(model, times, mask, write_array):
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.models.ssm.observation_support import simulation_observation_support

    support = simulation_observation_support(model, np.asarray(times))
    return SimulationObservationLayout(
        variables=tuple(
            ObservationSpec.model_validate(
                {
                    **model.indicator(identity).model_dump(
                        include=set(ObservationSpec.model_fields)
                    ),
                    "observation_window": model.indicator(identity).observation_window
                    or model.measurement_clock,
                }
            )
            for identity in numeric.observation_ids(model)
        ),
        support_start_times=write_array(support.support_start_times),
        support_end_times=write_array(support.support_end_times),
        mask=write_array(mask),
    )
