"""Explicit data schemas accompanying test-owned observations."""

from datetime import UTC, datetime

import numpy as np

from nof1_causal_lab.artifacts.data_preparation import (
    DataPreparationSpec,
    DataVariableSpec,
    FileSourceRef,
    PreparedDataMetadata,
    SemanticExtractionSpec,
)
from nof1_causal_lab.artifacts.simulation import SimulationObservationLayout
from tests.model_fixtures import compile_model_fixture


def metadata_for_model(model):
    preparation = DataPreparationSpec(
        default_window=model.measurement_clock,
        variables=tuple(
            DataVariableSpec(
                observation=indicator.observation,
                extraction=SemanticExtractionSpec(
                    how_to_measure="Read " + indicator.observation.name
                ),
            )
            for indicator in model.indicators
        ),
    )
    return PreparedDataMetadata(
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        source=FileSourceRef(files=("observations.csv",)),
        variables=preparation.observation_schema(),
        preparation=preparation,
    )


def simulation_layout(model, times, mask, write_array):
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.models.ssm.observation_support import simulation_observation_support

    support = simulation_observation_support(compile_model_fixture(model), np.asarray(times))
    return SimulationObservationLayout(
        variables=tuple(
            model.indicator(identity).observation.resolved(
                (
                    model.indicator(identity).observation.observation_window
                    or model.measurement_clock
                ).source
            )
            for identity in numeric.observation_ids(compile_model_fixture(model))
        ),
        support_start_times=write_array(support.support_start_times),
        support_end_times=write_array(support.support_end_times),
        mask=write_array(mask),
    )
