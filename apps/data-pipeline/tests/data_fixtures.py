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
from tests.inference_fixtures import compile_model_fixture


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
        preparation=preparation,
    )


def simulation_layout(model, times, mask, write_array):
    from nof1_causal_lab.models.ssm.observation_support import simulation_observation_support

    compiled = compile_model_fixture(model)
    support = simulation_observation_support(compiled, np.asarray(times))
    return SimulationObservationLayout(
        variables=tuple(
            model.indicator(observation.id).observation.resolved(observation.observation_window)
            for observation in compiled.observations
        ),
        support_start_times=write_array(support.support_start_times),
        support_end_times=write_array(support.support_end_times),
        mask=write_array(mask),
    )


def preparation_input(store, preparation):
    """Publish the model owning a test recipe before submitting extraction instructions."""
    from nof1_causal_lab.actions.io import PrepareDataInput
    from nof1_causal_lab.artifacts.construct import replace_constructs
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from tests.helpers import make_model

    variables = preparation.definition.variables
    model = make_model([item.observation.name for item in variables])
    model = model.revised(measurement_clock=preparation.definition.default_window).with_entities(
        edges=replace_constructs(
            model.edges,
            tuple(
                construct.revised(
                    indicators=(
                        IndicatorSpec(observation=item.observation, construct_polarity="positive"),
                    )
                )
                for construct in model.constructs
                for item in variables
                if construct.name == item.observation.name
            ),
        ),
    )
    info = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    return PrepareDataInput[GitOid, FileSourceRef](
        model_ref=info.revision,
        source=preparation.source,
        extraction={item.observation.id: item.extraction for item in variables},
        context=preparation.definition.context,
    )
