"""Observation subjects remain stable and invalid references fail at production."""

from datetime import datetime

import polars as pl
import pytest

from nof1_causal_lab.actions.validation.flow import validate_extraction
from nof1_causal_lab.utils.aggregations import compute_indicators
from nof1_causal_lab.utils.data import annotate_observation_rows
from nof1_causal_lab.workers.schemas import validate_worker_output
from tests.helpers import make_model

pytestmark = pytest.mark.contract


def test_one_observation_definition_is_shared_by_preparation_and_model():
    from nof1_causal_lab.artifacts.data_preparation import (
        DataPreparationSpec,
        DataVariableSpec,
    )
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.observations import ObservationSpec

    observation = ObservationSpec(
        id="indicator:dose",
        name="dose",
        measurement_dtype="continuous",
        aggregation="last",
        observation_window="1d",
        fill_null="forward",
        fill_null_limit=2,
    )
    fields = observation.model_dump()
    recipe = DataPreparationSpec(
        default_window="1d",
        variables=(
            DataVariableSpec(
                **fields,
                extraction_mode="computed",
                source_columns=("dose",),
                how_to_measure="Read dose",
            ),
        ),
    )
    assert recipe.observation_schema() == (observation,)
    indicator = IndicatorSpec(**fields, construct_polarity="positive")
    assert indicator.model_dump(include=set(ObservationSpec.model_fields)) == fields


def _measurement(name="Mood"):
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "id": "indicator:mood",
                "construct_id": "construct:mood",
                "name": name,
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "source_columns": ["score"],
                "extraction_mode": "computed",
            }
        ],
    }


def test_computed_and_semantic_extraction_preserve_the_same_subject_after_rename():
    raw = pl.DataFrame({"timestamp": [datetime(2026, 1, 1)], "score": [4.0]})
    outputs = []
    for label in ("Mood", "Renamed mood"):
        measurement = _measurement(label)
        computed = compute_indicators(raw, measurement["indicators"], "1d", "timestamp")
        assert computed["indicator_id"].to_list() == ["indicator:mood"]
        worker, errors = validate_worker_output(
            {
                "extractions": [
                    {
                        "indicator_id": "indicator:mood",
                        "window_start": "2026-01-01T00:00:00",
                        "value": 4.0,
                    }
                ]
            },
            measurement,
        )
        assert errors == []
        assert worker is not None
        semantic = worker.to_dataframe()
        assert semantic["indicator_id"].to_list() == ["indicator:mood"]
        outputs.append(annotate_observation_rows(semantic, measurement))
    assert outputs[0].equals(outputs[1])


def test_observation_producers_reject_owners_outside_the_pinned_measurement():
    rows = pl.DataFrame(
        {"indicator_id": ["indicator:absent"], "value": [1.0], "timestamp": ["2026-01-01T00:00:00"]}
    )
    with pytest.raises(ValueError, match="unknown indicators"):
        annotate_observation_rows(rows, _measurement())
    report = validate_extraction(make_model(["mood"]), [rows])
    assert not report["is_valid"]
    worker, errors = validate_worker_output(
        {
            "extractions": [
                {"indicator_id": "indicator:absent", "window_start": "2026-01-01", "value": 1.0}
            ]
        },
        _measurement(),
    )
    assert worker is None
    assert any("not in indicators" in error for error in errors)
