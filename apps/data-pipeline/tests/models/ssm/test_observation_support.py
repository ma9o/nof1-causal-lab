from pathlib import Path

import polars as pl
import pytest

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.observation_support import validate_discrete_manifest_metadata
from tests.model_fixtures import compile_model_fixture

pytestmark = pytest.mark.contract


def _single_row_panel(**overrides: float) -> pl.DataFrame:
    values = {
        "stress_cont": 0.0,
        "adherence_flag": 1.0,
        "steps_count": 10.0,
        "fatigue_t": 0.0,
        "screen_gap": 1.0,
        "sleep_efficiency": 0.8,
        "symptom_severity": 0.0,
        "coping_style": 0.0,
        "rumination_count": 1.0,
        "focus_cont": 0.0,
    }
    values.update(overrides)
    return pl.DataFrame(values)


def test_declared_discrete_levels_allow_one_observed_level():
    spec = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "observation_support/mixed_family_model.json"
        ).read_text()
    )

    validate_discrete_manifest_metadata(compile_model_fixture(spec), _single_row_panel())

    counts = dict(
        zip(
            numeric.observation_names(compile_model_fixture(spec)),
            numeric.observation_level_counts(compile_model_fixture(spec)),
            strict=True,
        )
    )
    assert counts["symptom_severity"] == counts["coping_style"] == 4


def test_declared_discrete_levels_reject_out_of_range_code():
    spec = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "observation_support/mixed_family_model.json"
        ).read_text()
    )

    with pytest.raises(ValueError, match=r"outside declared range 0\.\.3"):
        validate_discrete_manifest_metadata(
            compile_model_fixture(spec), _single_row_panel(symptom_severity=4.0)
        )


def test_missing_declared_levels_are_rejected_in_the_scientific_definition():

    indicator = next(
        item
        for item in ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "observation_support/mixed_family_model.json"
            ).read_text()
        ).indicators
        if item.observation.measurement_dtype == "ordinal"
    )
    with pytest.raises(ValueError, match="ordinal_levels"):
        indicator.revised(observation=indicator.observation.revised(ordinal_levels=None))
