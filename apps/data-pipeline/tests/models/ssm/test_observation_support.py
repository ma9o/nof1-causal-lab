import polars as pl
import pytest

from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.observation_support import validate_discrete_manifest_metadata
from tests.models.ssm._support import complex_mixed_runtime_spec


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
    spec = complex_mixed_runtime_spec()

    validate_discrete_manifest_metadata(spec, _single_row_panel())

    counts = dict(
        zip(numeric.observation_names(spec), numeric.observation_level_counts(spec), strict=True)
    )
    assert counts["symptom_severity"] == counts["coping_style"] == 4


def test_declared_discrete_levels_reject_out_of_range_code():
    spec = complex_mixed_runtime_spec()

    with pytest.raises(ValueError, match=r"outside declared range 0\.\.3"):
        validate_discrete_manifest_metadata(spec, _single_row_panel(symptom_severity=4.0))


def test_missing_declared_levels_are_rejected_in_the_scientific_definition():
    from nof1_causal_lab.artifacts.indicator import Indicator

    indicator = next(
        item
        for item in complex_mixed_runtime_spec().indicators
        if item.measurement_dtype == "ordinal"
    )
    with pytest.raises(ValueError, match="ordinal_levels"):
        Indicator.model_validate({**indicator.model_dump(), "ordinal_levels": None})
