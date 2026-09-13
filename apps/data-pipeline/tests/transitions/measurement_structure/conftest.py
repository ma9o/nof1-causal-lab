"""measurement-structure fixtures (identifiability / proxy resolution)."""

import pytest


@pytest.fixture
def stage1b_simple_latent():
    """Simple chain: Treatment -> Outcome (all observable)."""
    return {
        "default_outcome": {"kind": "construct", "id": "construct:170504d1ef8631dda85d"},
        "constructs": [
            {
                "id": "construct:2219a835484dea8b586a",
                "name": "Treatment",
                "role": "exogenous",
                "description": "The intervention",
                "temporal_status": "time_invariant",
            },
            {
                "id": "construct:170504d1ef8631dda85d",
                "name": "Outcome",
                "role": "endogenous",
                "description": "The result",
                "temporal_status": "time_varying",
            },
        ],
        "edges": [
            {
                "cause_id": "construct:2219a835484dea8b586a",
                "effect_id": "construct:170504d1ef8631dda85d",
                "id": "edge:1534f2f5ac96c22d9d9d",
                "description": "Treatment causes Outcome",
            },
        ],
    }


@pytest.fixture
def stage1b_confounded_latent():
    """Confounded: Treatment -> Outcome, Confounder -> Treatment, Confounder -> Outcome."""
    return {
        "default_outcome": {"kind": "construct", "id": "construct:170504d1ef8631dda85d"},
        "constructs": [
            {
                "id": "construct:2219a835484dea8b586a",
                "name": "Treatment",
                "role": "endogenous",
                "description": "The intervention",
                "temporal_status": "time_varying",
            },
            {
                "id": "construct:170504d1ef8631dda85d",
                "name": "Outcome",
                "role": "endogenous",
                "description": "The result",
                "temporal_status": "time_varying",
            },
            {
                "id": "construct:c1b2cfd25e61e303586c",
                "name": "Confounder",
                "role": "exogenous",
                "description": "Unmeasured common cause",
                "temporal_status": "time_invariant",
            },
        ],
        "edges": [
            {
                "cause_id": "construct:2219a835484dea8b586a",
                "effect_id": "construct:170504d1ef8631dda85d",
                "id": "edge:1534f2f5ac96c22d9d9d",
                "description": "Treatment causes Outcome",
            },
            {
                "cause_id": "construct:c1b2cfd25e61e303586c",
                "effect_id": "construct:2219a835484dea8b586a",
                "id": "edge:5157b8165624d0d8e1c7",
                "description": "Confounder affects Treatment",
            },
            {
                "cause_id": "construct:c1b2cfd25e61e303586c",
                "effect_id": "construct:170504d1ef8631dda85d",
                "id": "edge:0f365ad0e4745822404f",
                "description": "Confounder affects Outcome",
            },
        ],
    }


@pytest.fixture
def stage1b_measurement_all_observed():
    """Measurement structure with indicators for Treatment and Outcome."""
    return {
        "model_clock": "1d",
        "known_inputs": [],
        "scientific_only_constructs": [],
        "indicators": [
            {
                "id": "indicator:0d729b16859bd5e9357e",
                "construct_id": "construct:2219a835484dea8b586a",
                "name": "treatment_dose",
                "construct_polarity": "positive",
                "how_to_measure": "Extract the treatment dosage from the data",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "source_columns": ["treatment_dose"],
            },
            {
                "id": "indicator:54a6e8d16cbce15b4427",
                "construct_id": "construct:170504d1ef8631dda85d",
                "name": "outcome_score",
                "construct_polarity": "positive",
                "how_to_measure": "Extract the outcome score from the data",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "source_columns": ["outcome_score"],
            },
        ],
    }


@pytest.fixture
def stage1b_dummy_chunks():
    """Dummy data chunks for measurement structure proposal."""
    return [
        "Day 1: Patient took 10mg treatment, outcome score was 5.",
        "Day 2: Patient took 15mg treatment, outcome score was 7.",
        "Day 3: Patient took 10mg treatment, outcome score was 6.",
    ]
