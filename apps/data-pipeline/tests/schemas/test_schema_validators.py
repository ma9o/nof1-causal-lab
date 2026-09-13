"""Tests for schema validation functions that collect all errors.

Covers: validate_latent_structure and validate_measurement_structure.
"""

from nof1_causal_lab.artifacts.latent_structure import LatentStructure, validate_latent_structure
from nof1_causal_lab.artifacts.measurement_structure import validate_measurement_structure
from tests.helpers import invalid_dict_payload


def _require_latent_structure(model: LatentStructure | None) -> LatentStructure:
    assert model is not None
    return model


def _valid_latent_data():
    """Minimal valid latent structure dict."""
    return {
        "default_outcome": {"kind": "construct", "id": "construct:cdc0b2958a9512b2abad"},
        "constructs": [
            {
                "id": "construct:6b04dc42c531e7091eb8",
                "name": "stress",
                "description": "Perceived stress",
                "role": "exogenous",
                "temporal_status": "time_varying",
            },
            {
                "id": "construct:cdc0b2958a9512b2abad",
                "name": "sleep",
                "description": "Sleep quality",
                "role": "endogenous",
                "temporal_status": "time_varying",
            },
        ],
        "edges": [
            {
                "cause_id": "construct:6b04dc42c531e7091eb8",
                "effect_id": "construct:cdc0b2958a9512b2abad",
                "id": "edge:4c52fe3e6d34a6f19f8c",
                "description": "Stress disrupts sleep",
            },
        ],
    }


def _valid_measurement_data():
    """Minimal valid measurement structure dict."""
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "id": "indicator:6bde869aba53fb51e0f4",
                "construct_id": "construct:6b04dc42c531e7091eb8",
                "name": "pss_score",
                "construct_polarity": "positive",
                "how_to_measure": "Perceived Stress Scale score",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            {
                "id": "indicator:9866c549bd1c25f0a5d7",
                "construct_id": "construct:cdc0b2958a9512b2abad",
                "name": "sleep_hours",
                "construct_polarity": "positive",
                "how_to_measure": "Hours of sleep reported",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
        ],
    }


# =============================================================================
# validate_latent_structure
# =============================================================================


class TestValidateLatentStructure:
    def test_valid_model_returns_model(self):
        model, errors = validate_latent_structure(_valid_latent_data())
        assert model is not None
        assert errors == []

    def test_not_dict_returns_error(self):
        model, errors = validate_latent_structure(invalid_dict_payload("not a dict"))
        assert model is None
        assert len(errors) == 1
        assert "dictionary" in errors[0].lower()

    def test_missing_constructs(self):
        model, errors = validate_latent_structure({"edges": []})
        assert model is None
        assert any("constructs" in e.lower() for e in errors)

    def test_constructs_not_list(self):
        model, errors = validate_latent_structure({"constructs": "not a list", "edges": []})
        assert model is None
        assert any("list" in e.lower() for e in errors)

    def test_duplicate_construct_name(self):
        data = _valid_latent_data()
        data["constructs"].append(data["constructs"][0].copy())
        model, errors = validate_latent_structure(data)
        assert model is None
        assert any("duplicate" in e.lower() for e in errors)

    def test_invalid_construct_schema(self):
        data = _valid_latent_data()
        data["constructs"][0] = {"name": "bad"}  # missing required fields
        model, errors = validate_latent_structure(data)
        assert model is None
        assert len(errors) > 0

    def test_multiple_errors_collected(self):
        """Should collect all errors, not just the first."""
        data = {
            "constructs": [
                {"name": "bad1"},  # invalid schema
                {"name": "bad2"},  # invalid schema
            ],
            "edges": [],
        }
        model, errors = validate_latent_structure(data)
        assert model is None
        assert len(errors) >= 2  # at least one per bad construct

    def test_edge_not_dict(self):
        data = _valid_latent_data()
        data["edges"].append("not a dict")
        model, errors = validate_latent_structure(data)
        assert model is None
        assert any("dictionary" in e.lower() for e in errors)

    def test_construct_not_dict(self):
        data = _valid_latent_data()
        data["constructs"].append(42)
        model, errors = validate_latent_structure(data)
        assert model is None
        assert any("dictionary" in e.lower() for e in errors)


# =============================================================================
# validate_measurement_structure
# =============================================================================


class TestValidateMeasurementStructure:
    def test_valid_model_returns_model(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        model, errors = validate_measurement_structure(
            _valid_measurement_data(), _require_latent_structure(latent)
        )
        assert model is not None
        assert errors == []

    def test_not_dict_returns_error(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        model, errors = validate_measurement_structure(
            invalid_dict_payload("not a dict"), _require_latent_structure(latent)
        )
        assert model is None
        assert len(errors) == 1

    def test_indicators_not_list(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        model, errors = validate_measurement_structure(
            {"indicators": "bad"}, _require_latent_structure(latent)
        )
        assert model is None
        assert any("list" in e.lower() for e in errors)

    def test_duplicate_indicator_name(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        data = _valid_measurement_data()
        data["indicators"].append(data["indicators"][0].copy())
        model, errors = validate_measurement_structure(data, _require_latent_structure(latent))
        assert model is None
        assert any("duplicate" in e.lower() for e in errors)

    def test_indicator_not_dict(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        data = {"indicators": [42]}
        model, errors = validate_measurement_structure(data, _require_latent_structure(latent))
        assert model is None
        assert any("dictionary" in e.lower() for e in errors)

    def test_invalid_indicator_schema(self):
        latent, _ = validate_latent_structure(_valid_latent_data())
        data = {"indicators": [{"name": "bad"}]}  # missing required fields
        model, errors = validate_measurement_structure(data, _require_latent_structure(latent))
        assert model is None
        assert len(errors) > 0
