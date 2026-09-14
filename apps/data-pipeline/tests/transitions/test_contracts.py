"""Tests for artifact payload contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from nof1_causal_lab.flows.context_tools import CONTEXT_TOOLS
from tests.artifact_contract_support import validate_artifact_payload
from tests.helpers import graph_constructs, make_model


@pytest.fixture
def valid_artifact_payloads() -> dict[str, dict[str, Any]]:
    """Minimal valid payload for each persisted artifact id."""
    return {
        "validation_report": {
            "is_valid": True,
            "indicators": {
                "indicator:3696aef3ff6f446744e5": {
                    "profile": {
                        "measurement_dtype": "continuous",
                        "n_obs": 10,
                        "mean": 3.2,
                        "std": 1.1,
                        "min": 1.0,
                        "max": 5.0,
                        "q25": 2.0,
                        "q50": 3.0,
                        "q75": 4.0,
                        "variance": 1.2,
                        "time_coverage_ratio": 1.0,
                        "max_gap_ratio": 0.2,
                        "dtype_violations": 0,
                        "duplicate_pct": 0.1,
                        "arithmetic_sequence_detected": False,
                        "n_unparseable_timestamps": 0,
                        "zero_fraction": 0.0,
                        "is_nonnegative": True,
                        "is_unit_interval": False,
                        "looks_integer_valued": True,
                        "variance_to_mean_ratio": 0.375,
                    },
                    "validation": {
                        "issues": [],
                        "checks": {
                            "n_obs": "ok",
                            "variance": "ok",
                            "n_unparseable_timestamps": "ok",
                            "time_coverage_ratio": "ok",
                            "max_gap_ratio": "ok",
                            "dtype_violations": "ok",
                            "duplicate_pct": "ok",
                            "arithmetic_sequence_detected": "ok",
                        },
                    },
                }
            },
            "dataset_issues": [],
        },
        "baseline_report": {
            "intervention_results": [
                {
                    "treatment": "Stress",
                    "treatment_id": "construct:stress",
                    "posterior_draws": [0.08, 0.11, 0.14, 0.09, 0.15, 0.12, 0.10, 0.13],
                    "summary": {
                        "mean": 0.115,
                        "median": 0.115,
                        "lower_95": 0.08175,
                        "upper_95": 0.14825,
                        "prob_positive": 1.0,
                    },
                }
            ],
            "simulation_results": [],
            "final_summary": "Stress reduction remains the dominant actionable lever.",
        },
        "model": make_model(["Stress", "Perf"], [("Stress", "Perf")]).model_dump(mode="json"),
    }


def test_tool_server_registry_matches_served_tool_contracts() -> None:
    """Served tool contracts should match the tool server registry exactly."""
    from nof1_causal_lab.tool_server import _TOOL_IMPLS

    served_context_ids = {context_id for context_id, _tool_name in _TOOL_IMPLS}
    assert served_context_ids == {
        "latent-structure",
        "measurement-structure",
        "measurement",
        "statistical-model-spec",
        "ranking",
    }

    for context_id in served_context_ids:
        contract_names = {tool.name for tool in CONTEXT_TOOLS[context_id]}
        runtime_names = {
            tool_name
            for served_context_id, tool_name in _TOOL_IMPLS
            if served_context_id == context_id
        }
        assert runtime_names == contract_names


def test_validate_artifact_payload_accepts_all_artifacts(
    valid_artifact_payloads: dict[str, dict[str, Any]],
):
    """Each artifact payload validates and round-trips to a JSON-serializable dict."""
    for artifact_id, payload in valid_artifact_payloads.items():
        validated = validate_artifact_payload(artifact_id, payload)
        assert isinstance(validated, dict)


def test_validate_artifact_payload_rejects_unknown_artifact():
    """Unknown artifact ids should fail fast."""
    with pytest.raises(ValueError, match="Unknown artifact_id"):
        validate_artifact_payload("unknown_artifact", {})


def test_validate_artifact_payload_rejects_missing_required_fields(
    valid_artifact_payloads: dict[str, dict[str, Any]],
):
    """Artifact contract validation should fail on contract violations."""
    bad = deepcopy(valid_artifact_payloads["validation_report"])
    bad.pop("indicators")
    with pytest.raises(ValidationError):
        validate_artifact_payload("validation_report", bad)


def test_known_input_declaration_requires_an_owned_source_indicator(valid_artifact_payloads):
    bad = deepcopy(valid_artifact_payloads["model"])
    graph_constructs(bad)[0]["usage"] = {"kind": "known_input"}
    with pytest.raises(ValidationError):
        validate_artifact_payload("model", bad)


def test_baseline_report_rejects_extra_fields(
    valid_artifact_payloads: dict[str, dict[str, Any]],
):
    """Extra fields on intervention results should be rejected (extra=forbid)."""
    bad = deepcopy(valid_artifact_payloads["baseline_report"])
    bad["intervention_results"][0]["unknown_field"] = 42
    with pytest.raises(ValidationError):
        validate_artifact_payload("baseline_report", bad)


def test_outcome_enum_no_longer_exists(
    valid_artifact_payloads: dict[str, dict[str, Any]],
):
    """Contracts are pure artifacts: execution failure is a typed exception on
    the transition, never an outcome flag on the payload (extra=forbid)."""
    bad = deepcopy(valid_artifact_payloads["model"])
    bad["outcome"] = "fail"
    with pytest.raises(ValidationError):
        validate_artifact_payload("model", bad)

    stray = deepcopy(valid_artifact_payloads["validation_report"])
    stray["fail_reason"] = "nope"
    with pytest.raises(ValidationError):
        validate_artifact_payload("validation_report", stray)
