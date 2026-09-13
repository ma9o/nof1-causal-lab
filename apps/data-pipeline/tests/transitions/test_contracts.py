"""Tests for artifact payload contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from nof1_causal_lab.flows.context_tools import CONTEXT_TOOLS
from tests.artifact_contract_support import validate_artifact_payload


def _saved_query():
    from nof1_causal_lab.artifacts.scenarios import ScenarioDefinition, ScenarioQuery

    return ScenarioQuery.from_definition(
        ScenarioDefinition.model_validate(
            {
                "start": {"kind": "baseline"},
                "clamps": [
                    {
                        "variable": "Stress",
                        "target": {"kind": "construct", "id": "construct:stress"},
                        "mode": "shift",
                        "amount": -0.5,
                    }
                ],
                "outcome": {"kind": "construct", "id": "construct:outcome"},
                "readout": {},
            }
        )
    ).model_dump(mode="json")


@pytest.fixture
def valid_artifact_payloads() -> dict[str, dict[str, Any]]:
    """Minimal valid payload for each persisted artifact id."""
    return {
        "raw_data": {
            "column_descriptions": [
                {"name": "date", "description": "Date of observation"},
                {"name": "value", "description": "Numeric value"},
                {"name": "category", "description": "Category label"},
            ],
        },
        "latent_structure": {
            "latent_structure": {
                "default_outcome": {"kind": "construct", "id": "construct:dc6723ce621183cd1182"},
                "constructs": [
                    {
                        "id": "construct:dc6723ce621183cd1182",
                        "name": "Perf",
                        "description": "Performance",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                    },
                    {
                        "id": "construct:98df502ac4daf088ca29",
                        "name": "Stress",
                        "description": "Stress level",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                    },
                ],
                "edges": [
                    {
                        "cause_id": "construct:98df502ac4daf088ca29",
                        "effect_id": "construct:dc6723ce621183cd1182",
                        "id": "edge:399745ac3ae059a06462",
                        "description": "Stress reduces performance",
                        "lagged": True,
                    }
                ],
            },
        },
        "measurement_structure": {
            "measurement_structure": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "id": "indicator:3696aef3ff6f446744e5",
                        "construct_id": "construct:98df502ac4daf088ca29",
                        "name": "stress_score",
                        "construct_polarity": "positive",
                        "how_to_measure": "Self-reported stress",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    }
                ],
            },
            "known_inputs": [],
            "scientific_only_constructs": [],
        },
        "measurements": {
            "workers": [
                {
                    "worker_id": 0,
                    "status": "completed",
                    "n_extractions": 3,
                    "n_windows": 7,
                }
            ],
        },
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
        "statistical_model_spec": {
            "statistical_model_spec": {
                "mechanisms": [
                    {
                        "kind": "node_potential",
                        "target_id": "construct:98df502ac4daf088ca29",
                        "center": {"kind": "fixed", "value": 0},
                        "stiffness": {
                            "kind": "estimated",
                            "parameter_id": "parameter:067ff49138696d741faffe7e2dc6684225435e905b47cba9dbd3b216d7bbe749",
                        },
                        "quartic": {"kind": "fixed", "value": 0},
                    }
                ],
                "likelihoods": [
                    {
                        "indicator_id": "indicator:3696aef3ff6f446744e5",
                        "distribution": "gaussian",
                        "link": "identity",
                        "reasoning": "continuous variable",
                    }
                ],
                "parameters": [
                    {
                        "prior_transform": "dt_persistence_to_ct_decay",
                        "id": "parameter:067ff49138696d741faffe7e2dc6684225435e905b47cba9dbd3b216d7bbe749",
                        "owners": [{"kind": "construct", "id": "construct:98df502ac4daf088ca29"}],
                        "quantity": "dynamics_decay",
                        "name": "rho_Stress",
                        "role": "ar_coefficient",
                        "constraint": "unit_interval",
                        "description": "AR coefficient",
                        "prior": {
                            "distribution": "Beta",
                            "params": {"concentration1": 2.0, "concentration0": 3.0},
                        },
                        "prior_reasoning": "Weakly informative",
                    }
                ],
            },
            "prior_predictive_samples": {"indicator:3696aef3ff6f446744e5": [0.1, -0.2, 0.3]},
        },
        "posterior": {
            "draws": {"n_draws": 1000, "parameter_shapes": {"theta": [1]}, "latent_shape": [3, 1]},
            "provenance": {
                "causal_design": {"workspace_id": "TEST", "version": 1},
                "compiled_ssm_version": 1,
                "panel_version": 1,
            },
            "inference_metadata": {
                "method": "marginal_particle_gibbs",
                "n_samples": 1000,
                "duration_seconds": 1.2,
            },
            "assessment": {
                "ppc": {
                    "per_variable_warnings": [],
                    "checked": True,
                    "overlays": [],
                    "test_stats": [],
                },
            },
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
            "saved_scenarios": [
                {
                    "label": "Stress shift",
                    "query": _saved_query(),
                }
            ],
            "final_summary": "Stress reduction remains the dominant actionable lever.",
        },
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
    bad = deepcopy(valid_artifact_payloads["measurements"])
    bad.pop("workers")
    with pytest.raises(ValidationError):
        validate_artifact_payload("measurements", bad)


def test_measurement_structure_requires_known_input_decision(
    valid_artifact_payloads: dict[str, dict[str, Any]],
):
    """The authored projection decision cannot disappear through a silent default."""
    bad = deepcopy(valid_artifact_payloads["measurement_structure"])
    bad.pop("known_inputs")
    with pytest.raises(ValidationError):
        validate_artifact_payload("measurement_structure", bad)


def test_baseline_report_rejects_extra_fields(
    valid_artifact_payloads: dict[str, dict[str, Any]],
):
    """Extra fields on intervention results should be rejected (extra=forbid)."""
    bad = deepcopy(valid_artifact_payloads["baseline_report"])
    bad["intervention_results"][0]["unknown_field"] = 42
    with pytest.raises(ValidationError):
        validate_artifact_payload("baseline_report", bad)


def test_saved_scenarios_reject_extra_fields(
    valid_artifact_payloads: dict[str, dict[str, Any]],
):
    """Saved scenario entries should remain schema-checked."""
    bad = deepcopy(valid_artifact_payloads["baseline_report"])
    bad["saved_scenarios"][0]["unknown_field"] = 42
    with pytest.raises(ValidationError):
        validate_artifact_payload("baseline_report", bad)


def test_outcome_enum_no_longer_exists(
    valid_artifact_payloads: dict[str, dict[str, Any]],
):
    """Contracts are pure artifacts: execution failure is a typed exception on
    the transition, never an outcome flag on the payload (extra=forbid)."""
    bad = deepcopy(valid_artifact_payloads["measurement_structure"])
    bad["outcome"] = "fail"
    with pytest.raises(ValidationError):
        validate_artifact_payload("measurement_structure", bad)

    stray = deepcopy(valid_artifact_payloads["raw_data"])
    stray["fail_reason"] = "nope"
    with pytest.raises(ValidationError):
        validate_artifact_payload("raw_data", stray)
