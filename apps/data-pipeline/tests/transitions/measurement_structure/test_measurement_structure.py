"""Test measurement-structure measurement structure proposal and related assembly helpers."""

from typing import Any

import pytest

from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.latent_structure import LatentStructure
from nof1_causal_lab.flows.transitions.measurement_structure.assemble import build_causal_design
from nof1_causal_lab.models.ssm.compile.artifact import (
    validate_measurement_structure_for_compilation,
)
from nof1_causal_lab.models.structural import build_structural_plan
from nof1_causal_lab.utils.causal_design import get_outcome_name
from nof1_causal_lab.utils.structural_plan import get_edges, get_known_inputs, get_state_names
from tests.helpers import fixture_entity_id


def _assert_same_declared_measurement(
    actual: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert actual["model_clock"] == expected["model_clock"]
    assert [item["name"] for item in actual["indicators"]] == [
        item["name"] for item in expected["indicators"]
    ]
    assert [item["construct_id"] for item in actual["indicators"]] == [
        item["construct_id"] for item in expected["indicators"]
    ]


def _measurement_compile_feedback(
    measurement_structure: dict[str, Any],
    latent_structure: dict[str, Any],
) -> str | None:
    _, errors = validate_measurement_structure_for_compilation(
        measurement_structure,
        LatentStructure.model_validate(latent_structure),
    )
    return "\n".join(errors) if errors else None


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: Measurement Compiler
# ══════════════════════════════════════════════════════════════════════════════


class TestMeasurementCompiler:
    """Test compiler-level measurement validation used in measurement-structure."""

    def test_valid_measurement_returns_none(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """A valid measurement structure compiles cleanly."""
        result = _measurement_compile_feedback(
            stage1b_measurement_all_observed, stage1b_simple_latent
        )
        assert result is None

    def test_missing_outcome_indicator_returns_error(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """Outcome coverage is enforced at compile time."""
        measurement = {
            "model_clock": "1d",
            "indicators": [
                indicator
                for indicator in stage1b_measurement_all_observed["indicators"]
                if indicator["construct_id"] != fixture_entity_id("construct", "Outcome")
            ],
        }

        result = _measurement_compile_feedback(measurement, stage1b_simple_latent)

        assert result is not None
        assert "Outcome construct 'Outcome'" in result

    def test_duplicate_operationalization_returns_error(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """Identical indicators for the same construct are rejected."""
        measurement = {
            "model_clock": "1d",
            "indicators": [
                stage1b_measurement_all_observed["indicators"][0],
                {
                    "id": "indicator:ff9b857400b83cf20b8e",
                    "construct_id": "construct:2219a835484dea8b586a",
                    "name": "treatment_dose_copy",
                    "construct_polarity": "positive",
                    "how_to_measure": "Extract the treatment dosage from the data",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                stage1b_measurement_all_observed["indicators"][1],
            ],
        }

        result = _measurement_compile_feedback(measurement, stage1b_simple_latent)

        assert result is not None
        assert "duplicate indicator operationalizations" in result

    def test_semantic_collision_returns_error(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """Compiler surfaces aggregation/measurement semantic mismatches."""
        measurement = {
            "model_clock": "1d",
            "indicators": [
                {
                    "id": "indicator:a87ef06a71c2a47d6b81",
                    "construct_id": "construct:2219a835484dea8b586a",
                    "name": "treatment_count",
                    "construct_polarity": "positive",
                    "how_to_measure": "Count the number of treatments administered",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                stage1b_measurement_all_observed["indicators"][1],
            ],
        }

        result = _measurement_compile_feedback(measurement, stage1b_simple_latent)

        assert result is not None
        assert "Semantic collision" in result


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: Compute functions
# ══════════════════════════════════════════════════════════════════════════════


class TestStage1bGrounding:
    """Test the grounding function directly."""

    def test_build_causal_design_round_trips_schema(
        self, stage1b_simple_latent, stage1b_measurement_all_observed
    ):
        """build_causal_design() returns a valid CausalDesign with the expected outcome."""
        spec = build_causal_design(
            stage1b_simple_latent,
            stage1b_measurement_all_observed,
            identifiability_status={
                "identifiable_treatments": {
                    "Treatment": {
                        "method": "do_calculus",
                        "estimand": "P(Outcome|do(Treatment))",
                        "marginalized_confounders": [],
                        "instruments": [],
                    }
                },
                "non_identifiable_treatments": {},
            },
            known_inputs=[],
            scientific_only_constructs=[],
        )

        validated = CausalDesign.model_validate(spec)
        assert len(validated.latent.constructs) == 2
        assert len(validated.measurement.indicators) == 2
        plan = build_structural_plan(validated)
        assert set(get_state_names(plan)) == {"Treatment", "Outcome"}
        assert get_outcome_name(spec.latent.model_dump(mode="json")) == "Outcome"

    def test_valid_known_input_is_authored_into_structural_plan(
        self,
        stage1b_simple_latent,
        stage1b_measurement_all_observed,
    ):
        """A valid declaration survives grounding and drives the projection."""
        from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
            measurement_structure_grounding,
        )

        proposal = {
            **stage1b_measurement_all_observed,
            "known_inputs": [
                {
                    "construct_id": "construct:2219a835484dea8b586a",
                    "source_indicator_id": "indicator:0d729b16859bd5e9357e",
                }
            ],
            "scientific_only_constructs": [],
        }

        output, feedback = measurement_structure_grounding(proposal, stage1b_simple_latent)

        assert feedback == "VALID"
        assert output is not None
        assert output["known_inputs"] == [
            {
                "construct_id": "construct:2219a835484dea8b586a",
                "source_indicator_id": "indicator:0d729b16859bd5e9357e",
                "scale": 1.0,
                "missing_policy": "zero",
            }
        ]

        causal_design = build_causal_design(
            stage1b_simple_latent,
            output["measurement_structure"],
            known_inputs=output["known_inputs"],
            scientific_only_constructs=output["scientific_only_constructs"],
        )
        plan = build_structural_plan(causal_design)
        assert get_state_names(plan) == ["Outcome"]
        [known_input] = get_known_inputs(plan)
        assert {
            key: known_input[key]
            for key in ("construct_id", "source_indicator_id", "scale", "missing_policy")
        } == output["known_inputs"][0]
        assert known_input["source_id"].startswith("known_input:")
        assert known_input["construct_id"].startswith("construct:")
        assert known_input["source_indicator_id"].startswith("indicator:")
        assert [(edge["cause"], edge["effect"]) for edge in get_edges(plan)] == [
            ("Treatment", "Outcome")
        ]

    def test_known_input_source_must_measure_declared_construct(
        self,
        stage1b_simple_latent,
        stage1b_measurement_all_observed,
    ):
        """Grounding rejects a source indicator attached to another construct."""
        from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
            measurement_structure_grounding,
        )

        proposal = {
            **stage1b_measurement_all_observed,
            "known_inputs": [
                {
                    "construct_id": "construct:2219a835484dea8b586a",
                    "source_indicator_id": "indicator:54a6e8d16cbce15b4427",
                }
            ],
            "scientific_only_constructs": [],
        }

        output, feedback = measurement_structure_grounding(proposal, stage1b_simple_latent)

        assert output is None
        assert "source_indicator must measure the same construct" in feedback

    def test_grounding_requires_explicit_known_inputs(
        self,
        stage1b_simple_latent,
        stage1b_measurement_all_observed,
    ):
        """Omitting the authored decision is rejected rather than defaulted."""
        from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
            measurement_structure_grounding,
        )

        proposal = {
            key: value
            for key, value in stage1b_measurement_all_observed.items()
            if key != "known_inputs"
        }

        output, feedback = measurement_structure_grounding(proposal, stage1b_simple_latent)

        assert output is None
        assert "'known_inputs' must be a list" in feedback

    def test_valid_identifiable(self, stage1b_simple_latent, stage1b_measurement_all_observed):
        """Valid measurement structure returns VALID feedback and context_output."""
        from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
            measurement_structure_grounding,
        )

        output, feedback = measurement_structure_grounding(
            stage1b_measurement_all_observed, stage1b_simple_latent
        )

        assert feedback == "VALID"
        assert output is not None
        _assert_same_declared_measurement(
            output["measurement_structure"],
            stage1b_measurement_all_observed,
        )

    def test_valid_confounded_measurement_still_validates(
        self, stage1b_confounded_latent, stage1b_measurement_all_observed
    ):
        """Identifiability is not checked inside the measurement proposal."""
        from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
            measurement_structure_grounding,
        )

        output, feedback = measurement_structure_grounding(
            stage1b_measurement_all_observed, stage1b_confounded_latent
        )

        assert output is not None
        _assert_same_declared_measurement(
            output["measurement_structure"],
            stage1b_measurement_all_observed,
        )
        assert feedback == "VALID"

    def test_lagged_time_varying_query_uses_prior_timestep(self):
        """Lagged X->Y effects are checked as X_{t-1}->Y_t, not X_t->Y_t."""
        from nof1_causal_lab.utils.identifiability import check_identifiability

        latent_structure = {
            "default_outcome": {"kind": "construct", "id": "construct:e8c8ad65bcd90a0245ee"},
            "constructs": [
                {
                    "id": "construct:259490b93f7f2c38ed40",
                    "name": "Sleep",
                    "description": "Sleep quality",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "id": "construct:e8c8ad65bcd90a0245ee",
                    "name": "Mood",
                    "description": "Mood state",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "id": "construct:7e2e66cf0a7658ce50f9",
                    "name": "Chronotype",
                    "description": "Unobserved stable circadian preference",
                    "role": "exogenous",
                    "temporal_status": "time_invariant",
                },
            ],
            "edges": [
                {
                    "cause_id": "construct:259490b93f7f2c38ed40",
                    "effect_id": "construct:e8c8ad65bcd90a0245ee",
                    "id": "edge:a0557344502eff589862",
                    "description": "Better sleep improves later mood",
                    "lagged": True,
                },
                {
                    "cause_id": "construct:7e2e66cf0a7658ce50f9",
                    "effect_id": "construct:259490b93f7f2c38ed40",
                    "id": "edge:8fe257111b4a566bd857",
                    "description": "Chronotype shifts sleep quality",
                    "lagged": False,
                },
                {
                    "cause_id": "construct:7e2e66cf0a7658ce50f9",
                    "effect_id": "construct:e8c8ad65bcd90a0245ee",
                    "id": "edge:54d7f3c4fd0babe323da",
                    "description": "Chronotype shifts mood vulnerability",
                    "lagged": False,
                },
            ],
        }
        measurement_structure = {
            "model_clock": "1d",
            "indicators": [
                {
                    "id": "indicator:7f807162156d3eb1b611",
                    "construct_id": "construct:259490b93f7f2c38ed40",
                    "name": "sleep_score",
                    "construct_polarity": "positive",
                    "how_to_measure": "Extract the sleep score",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "source_columns": ["sleep_score"],
                },
                {
                    "id": "indicator:45f78731e3e0c6f3efe1",
                    "construct_id": "construct:e8c8ad65bcd90a0245ee",
                    "name": "mood_score",
                    "construct_polarity": "positive",
                    "how_to_measure": "Extract the mood score",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "source_columns": ["mood_score"],
                },
            ],
        }

        result = check_identifiability(latent_structure, measurement_structure)
        non_identifiable = result["non_identifiable_treatments"]
        assert non_identifiable["Sleep"]["confounders"] == ["Chronotype"]

    def test_invalid_schema(self, stage1b_simple_latent):
        """Invalid schema returns None context_output."""
        from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
            measurement_structure_grounding,
        )

        output, feedback = measurement_structure_grounding(
            {"indicators": [{"bad": "data"}]}, stage1b_simple_latent
        )

        assert output is None
        assert "VALIDATION ERRORS" in feedback

    def test_projects_unmeasured_constructs_from_structural_plan(self):
        """Latent-only constructs should not remain in the executable state vector."""
        latent_structure = {
            "default_outcome": {"kind": "construct", "id": "construct:170504d1ef8631dda85d"},
            "constructs": [
                {
                    "id": "construct:2219a835484dea8b586a",
                    "name": "Treatment",
                    "description": "Observed treatment",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "id": "construct:c10d32fa64fe96b49b0a",
                    "name": "Mediator",
                    "description": "Unmeasured mediator",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "id": "construct:170504d1ef8631dda85d",
                    "name": "Outcome",
                    "description": "Observed outcome",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
            ],
            "edges": [
                {
                    "cause_id": "construct:2219a835484dea8b586a",
                    "effect_id": "construct:c10d32fa64fe96b49b0a",
                    "id": "edge:29f939f9145895d08640",
                    "description": "Treatment shifts mediator",
                },
                {
                    "cause_id": "construct:c10d32fa64fe96b49b0a",
                    "effect_id": "construct:170504d1ef8631dda85d",
                    "id": "edge:8879b5ba3593f2926c48",
                    "description": "Mediator shifts outcome",
                },
            ],
        }
        measurement_structure = {
            "model_clock": "1d",
            "indicators": [
                {
                    "id": "indicator:5b1771032d20744366ec",
                    "construct_id": "construct:2219a835484dea8b586a",
                    "name": "treatment_signal",
                    "construct_polarity": "positive",
                    "how_to_measure": "Use the treatment column directly",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "id": "indicator:a1e2398902f03e507998",
                    "construct_id": "construct:170504d1ef8631dda85d",
                    "name": "outcome_signal",
                    "construct_polarity": "positive",
                    "how_to_measure": "Use the outcome column directly",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
        }

        causal_design = build_causal_design(
            latent_structure,
            measurement_structure,
            {
                "identifiable_treatments": {},
                "non_identifiable_treatments": {},
            },
            known_inputs=[],
            scientific_only_constructs=[],
        )

        plan = build_structural_plan(causal_design)
        assert get_state_names(plan) == ["Treatment", "Outcome"]
        assert get_edges(plan) == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
