"""Tests for causal design schema computed properties and utility functions.

Object-level construction validation (Construct, LatentStructure, Indicator,
MeasurementStructure) is covered by test_schema_validators.py via dict validators.
This file tests CausalDesign composition, computed properties, and utility
functions that are not exercised through dict validation.
"""

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.duration import parse_duration_to_hours
from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.latent_structure import (
    CausalEdge,
    Construct,
    LatentStructure,
    Role,
    TemporalStatus,
)
from nof1_causal_lab.artifacts.measurement_structure import Indicator as IndicatorModel
from nof1_causal_lab.artifacts.measurement_structure import (
    MeasurementStructure,
    WindowExpression,
    check_semantic_collisions,
)
from nof1_causal_lab.utils.observation_semantics import (
    AnchorPolicy,
    SummaryOperator,
    SupportKind,
    derive_indicator_observation_semantics,
)
from tests.helpers import fixture_entity_id


def Indicator(**kwargs: Any) -> IndicatorModel:
    """Build test indicators with the current required schema defaults."""
    kwargs.setdefault("construct_polarity", "positive")
    return IndicatorModel(**kwargs)


class TestConstruct:
    """Tests for Construct validation."""

    def test_outcome_is_not_an_intrinsic_construct_property(self, construct_factory):
        construct = construct_factory("mood")
        assert "is_outcome" not in construct.model_dump()
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            Construct.model_validate({**construct.model_dump(), "is_outcome": True})


class TestLatentStructure:
    """Tests for LatentStructure validation."""

    def test_valid_simple_structure(self, construct_factory):
        """Simple valid structure passes validation."""
        structure = LatentStructure(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS),
            ],
            edges=[
                CausalEdge(
                    cause_id="construct:6b04dc42c531e7091eb8",
                    effect_id="construct:bbc87212909e45b9e6c3",
                    id="edge:48c8a999ded7cb7f54aa",
                    description="Stress affects mood",
                    lagged=False,
                )
            ],
        )
        assert len(structure.constructs) == 2
        assert len(structure.edges) == 1

    def test_invalid_edge_cause_not_in_constructs(self, construct_factory):
        """Edge cause must exist in constructs."""
        with pytest.raises(ValueError, match=r"Edge cause .* not in constructs"):
            LatentStructure(
                constructs=[construct_factory("mood", Role.ENDOGENOUS)],
                edges=[
                    CausalEdge(
                        cause_id="construct:697a57ed87711fa1868c",
                        effect_id="construct:bbc87212909e45b9e6c3",
                        id="edge:1caa163cd3331ce6d038",
                        description="Test edge",
                    )
                ],
            )

    def test_invalid_edge_effect_not_in_constructs(self, construct_factory):
        """Edge effect must exist in constructs."""
        with pytest.raises(ValueError, match=r"Edge effect .* not in constructs"):
            LatentStructure(
                constructs=[
                    construct_factory("stress", Role.EXOGENOUS),
                    construct_factory("mood", Role.ENDOGENOUS),
                ],
                edges=[
                    CausalEdge(
                        cause_id="construct:6b04dc42c531e7091eb8",
                        effect_id="construct:697a57ed87711fa1868c",
                        id="edge:540b90cf355b77543e20",
                        description="Test edge",
                    )
                ],
            )

    def test_invalid_exogenous_cannot_be_effect(self, construct_factory):
        """Exogenous construct cannot be an effect."""
        with pytest.raises(ValueError, match="Exogenous construct 'weather' cannot be an effect"):
            LatentStructure(
                constructs=[
                    construct_factory("mood", Role.ENDOGENOUS),
                    construct_factory("weather", Role.EXOGENOUS),
                ],
                edges=[
                    CausalEdge(
                        cause_id="construct:bbc87212909e45b9e6c3",
                        effect_id="construct:5093142e930692b0545c",
                        id="edge:704be5eb399b1f0d67e4",
                        description="Invalid edge",
                        lagged=False,
                    )
                ],
            )

    def test_valid_exogenous_to_endogenous_contemporaneous(self, construct_factory):
        """Exogenous → endogenous contemporaneous edge is valid."""
        model = LatentStructure(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS),
            ],
            edges=[
                CausalEdge(
                    cause_id="construct:6b04dc42c531e7091eb8",
                    effect_id="construct:bbc87212909e45b9e6c3",
                    id="edge:48c8a999ded7cb7f54aa",
                    description="Contemporaneous exo→endo",
                    lagged=False,
                )
            ],
        )
        assert len(model.edges) == 1
        assert model.edges[0].lagged is False

    def test_invalid_time_varying_to_time_invariant_edge(self, construct_factory):
        """Time-varying constructs cannot cause time-invariant constructs."""
        with pytest.raises(ValueError, match="cannot be a cause of time-invariant construct"):
            LatentStructure(
                constructs=[
                    construct_factory("habit", Role.ENDOGENOUS),
                    construct_factory(
                        "trait",
                        Role.ENDOGENOUS,
                        temporal_status=TemporalStatus.TIME_INVARIANT,
                    ),
                ],
                edges=[
                    CausalEdge(
                        cause_id="construct:a99b82b10cc324c3bc82",
                        effect_id="construct:015bde83e98863c8c393",
                        id="edge:909a6ee7bdb479b858c8",
                        description="Habit changes a fixed trait",
                    )
                ],
            )

    def test_default_query_outcome_does_not_require_incoming_edges(self, construct_factory):
        target = construct_factory("mood")
        structure = LatentStructure(
            default_outcome=ConstructRef(id=target.id), constructs=[target], edges=[]
        )
        assert structure.default_outcome == ConstructRef(id=target.id)

    def test_structure_can_precede_outcome_selection(self, construct_factory):
        structure = LatentStructure(
            constructs=[construct_factory("stress"), construct_factory("mood")], edges=[]
        )
        assert structure.default_outcome is None

    def test_default_query_outcome_must_exist(self, construct_factory):
        with pytest.raises(ValueError, match="unknown construct"):
            LatentStructure(
                constructs=[construct_factory("mood")],
                edges=[],
                default_outcome=ConstructRef(id="construct:unknown"),
            )

    def test_default_query_outcome_must_be_endogenous(self, construct_factory):
        target = construct_factory("weather", Role.EXOGENOUS)
        with pytest.raises(ValueError, match="endogenous"):
            LatentStructure(
                constructs=[target], edges=[], default_outcome=ConstructRef(id=target.id)
            )


class TestIndicator:
    """Tests for Indicator validation."""

    def test_invalid_aggregation(self):
        """Invalid aggregation is rejected."""
        with pytest.raises(ValueError, match="aggregation"):
            Indicator(
                id="indicator:e05e217de7f4442abdc5",
                construct_id="construct:bbc87212909e45b9e6c3",
                name="mood_rating",
                how_to_measure="Extract mood",
                measurement_dtype="continuous",
                aggregation="invalid_agg",
            )

    def test_invalid_measurement_dtype(self):
        """Invalid measurement_dtype is rejected."""
        with pytest.raises(ValueError, match="measurement_dtype"):
            Indicator(
                id="indicator:e05e217de7f4442abdc5",
                construct_id="construct:bbc87212909e45b9e6c3",
                name="mood_rating",
                how_to_measure="Extract mood",
                measurement_dtype="invalid_type",
                aggregation="mean",
            )

    def test_ordinal_requires_levels(self):
        """Ordinal dtype without ordinal_levels is rejected."""
        with pytest.raises(ValueError, match="ordinal_levels is required"):
            Indicator(
                id="indicator:036fd134b9ad32d7a2ca",
                construct_id="construct:bd4a4c688dffc6fe84fd",
                name="pain",
                how_to_measure="Extract pain level",
                measurement_dtype="ordinal",
                aggregation="last",
            )

    def test_ordinal_needs_at_least_two_levels(self):
        """Ordinal with only one level is rejected."""
        with pytest.raises(ValueError, match="at least 2 items"):
            Indicator(
                id="indicator:036fd134b9ad32d7a2ca",
                construct_id="construct:bd4a4c688dffc6fe84fd",
                name="pain",
                how_to_measure="Extract pain level",
                measurement_dtype="ordinal",
                aggregation="last",
                ordinal_levels=["only_one"],
            )

    def test_ordinal_no_duplicate_levels(self):
        """Ordinal with duplicate levels is rejected."""
        with pytest.raises(ValueError, match="duplicate labels"):
            Indicator(
                id="indicator:036fd134b9ad32d7a2ca",
                construct_id="construct:bd4a4c688dffc6fe84fd",
                name="pain",
                how_to_measure="Extract pain level",
                measurement_dtype="ordinal",
                aggregation="last",
                ordinal_levels=["low", "low", "high"],
            )

    def test_ordinal_valid_levels(self):
        """Ordinal with valid levels passes."""
        ind = Indicator(
            id="indicator:036fd134b9ad32d7a2ca",
            construct_id="construct:bd4a4c688dffc6fe84fd",
            name="pain",
            how_to_measure="Extract pain level",
            measurement_dtype="ordinal",
            aggregation="last",
            ordinal_levels=["low", "medium", "high"],
        )
        assert ind.ordinal_levels == ["low", "medium", "high"]

    def test_categorical_requires_levels(self):
        with pytest.raises(ValueError, match="categorical_levels is required"):
            Indicator(
                id="indicator:73fed6c52a41057ef02f",
                construct_id="construct:ca2db375416dbb355458",
                name="location",
                how_to_measure="Extract location",
                measurement_dtype="categorical",
                aggregation="last",
            )

    def test_categorical_needs_at_least_two_levels(self):
        with pytest.raises(ValueError, match="at least 2 items"):
            Indicator(
                id="indicator:73fed6c52a41057ef02f",
                construct_id="construct:ca2db375416dbb355458",
                name="location",
                how_to_measure="Extract location",
                measurement_dtype="categorical",
                aggregation="last",
                categorical_levels=["home"],
            )

    def test_categorical_rejects_duplicate_levels(self):
        with pytest.raises(ValueError, match="duplicate labels"):
            Indicator(
                id="indicator:73fed6c52a41057ef02f",
                construct_id="construct:ca2db375416dbb355458",
                name="location",
                how_to_measure="Extract location",
                measurement_dtype="categorical",
                aggregation="last",
                categorical_levels=["home", "home"],
            )

    def test_non_ordinal_ignores_levels(self):
        """Non-ordinal dtype doesn't require ordinal_levels."""
        ind = Indicator(
            id="indicator:c5b118ae552981435d7b",
            construct_id="construct:61f4ad22907ed956857b",
            name="weight",
            how_to_measure="Extract weight",
            measurement_dtype="continuous",
            aggregation="mean",
        )
        assert ind.ordinal_levels is None

    def test_semantic_default(self):
        """Extraction mode defaults to 'semantic'."""
        ind = Indicator(
            id="indicator:e05e217de7f4442abdc5",
            construct_id="construct:bbc87212909e45b9e6c3",
            name="mood_rating",
            how_to_measure="Extract mood",
            measurement_dtype="continuous",
            aggregation="mean",
        )
        assert ind.extraction_mode == "semantic"

    def test_invalid_extraction_mode(self):
        """Invalid extraction_mode is rejected."""
        with pytest.raises(ValueError, match="extraction_mode"):
            Indicator(
                id="indicator:e05e217de7f4442abdc5",
                construct_id="construct:bbc87212909e45b9e6c3",
                name="mood_rating",
                how_to_measure="Extract mood",
                measurement_dtype="continuous",
                aggregation="mean",
                extraction_mode="invalid",
            )

    def test_computed_valid(self):
        """Computed indicator with single source column and continuous dtype passes."""
        ind = Indicator(
            id="indicator:aa573b5cc0c0a1837e05",
            construct_id="construct:61f4ad22907ed956857b",
            name="avg_heart_rate",
            how_to_measure="Use heart_rate column directly",
            measurement_dtype="continuous",
            aggregation="mean",
            source_columns=["heart_rate"],
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"

    def test_computed_count_dtype(self):
        """Computed indicator with count dtype passes."""
        ind = Indicator(
            id="indicator:a0ce08437c19d06aafd1",
            construct_id="construct:42739985076ec8fdcd0a",
            name="total_steps",
            how_to_measure="Use steps column directly",
            measurement_dtype="count",
            aggregation="sum",
            source_columns=["steps"],
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"

    def test_computed_binary_point_dtype(self):
        """Computed indicator with binary dtype passes for direct point aggregation."""
        ind = Indicator(
            id="indicator:58ba7e8b022133d4764f",
            construct_id="construct:c455f68ba2b4f69e130c",
            name="alarm_state",
            how_to_measure="Use the last observed alarm_state value directly",
            measurement_dtype="binary",
            aggregation="last",
            source_columns=["alarm_state"],
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"

    def test_computed_ordinal_point_dtype(self):
        """Computed indicator with ordinal dtype passes for direct point aggregation."""
        ind = Indicator(
            id="indicator:745132edf4775f59f221",
            construct_id="construct:bbc87212909e45b9e6c3",
            name="mood_label",
            how_to_measure="Use the last observed mood_label value directly",
            measurement_dtype="ordinal",
            aggregation="last",
            ordinal_levels=["bad", "ok", "good"],
            source_columns=["mood_label"],
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"

    def test_computed_categorical_point_dtype(self):
        """Computed indicator with categorical dtype passes for direct point aggregation."""
        ind = Indicator(
            id="indicator:42f1f2a7a4c92ee606b6",
            construct_id="construct:7dc90b263feebe504c2c",
            name="care_setting",
            how_to_measure="Use the first observed care_setting value directly",
            measurement_dtype="categorical",
            aggregation="first",
            categorical_levels=["home", "clinic"],
            source_columns=["care_setting"],
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"

    def test_computed_requires_single_source_column(self):
        """Direct computed indicators with 0 or 2+ source_columns are rejected."""
        with pytest.raises(ValueError, match="exactly 1 direct source_column"):
            Indicator(
                id="indicator:c21b43949b3712e734c8",
                construct_id="construct:61f4ad22907ed956857b",
                name="avg_hr",
                how_to_measure="Use heart_rate",
                measurement_dtype="continuous",
                aggregation="mean",
                source_columns=[],
                extraction_mode="computed",
            )
        with pytest.raises(ValueError, match="exactly 1 direct source_column"):
            Indicator(
                id="indicator:c21b43949b3712e734c8",
                construct_id="construct:61f4ad22907ed956857b",
                name="avg_hr",
                how_to_measure="Compute from systolic and diastolic",
                measurement_dtype="continuous",
                aggregation="mean",
                source_columns=["systolic_bp", "diastolic_bp"],
                extraction_mode="computed",
            )

    def test_computed_rule_allows_multi_source_deterministic_formula(self):
        """Computed rules can reference multiple source columns deterministically."""
        ind = Indicator(
            id="indicator:e33fbf156ca312595e47",
            construct_id="construct:8be7488c482644fc150c",
            name="mean_arterial_pressure",
            how_to_measure="Compute deterministically from systolic and diastolic blood pressure",
            measurement_dtype="continuous",
            aggregation="mean",
            source_columns=["systolic_bp", "diastolic_bp"],
            computed_rule="mean(diastolic_bp + (systolic_bp - diastolic_bp) / 3)",
            extraction_mode="computed",
        )
        assert ind.extraction_mode == "computed"
        assert ind.computed_rule is not None

    def test_computed_rule_rejects_semantic_mode(self):
        """computed_rule is only valid when extraction_mode='computed'."""
        with pytest.raises(ValueError, match="computed_rule but extraction_mode is 'semantic'"):
            Indicator(
                id="indicator:86e4453f8f098e1007ef",
                construct_id="construct:3e67f13db8804f7e0f4c",
                name="low_spo2",
                how_to_measure="Deterministically compute low SpO2 from spo2_pct",
                measurement_dtype="binary",
                aggregation="last",
                source_columns=["spo2_pct"],
                computed_rule="1 if any(spo2_pct < 92) else (0 if count_non_null(spo2_pct) > 0 else None)",
                extraction_mode="semantic",
            )

    def test_computed_rule_rejects_undeclared_source_column(self):
        """computed_rule must reference only declared source_columns."""
        with pytest.raises(ValueError, match="references undeclared source_columns"):
            Indicator(
                id="indicator:0c111e9b74f243fbc086",
                construct_id="construct:c1f9f33e010b2a27f4ed",
                name="glucose_out_of_range",
                how_to_measure="Count out-of-range glucose values deterministically",
                measurement_dtype="count",
                aggregation="sum",
                source_columns=["glucose_mg_dl"],
                computed_rule="None if count_non_null(glucose_mg_dl) == 0 else sum(1 if (glucose_mg_dl < 70 or serum_glucose > 180) else 0)",
                extraction_mode="computed",
            )

    def test_computed_rule_requires_source_reference(self):
        """computed_rule must actually use at least one declared source column."""
        with pytest.raises(ValueError, match="does not reference any source_columns"):
            Indicator(
                id="indicator:4aa7a5f09fd3489f4f2e",
                construct_id="construct:c455f68ba2b4f69e130c",
                name="constant_flag",
                how_to_measure="Always emit a constant flag",
                measurement_dtype="binary",
                aggregation="last",
                source_columns=["spo2_pct"],
                computed_rule="1",
                extraction_mode="computed",
            )

    def test_computed_still_rejects_invalid_semantics(self):
        """Computed indicators still respect the measurement-semantics grid."""
        with pytest.raises(
            ValueError, match="aggregation 'mean' requires measurement_dtype='continuous'"
        ):
            Indicator(
                id="indicator:58ba7e8b022133d4764f",
                construct_id="construct:c455f68ba2b4f69e130c",
                name="alarm_state",
                how_to_measure="Use alarm_state directly",
                measurement_dtype="binary",
                aggregation="mean",
                source_columns=["alarm_state"],
                extraction_mode="computed",
            )


class TestMeasurementStructure:
    """Tests for MeasurementStructure."""

    def test_get_indicators_for_construct(self):
        """get_indicators_for_construct returns correct indicators."""
        model = MeasurementStructure(
            model_clock="1d",
            indicators=[
                Indicator(
                    id="indicator:e05e217de7f4442abdc5",
                    construct_id="construct:bbc87212909e45b9e6c3",
                    name="mood_rating",
                    how_to_measure="Extract mood ratings",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
                Indicator(
                    id="indicator:bc6cf5dba2d9dd23a52b",
                    construct_id="construct:bbc87212909e45b9e6c3",
                    name="mood_text",
                    how_to_measure="Extract mood from text",
                    measurement_dtype="ordinal",
                    aggregation="last",
                    ordinal_levels=["low", "medium", "high"],
                ),
                Indicator(
                    id="indicator:9f3b6ee9465c31d19d78",
                    construct_id="construct:6b04dc42c531e7091eb8",
                    name="stress_level",
                    how_to_measure="Extract stress ratings",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
            ],
        )
        mood_indicators = model.get_indicators_for_construct(fixture_entity_id("construct", "mood"))
        assert len(mood_indicators) == 2
        assert all(
            i.construct_id == fixture_entity_id("construct", "mood") for i in mood_indicators
        )


class TestCausalDesign:
    """Tests for CausalDesign validation."""

    def test_valid_causal_design(self, construct_factory, indicator_factory):
        """Valid CausalDesign passes validation."""
        latent = LatentStructure(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS),
            ],
            edges=[
                CausalEdge(
                    cause_id="construct:6b04dc42c531e7091eb8",
                    effect_id="construct:bbc87212909e45b9e6c3",
                    id="edge:923689028b6b177617c2",
                    description="Stress affects mood",
                )
            ],
        )
        measurement = MeasurementStructure(
            model_clock="1d",
            indicators=[
                indicator_factory("stress_rating", "stress"),
                indicator_factory("mood_rating", "mood"),
            ],
        )
        causal_design = CausalDesign(latent=latent, measurement=measurement)
        assert len(causal_design.latent.constructs) == 2
        assert len(causal_design.measurement.indicators) == 2

    def test_invalid_indicator_references_unknown_construct(
        self, construct_factory, indicator_factory
    ):
        """Indicator must reference a valid construct."""
        latent = LatentStructure(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS),
            ],
            edges=[
                CausalEdge(
                    cause_id="construct:6b04dc42c531e7091eb8",
                    effect_id="construct:bbc87212909e45b9e6c3",
                    id="edge:923689028b6b177617c2",
                    description="Test",
                )
            ],
        )
        measurement = MeasurementStructure(
            model_clock="1d",
            indicators=[
                indicator_factory("mood_rating", "mood"),
                indicator_factory("unknown_indicator", "unknown"),  # invalid reference
            ],
        )
        with pytest.raises(ValueError, match="references unknown construct"):
            CausalDesign(latent=latent, measurement=measurement)

    def test_latent_construct_without_indicator_is_valid(
        self, construct_factory, indicator_factory
    ):
        """Latent constructs without indicators are allowed (A2 deferred to y0)."""
        latent = LatentStructure(
            constructs=[
                construct_factory("stress", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS),
            ],
            edges=[
                CausalEdge(
                    cause_id="construct:6b04dc42c531e7091eb8",
                    effect_id="construct:bbc87212909e45b9e6c3",
                    id="edge:923689028b6b177617c2",
                    description="Test",
                )
            ],
        )
        measurement = MeasurementStructure(
            model_clock="1d",
            indicators=[
                indicator_factory("mood_rating", "mood"),
                # stress has no indicator - it's a latent construct
            ],
        )
        # This should now be valid - y0 will check identification in validation
        causal_design = CausalDesign(latent=latent, measurement=measurement)
        assert len(causal_design.latent.constructs) == 2
        assert len(causal_design.measurement.indicators) == 1

    def test_known_input_cannot_also_be_scientific_only(self, construct_factory, indicator_factory):
        """An authored construct must have exactly one executable disposition."""
        latent = LatentStructure(
            constructs=[
                construct_factory("x", Role.EXOGENOUS),
                construct_factory("y", Role.ENDOGENOUS),
            ],
            edges=[
                CausalEdge(
                    cause_id="construct:2807fb9951f80b57d672",
                    effect_id="construct:8e84a11916a1696aedb7",
                    id="edge:c1324694005b0e3d2d4a",
                    description="Treatment path",
                ),
            ],
        )
        measurement = MeasurementStructure(
            model_clock="1d",
            indicators=[
                indicator_factory("x_obs", "x"),
                indicator_factory("y_obs", "y"),
            ],
        )
        with pytest.raises(ValueError, match="both known inputs and scientific-only"):
            CausalDesign.model_validate(
                {
                    "latent": latent.model_dump(),
                    "measurement": measurement.model_dump(),
                    "known_inputs": [
                        {
                            "construct_id": "construct:2807fb9951f80b57d672",
                            "source_indicator_id": "indicator:27a34f01c0d54e901b81",
                        }
                    ],
                    "scientific_only_constructs": [
                        {"construct_id": "construct:2807fb9951f80b57d672", "reason": "context only"}
                    ],
                }
            )

    def test_lagged_edge_uses_measurement_clock(self, construct_factory, indicator_factory):
        latent = LatentStructure(
            constructs=[
                construct_factory("sleep", Role.EXOGENOUS),
                construct_factory("mood", Role.ENDOGENOUS),
            ],
            edges=[
                CausalEdge(
                    cause_id="construct:cdc0b2958a9512b2abad",
                    effect_id="construct:bbc87212909e45b9e6c3",
                    id="edge:5bac952826a86ec4fb15",
                    description="Sleep affects mood",
                    lagged=True,
                )
            ],
        )
        measurement = MeasurementStructure(
            model_clock="1d",
            indicators=[
                indicator_factory("sleep_hours", "sleep"),
                indicator_factory("mood_rating", "mood"),
            ],
        )
        causal_design = CausalDesign(latent=latent, measurement=measurement)
        assert latent.edges[0].lagged is True
        assert causal_design.measurement.model_clock_hours == 24


class TestParseDurationToHours:
    """Tests for parse_duration_to_hours function."""

    def test_seconds(self):
        assert parse_duration_to_hours("3600s") == 1.0

    def test_minutes(self):
        assert parse_duration_to_hours("60m") == 1.0

    def test_hours(self):
        assert parse_duration_to_hours("4h") == 4.0
        assert parse_duration_to_hours("1h") == 1.0

    def test_days(self):
        assert parse_duration_to_hours("1d") == 24.0
        assert parse_duration_to_hours("7d") == 168.0

    def test_weeks(self):
        assert parse_duration_to_hours("1w") == 168.0
        assert parse_duration_to_hours("2w") == 336.0

    def test_months(self):
        assert parse_duration_to_hours("1mo") == 720.0

    def test_quarters(self):
        assert parse_duration_to_hours("1q") == 2160.0

    def test_years(self):
        assert parse_duration_to_hours("1y") == 8760.0

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration_to_hours("abc")

    def test_invalid_unit(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration_to_hours("5x")

    def test_zero_duration(self):
        with pytest.raises(ValueError, match="positive"):
            parse_duration_to_hours("0d")

    def test_no_number(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            parse_duration_to_hours("d")

    def test_fractional_days(self):
        """model_clock_days property converts correctly."""
        m = MeasurementStructure(
            model_clock="6h",
            indicators=[
                Indicator(
                    id="indicator:1f4c67cecb9238ee1a80",
                    construct_id="construct:311c9047b5ede16a8f26",
                    name="x",
                    how_to_measure="test",
                    measurement_dtype="continuous",
                    aggregation="mean",
                ),
            ],
        )
        assert m.model_clock_hours == 6.0
        assert m.model_clock_days == 0.25

    def test_invalid_model_clock_on_measurement_structure(self):
        """MeasurementStructure rejects invalid model_clock."""
        with pytest.raises(ValueError, match="Invalid duration"):
            MeasurementStructure(
                model_clock="bad",
                indicators=[
                    Indicator(
                        id="indicator:1f4c67cecb9238ee1a80",
                        construct_id="construct:311c9047b5ede16a8f26",
                        name="x",
                        how_to_measure="test",
                        measurement_dtype="continuous",
                        aggregation="mean",
                    ),
                ],
            )


class TestDeriveObservationSemantics:
    """Tests for derive_indicator_observation_semantics."""

    def test_first_maps_to_point_at_window_start(self):
        semantics = derive_indicator_observation_semantics("first", "continuous")
        assert semantics.support_kind == SupportKind.POINT
        assert semantics.summary_operator == SummaryOperator.FIRST
        assert semantics.anchor_policy == AnchorPolicy.SUPPORT_START

    def test_last_maps_to_point_at_window_end(self):
        semantics = derive_indicator_observation_semantics("last", "continuous")
        assert semantics.support_kind == SupportKind.POINT
        assert semantics.summary_operator == SummaryOperator.LAST
        assert semantics.anchor_policy == AnchorPolicy.SUPPORT_END

    def test_interval_summary_operator_maps_to_interval_support(self):
        semantics = derive_indicator_observation_semantics("sum", "count")
        assert semantics.support_kind == SupportKind.INTERVAL
        assert semantics.summary_operator == SummaryOperator.SUM
        assert semantics.anchor_policy == AnchorPolicy.SUPPORT_END

    def test_std_requires_continuous_measurements(self):
        with pytest.raises(
            ValueError, match="aggregation 'std' requires measurement_dtype='continuous'"
        ):
            derive_indicator_observation_semantics("std", "count")

    def test_ordinal_indicators_only_support_point_operators(self):
        with pytest.raises(
            ValueError, match="ordinal indicators currently support only first/last"
        ):
            derive_indicator_observation_semantics("mean", "ordinal")

    def test_unsupported_aggregations_fail_fast(self):
        with pytest.raises(ValueError, match="not yet supported by the measurement structure"):
            derive_indicator_observation_semantics("median", "continuous")


class TestSemanticCollisions:
    """Tests for check_semantic_collisions function."""

    def test_count_text_mean_agg_collision(self):
        """'count' in how_to_measure + mean aggregation → warning."""
        warnings = check_semantic_collisions("Count the number of exercise sessions", "mean")
        assert len(warnings) >= 1
        assert "counting" in warnings[0].lower() or "count" in warnings[0].lower()

    def test_no_collision(self):
        """Consistent text and aggregation → no warnings."""
        warnings = check_semantic_collisions("Average daily mood rating", "mean")
        assert len(warnings) == 0

    def test_total_text_mean_agg_collision(self):
        """'total' in text + mean aggregation → warning."""
        warnings = check_semantic_collisions("Total steps walked during the day", "mean")
        assert len(warnings) >= 1

    def test_last_text_sum_agg_collision(self):
        """'most recent' in text + sum aggregation → warning."""
        warnings = check_semantic_collisions("The most recent blood pressure reading", "sum")
        assert len(warnings) >= 1


class TestIndicatorObservationSemantics:
    """Tests for Indicator computed observation semantics."""

    def test_interval_indicator_serializes_semantics(self, indicator_factory):
        ind = indicator_factory("steps", "activity", aggregation="sum", dtype="count")
        assert ind.support_kind == SupportKind.INTERVAL
        assert ind.summary_operator == SummaryOperator.SUM
        assert ind.anchor_policy == AnchorPolicy.SUPPORT_END
        assert ind.requires_interval_summary_measurement is True

    def test_point_indicator_serializes_semantics(self, indicator_factory):
        ind = indicator_factory("last_bp", "bp", aggregation="last", dtype="continuous")
        assert ind.support_kind == SupportKind.POINT
        assert ind.summary_operator == SummaryOperator.LAST
        assert ind.anchor_policy == AnchorPolicy.SUPPORT_END
        assert ind.requires_interval_summary_measurement is False

    def test_ordinal_indicator_uses_point_semantics(self, indicator_factory):
        ind = indicator_factory("pain_level", "pain", aggregation="last", dtype="ordinal")
        assert ind.support_kind == SupportKind.POINT
        assert ind.summary_operator == SummaryOperator.LAST
        assert ind.anchor_policy == AnchorPolicy.SUPPORT_END

    def test_unsupported_aggregation_is_rejected_on_indicator(self, indicator_factory):
        with pytest.raises(ValueError, match="not yet supported by the measurement structure"):
            indicator_factory("median_hr", "hr", aggregation="median", dtype="continuous")

    def test_ordinal_interval_summary_is_rejected_on_indicator(self, indicator_factory):
        with pytest.raises(
            ValueError, match="ordinal indicators currently support only first/last"
        ):
            indicator_factory("pain_level", "pain", aggregation="mean", dtype="ordinal")


class TestIndicatorObservationWindow:
    def test_valid_observation_window(self):
        indicator = Indicator(
            id="indicator:b41c85c254676b4bc588",
            construct_id="construct:bbc87212909e45b9e6c3",
            name="monthly_mood",
            how_to_measure="Average mood over the last month",
            measurement_dtype="continuous",
            aggregation="mean",
            observation_window="1mo",
        )

        assert indicator.observation_window == "1mo"

    def test_invalid_observation_window(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            Indicator(
                id="indicator:b41c85c254676b4bc588",
                construct_id="construct:bbc87212909e45b9e6c3",
                name="monthly_mood",
                how_to_measure="Average mood over the last month",
                measurement_dtype="continuous",
                aggregation="mean",
                observation_window="monthly",
            )


@pytest.mark.parametrize(
    "expression",
    ["mean(values", "values.mean()", "unknown_helper(values)", {"window_expr": "mean(values)"}],
)
def test_window_expression_is_validated_at_the_scalar_boundary(expression):
    with pytest.raises(ValidationError):
        TypeAdapter(WindowExpression).validate_python(expression)


def test_window_expression_serializes_without_a_wrapper():
    adapter = TypeAdapter(WindowExpression)
    expression = adapter.validate_python("mean(values)")
    assert adapter.dump_json(expression) == b'"mean(values)"'
