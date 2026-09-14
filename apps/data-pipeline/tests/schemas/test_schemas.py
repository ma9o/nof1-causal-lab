"""Tests for causal design schema computed properties and utility functions.

Object-level construction validation (Construct, ModelSpec, Indicator,
MeasurementStructure) is covered by test_schema_validators.py via dict validators.
This file tests CausalDesign composition, computed properties, and utility
functions that are not exercised through dict validation.
"""

from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.construct import (
    CausalEdge,
    Construct,
    Role,
    TemporalStatus,
    replace_constructs,
)
from nof1_causal_lab.artifacts.duration import parse_duration_to_hours
from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.indicator import Indicator as IndicatorModel
from nof1_causal_lab.artifacts.indicator import WindowExpression, check_semantic_collisions
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.utils.observation_semantics import (
    AnchorPolicy,
    SummaryOperator,
    SupportKind,
    derive_indicator_observation_semantics,
)
from tests.helpers import graph_constructs, make_model


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


class TestModel:
    """Scientific membership follows the endpoints of one connected graph."""

    def test_valid_simple_structure(self, construct_factory):
        edge = CausalEdge(
            id="edge:stress-mood",
            cause=construct_factory("stress", Role.EXOGENOUS),
            effect=construct_factory("mood"),
            description="Stress affects mood",
            lagged=False,
        )
        model = ModelSpec(edges=(edge,))
        assert [construct.name for construct in model.constructs] == ["stress", "mood"]
        assert model.edges[0].cause is model.get_construct(edge.cause.id)

    @pytest.mark.parametrize("endpoint", ["cause", "effect"])
    def test_undefined_endpoint_reference_is_rejected(self, endpoint):
        payload = make_model(["stress", "mood"], [("stress", "mood")]).model_dump(mode="json")
        payload["edges"][0][endpoint] = {"kind": "construct", "id": "construct:unknown"}
        with pytest.raises(ValueError, match="Undefined construct endpoint"):
            ModelSpec.model_validate(payload)

    def test_invalid_exogenous_cannot_be_effect(self, construct_factory):
        with pytest.raises(ValueError, match="Exogenous construct 'weather' cannot be an effect"):
            ModelSpec(
                edges=(
                    CausalEdge(
                        id="edge:mood-weather",
                        cause=construct_factory("mood"),
                        effect=construct_factory("weather", Role.EXOGENOUS),
                        description="Invalid edge",
                    ),
                )
            )

    def test_invalid_time_varying_to_time_invariant_edge(self, construct_factory):
        with pytest.raises(ValueError, match="cannot be a cause of time-invariant construct"):
            ModelSpec(
                edges=(
                    CausalEdge(
                        id="edge:habit-trait",
                        cause=construct_factory("habit"),
                        effect=construct_factory(
                            "trait", temporal_status=TemporalStatus.TIME_INVARIANT
                        ),
                        description="Habit changes a fixed trait",
                    ),
                )
            )

    def test_default_query_outcome_does_not_require_incoming_edges(self):
        model = make_model(["mood", "sleep"], [("mood", "sleep")])
        reference = ConstructRef(id=model.edges[0].cause.id)
        assert model.revised(default_outcome=reference).default_outcome == reference

    def test_structure_can_precede_outcome_selection(self):
        assert make_model(["stress", "mood"], [("stress", "mood")]).default_outcome is None

    def test_default_query_outcome_must_exist(self):
        with pytest.raises(ValueError, match="unknown construct"):
            make_model(["mood"]).revised(default_outcome=ConstructRef(id="construct:unknown"))

    def test_default_query_outcome_must_be_endogenous(self, construct_factory):
        target = construct_factory("weather", Role.EXOGENOUS)
        with pytest.raises(ValueError, match="endogenous"):
            ModelSpec(
                edges=(
                    CausalEdge(
                        id="edge:weather-mood",
                        cause=target,
                        effect=construct_factory("mood"),
                        description="Weather affects mood",
                    ),
                ),
                default_outcome=ConstructRef(id=target.id),
            )


class TestIndicator:
    """Tests for Indicator validation."""

    def test_invalid_aggregation(self):
        """Invalid aggregation is rejected."""
        with pytest.raises(ValueError, match="aggregation"):
            Indicator(
                id="indicator:e05e217de7f4442abdc5",
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
            name="pain",
            how_to_measure="Extract pain level",
            measurement_dtype="ordinal",
            aggregation="last",
            ordinal_levels=["low", "medium", "high"],
        )
        assert ind.ordinal_levels == ("low", "medium", "high")

    def test_categorical_requires_levels(self):
        with pytest.raises(ValueError, match="categorical_levels is required"):
            Indicator(
                id="indicator:73fed6c52a41057ef02f",
                name="location",
                how_to_measure="Extract location",
                measurement_dtype="categorical",
                aggregation="last",
            )

    def test_categorical_needs_at_least_two_levels(self):
        with pytest.raises(ValueError, match="at least 2 items"):
            Indicator(
                id="indicator:73fed6c52a41057ef02f",
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
                name="alarm_state",
                how_to_measure="Use alarm_state directly",
                measurement_dtype="binary",
                aggregation="mean",
                source_columns=["alarm_state"],
                extraction_mode="computed",
            )


class TestModelContainment:
    def test_construct_owns_its_indicators(self):
        model = make_model(["mood", "stress"], [("mood", "stress")])
        mood, stress = model.constructs
        assert model.get_construct(mood.id).indicators == mood.indicators
        assert model.indicator_owner(mood.indicators[0].id) is mood
        assert model.indicator_owner(stress.indicators[0].id) is stress
        assert "construct_id" not in mood.indicators[0].model_dump()

    def test_indicator_cannot_have_an_independent_unknown_owner(self):
        model = make_model(["mood"])
        value = model.model_dump(mode="json")
        graph_constructs(value)[0]["indicators"][0]["construct_id"] = "construct:unknown"
        with pytest.raises(ValidationError, match="Extra inputs"):
            ModelSpec.model_validate(value)

    def test_latent_construct_without_indicators_is_valid(self):
        model = make_model(["observed", "latent"], [("observed", "latent")])
        observed, latent = model.constructs
        result = model.revised(
            edges=replace_constructs(
                model.edges, (observed, latent.model_copy(update={"indicators": ()}))
            )
        )
        assert result.get_construct(latent.id).indicators == ()

    def test_usage_is_one_choice_and_requires_its_own_indicator(self):
        model = make_model(["X", "Y"], [("X", "Y")])
        data = model.model_dump(mode="json")
        graph_constructs(data)[0]["usage"] = {
            "kind": "known_input",
            "source_indicator_id": model.constructs[1].indicators[0].id,
        }
        with pytest.raises(ValueError, match="same construct"):
            ModelSpec.model_validate(data)
        graph_constructs(data)[0]["usage"] = [{"kind": "scientific_only", "reason": "context"}]
        with pytest.raises(ValidationError):
            ModelSpec.model_validate(data)

    def test_lagged_edge_uses_the_canonical_model_clock(self):
        model = make_model(["sleep", "mood"], [("sleep", "mood")]).revised(measurement_clock="6h")
        assert model.edges[0].lagged
        assert model.model_clock_days == 0.25


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
        assert make_model(["X"]).revised(measurement_clock="6h").model_clock_days == 0.25

    def test_invalid_model_clock(self):
        with pytest.raises(ValueError, match="Invalid duration"):
            make_model(["X"]).revised(measurement_clock="bad")


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
        ind = indicator_factory("steps", aggregation="sum", dtype="count")
        assert ind.support_kind == SupportKind.INTERVAL
        assert ind.summary_operator == SummaryOperator.SUM
        assert ind.anchor_policy == AnchorPolicy.SUPPORT_END
        assert ind.requires_interval_summary_measurement is True

    def test_point_indicator_serializes_semantics(self, indicator_factory):
        ind = indicator_factory("last_bp", aggregation="last", dtype="continuous")
        assert ind.support_kind == SupportKind.POINT
        assert ind.summary_operator == SummaryOperator.LAST
        assert ind.anchor_policy == AnchorPolicy.SUPPORT_END
        assert ind.requires_interval_summary_measurement is False

    def test_ordinal_indicator_uses_point_semantics(self, indicator_factory):
        ind = indicator_factory("pain_level", aggregation="last", dtype="ordinal")
        assert ind.support_kind == SupportKind.POINT
        assert ind.summary_operator == SummaryOperator.LAST
        assert ind.anchor_policy == AnchorPolicy.SUPPORT_END

    def test_unsupported_aggregation_is_rejected_on_indicator(self, indicator_factory):
        with pytest.raises(ValueError, match="not yet supported by the measurement structure"):
            indicator_factory("median_hr", aggregation="median", dtype="continuous")

    def test_ordinal_interval_summary_is_rejected_on_indicator(self, indicator_factory):
        with pytest.raises(
            ValueError, match="ordinal indicators currently support only first/last"
        ):
            indicator_factory("pain_level", aggregation="mean", dtype="ordinal")


class TestIndicatorObservationWindow:
    def test_valid_observation_window(self):
        indicator = Indicator(
            id="indicator:b41c85c254676b4bc588",
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
