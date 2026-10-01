"""Guard tests for the per-construct identification-anchor invariant.

Every retained construct must have exactly one location anchor and one scale
anchor (docs/assumptions.md#parameter-anchors). These tests
enumerate the family/policy combinations so that any future eligibility change
that reopens an exact likelihood ridge fails at construction.
"""

from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.likelihood import (
    DistributionFamily,
    LinkFunction,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.support import NumericalSupportError
from nof1_causal_lab.models.ssm.likelihood_extra_params import assemble_sampled_extra_params
from tests.helpers import fixture_entity_id, make_model

pytestmark = pytest.mark.contract

# ═══════════════════════════════════════════════════════════════════════
# Fixture builders
# ═══════════════════════════════════════════════════════════════════════


def _indicator(
    name: str,
    construct_name: str,
    dtype: str,
    *,
    polarity: str = "positive",
) -> dict[str, Any]:
    indicator = {
        "id": fixture_entity_id("indicator", name),
        "construct_id": fixture_entity_id("construct", construct_name),
        "name": name,
        "construct_polarity": polarity,
        "measurement_dtype": dtype,
        "aggregation": "mean" if dtype == "continuous" else "last",
    }
    if dtype == "ordinal":
        indicator["ordinal_levels"] = ["low", "medium", "high"]
    if dtype == "categorical":
        indicator["categorical_levels"] = ["a", "b", "c"]
    return indicator


def _structure(
    construct_names: list[str],
    indicators: list[dict[str, Any]],
    *,
    time_invariant: set[str] | None = None,
) -> ModelSpec:
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec

    model = make_model(construct_names)
    return model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                type(construct).model_validate(
                    {
                        **construct.model_dump(),
                        "temporal_status": "time_invariant"
                        if construct.name in (time_invariant or set())
                        else "time_varying",
                        "indicators": tuple(
                            IndicatorSpec.model_validate(
                                {key: value for key, value in row.items() if key != "construct_id"}
                            )
                            for row in indicators
                            if row["construct_id"] == construct.id
                        ),
                    }
                )
                for construct in model.constructs
            ),
        )
    )


_LIKELIHOOD_BY_DTYPE = {
    "continuous": (DistributionFamily.GAUSSIAN, LinkFunction.IDENTITY),
    "binary": (DistributionFamily.BERNOULLI, LinkFunction.LOGIT),
    "ordinal": (DistributionFamily.ORDERED_LOGISTIC, LinkFunction.CUMULATIVE_LOGIT),
    "categorical": (DistributionFamily.CATEGORICAL, LinkFunction.SOFTMAX),
}


# ═══════════════════════════════════════════════════════════════════════
# Ordered-logistic: free threshold base, no centering
# ═══════════════════════════════════════════════════════════════════════


class TestOrderedThresholds:
    def test_cutpoints_keep_free_base(self):
        """The threshold base shifts the cutpoints instead of cancelling out."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "identification_anchors/testorderedthresholds_test_cutpoints_keep_free_base_model_fixture.json"
            ).read_text()
        )
        extra = assemble_sampled_extra_params(
            spec,
            {
                "obs_ordered_base": jnp.array([0.7]),
                "obs_ordered_gaps": jnp.array([[0.5]]),
            },
        )
        np.testing.assert_allclose(
            np.asarray(extra["obs_ordered_cutpoints"]), np.array([[0.7, 1.2]])
        )

    def test_ordinal_only_construct_compiles(self):
        """Well-at-zero anchors location; the fixed logistic link anchors scale."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "identification_anchors/testorderedthresholds_test_ordinal_only_construct_compiles__with_likelihoods.json"
            ).read_text()
        )
        assert numeric.categorical_anchors(spec) is not None
        assert not any(numeric.categorical_anchors(spec))
        assert float(numeric.loading_block(spec).template[0, 0]) == 1.0
        assert not numeric.loading_block(spec).free_support[0, 0]

    def test_manifest_intercept_is_rejected_for_threshold_channel(self):
        with pytest.raises(NumericalSupportError, match=r"Observation intercept.*is inactive"):
            numeric.validate_execution(
                ModelSpec.model_validate_json(
                    (
                        Path(__file__).resolve().parents[2]
                        / "fixtures/models"
                        / "identification_anchors/testorderedthresholds_test_manifest_intercept_is_rejected_for_threshold_channel__with_likelihoods.json"
                    ).read_text()
                )
            )


# ═══════════════════════════════════════════════════════════════════════
# Location anchors: equilibrium center and static t0 mean
# ═══════════════════════════════════════════════════════════════════════


class TestLocationAnchors:
    def test_manifest_intercept_is_rejected_for_standardized_channel(self):
        with pytest.raises(NumericalSupportError, match=r"Observation intercept.*is inactive"):
            numeric.validate_execution(
                ModelSpec.model_validate_json(
                    (
                        Path(__file__).resolve().parents[2]
                        / "fixtures/models"
                        / "identification_anchors/testlocationanchors_test_manifest_intercept_is_rejected_for_standardized_channel__with_likelihoods.json"
                    ).read_text()
                )
            )

    def test_manifest_intercept_remains_free_for_raw_gaussian_sum_channel(self):
        indicator = _indicator("fill_quantity", "dose", "continuous")
        indicator["aggregation"] = "sum"
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "identification_anchors/testlocationanchors_test_manifest_intercept_remains_free_for_raw_gaussian_sum_channel__with_likelihoods.json"
            ).read_text()
        )

        assert numeric.observation_standardized(spec) == [False]
        assert numeric.observation_mean_block(spec).free_support.tolist() == [True]

    def test_manifest_intercept_remains_free_for_binary_channel(self):
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "identification_anchors/testlocationanchors_test_manifest_intercept_remains_free_for_binary_channel__with_likelihoods.json"
            ).read_text()
        )
        assert numeric.observation_mean_block(spec).free_support.tolist() == [True]

    def test_free_center_without_standardized_channel_fails(self):
        with pytest.raises(ValueError, match="Construct 'mood' has no location anchor"):
            ModelSpec.model_validate_json(
                (
                    Path(__file__).resolve().parents[2]
                    / "fixtures/models"
                    / "identification_anchors/testlocationanchors_test_free_center_without_standardized_channel_fails__with_likelihoods.json"
                ).read_text()
            )

    def test_free_center_with_standardized_channel_compiles(self):
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "identification_anchors/testlocationanchors_test_free_center_with_standardized_channel_compiles__with_likelihoods.json"
            ).read_text()
        )
        assert numeric.observation_standardized(spec) is not None
        assert numeric.observation_standardized(spec)[0]

    @pytest.mark.parametrize(
        ("affine", "model_payload"),
        [
            pytest.param(
                False,
                "identification_anchors/testlocationanchors_test_exact_state_anchors_location_for_every_summary_complete_component_slots_2_false-last.json",
                id="False-last",
            ),
            pytest.param(
                False,
                "identification_anchors/testlocationanchors_test_exact_state_anchors_location_for_every_summary_complete_component_slots_2_false-sum.json",
                id="False-sum",
            ),
            pytest.param(
                False,
                "identification_anchors/testlocationanchors_test_exact_state_anchors_location_for_every_summary_complete_component_slots_2_false-count.json",
                id="False-count",
            ),
            pytest.param(
                True,
                "identification_anchors/testlocationanchors_test_exact_state_anchors_location_for_every_summary_complete_component_slots_true-last.json",
                id="True-last",
            ),
            pytest.param(
                True,
                "identification_anchors/testlocationanchors_test_exact_state_anchors_location_for_every_summary_complete_component_slots_true-sum.json",
                id="True-sum",
            ),
            pytest.param(
                True,
                "identification_anchors/testlocationanchors_test_exact_state_anchors_location_for_every_summary_complete_component_slots_true-count.json",
                id="True-count",
            ),
        ],
    )
    def test_exact_state_anchors_location_for_every_summary(self, affine, model_payload):
        payload = (
            Path(__file__).resolve().parents[2] / "fixtures/models" / model_payload
        ).read_text()
        if affine:
            with pytest.raises(ValueError, match="Construct 'mood' has no location anchor"):
                ModelSpec.model_validate_json(payload)
        else:
            ModelSpec.model_validate_json(payload)

    def test_static_t0_mean_gated_without_standardized_channel(self):
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "identification_anchors/testlocationanchors_test_static_t0_mean_gated_without_standardized_channel__with_likelihoods.json"
            ).read_text()
        )
        assert numeric.state_names(spec) is not None
        trait_index = numeric.state_names(spec).index("trait")
        assert not numeric.initial_mean_block(spec).free_support[trait_index]

    def test_static_t0_mean_free_with_standardized_channel(self):
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "identification_anchors/testlocationanchors_test_static_t0_mean_free_with_standardized_channel__with_likelihoods.json"
            ).read_text()
        )
        assert numeric.state_names(spec) is not None
        trait_index = numeric.state_names(spec).index("trait")
        assert numeric.initial_mean_block(spec).free_support[trait_index]

    def test_unmeasured_construct_stays_scientific_without_an_unidentified_state(self):
        plan = _structure(["mood", "ghost"], [_indicator("mood_rating", "mood", "continuous")])
        assert plan.get_construct(fixture_entity_id("construct", "ghost")).indicators == ()
        assert fixture_entity_id("construct", "ghost") not in plan.state_order


# ═══════════════════════════════════════════════════════════════════════
# Categorical: pinned loadings and anchor slopes
# ═══════════════════════════════════════════════════════════════════════


class TestCategoricalAnchors:
    def test_categorical_loading_pinned_in_mixed_construct(self):
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "identification_anchors/testcategoricalanchors_test_categorical_loading_pinned_in_mixed_construct__with_likelihoods.json"
            ).read_text()
        )
        assert numeric.observation_names(spec) is not None
        assert numeric.categorical_anchors(spec) is not None
        cat_row = numeric.observation_names(spec).index("mood_kind")
        assert float(numeric.loading_block(spec).template[cat_row, 0]) == 1.0
        assert not numeric.loading_block(spec).free_support[cat_row, 0]
        assert not numeric.categorical_anchors(spec)[cat_row]

    def test_all_categorical_construct_gets_anchor_slope(self):
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "identification_anchors/testcategoricalanchors_test_all_categorical_construct_gets_anchor_slope__with_likelihoods.json"
            ).read_text()
        )
        assert numeric.categorical_anchors(spec) == [True]
        assert numeric.observation_level_counts(spec) == [3]

        extra = assemble_sampled_extra_params(
            spec,
            {
                "obs_cat_intercepts": jnp.array([[0.3, -0.4]]),
                "obs_cat_slopes": jnp.array([[9.9, 2.0]]),
            },
        )
        np.testing.assert_allclose(np.asarray(extra["obs_cat_slopes"]), np.array([[1.0, 2.0]]))

    def test_manifest_intercept_is_rejected_for_categorical_channel(self):
        with pytest.raises(NumericalSupportError, match=r"Observation intercept.*is inactive"):
            numeric.validate_execution(
                ModelSpec.model_validate_json(
                    (
                        Path(__file__).resolve().parents[2]
                        / "fixtures/models"
                        / "identification_anchors/testcategoricalanchors_test_manifest_intercept_is_rejected_for_categorical_channel__with_likelihoods.json"
                    ).read_text()
                )
            )


# ═══════════════════════════════════════════════════════════════════════
# Reference indicator preference and prior-surface activation
# ═══════════════════════════════════════════════════════════════════════


class TestAnchorSurfaces:
    def test_reference_prefers_continuous_over_ordinal(self):
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "identification_anchors/testanchorsurfaces_test_reference_prefers_continuous_over_ordinal__with_likelihoods.json"
            ).read_text()
        )
        assert numeric.observation_names(spec) is not None
        continuous_row = numeric.observation_names(spec).index("mood_rating")
        ordinal_row = numeric.observation_names(spec).index("mood_level")
        assert float(numeric.loading_block(spec).template[continuous_row, 0]) == 1.0
        assert numeric.loading_block(spec).free_support[ordinal_row, 0]
