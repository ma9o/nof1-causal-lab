"""Guard tests for the per-construct identification-anchor invariant.

Every retained construct must have exactly one location anchor and one scale
anchor (docs/reference/statistical-model-spec/identification.md). These tests
enumerate the family/policy combinations so that any future eligibility change
that reopens an exact likelihood ridge fails loudly at compile time.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.identity import ConstructRef, IndicatorRef
from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LikelihoodSpec, LinkFunction
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.parameter_spec import (
    ParameterSpec,
)
from nof1_causal_lab.models.likelihoods import observation_law
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.structural import StructuralClosureError
from nof1_causal_lab.models.ssm.compile.support import NumericalSupportError
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.likelihood_extra_params import assemble_sampled_extra_params
from tests.helpers import declare_test_dynamics, fixture_entity_id, make_model
from tests.model_fixtures import model_fixture
from tests.slot_fixtures import fixture_parameter_id, with_likelihood_coefficients

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
        "how_to_measure": f"measure {name}",
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
    from nof1_causal_lab.artifacts.indicator import Indicator

    model = make_model(construct_names)
    return model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                construct.model_copy(
                    update={
                        "temporal_status": "time_invariant"
                        if construct.name in (time_invariant or set())
                        else "time_varying",
                        "indicators": tuple(
                            Indicator.model_validate(
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


def _likelihood(variable: str, dtype: str):
    return fixture_entity_id("indicator", variable), _LIKELIHOOD_BY_DTYPE[dtype]


def _with_likelihoods(
    likelihoods: list[tuple[str, tuple[DistributionFamily, LinkFunction]]],
    parameters: list[ParameterSpec] | None = None,
    *,
    plan: ModelSpec,
    centered_states: tuple[str, ...] = (),
) -> ModelSpec:
    from nof1_causal_lab.models.model_semantics import should_auto_standardize_indicator
    from nof1_causal_lab.utils.observation_semantics import get_observation_semantics

    selected = {
        identity: LikelihoodSpec(
            law=observation_law(plan.indicator_owner(identity).id, family, link), reasoning="test"
        )
        for identity, (family, link) in likelihoods
    }
    model = plan.revised(
        edges=replace_constructs(
            plan.edges,
            tuple(
                construct.model_copy(
                    update={
                        "indicators": tuple(
                            indicator.model_copy(
                                update={
                                    "likelihood": selected[indicator.id].model_copy(
                                        update={
                                            "standardized": should_auto_standardize_indicator(
                                                selected[indicator.id].law.family,
                                                selected[indicator.id].terms.link,
                                                get_observation_semantics(
                                                    indicator.model_dump(mode="json")
                                                ).support_kind,
                                                get_observation_semantics(
                                                    indicator.model_dump(mode="json")
                                                ).summary_operator,
                                            )
                                        }
                                    )
                                }
                            )
                            for indicator in construct.indicators
                        ),
                    }
                )
                for construct in plan.constructs
            ),
        )
    )
    from nof1_causal_lab.artifacts.coefficient import ParameterCoefficient
    from nof1_causal_lab.models.parameter_planning import complete_component_slots

    declared = declare_test_dynamics(model, centered_states=centered_states)
    replacements = {p.name: p for p in parameters or ()}
    updated = {p.id: p for p in declared.parameters}
    constructs = []
    for construct in declared.constructs:
        indicators = []
        for indicator in construct.indicators:
            parameter = replacements.get(f"manifest_mean_{indicator.name}")
            if parameter is not None:
                updated[parameter.id] = parameter
                indicator = indicator.model_copy(
                    update={
                        "likelihood": with_likelihood_coefficients(
                            indicator.likelihood,
                            {
                                "observation_intercept": ParameterCoefficient(
                                    parameter_id=parameter.id
                                )
                            },
                        )
                    }
                )
            indicators.append(indicator)
        constructs.append(construct.model_copy(update={"indicators": tuple(indicators)}))
    declared = declared.revised(
        edges=replace_constructs(declared.edges, tuple(constructs)),
        parameters=tuple(updated.values()),
    )
    return complete_component_slots(declared)


def _manifest_mean(variable: str, *, plan: ModelSpec) -> ParameterSpec:
    iid = fixture_entity_id("indicator", variable)
    owners = (ConstructRef(id=plan.indicator_owner(iid).id), IndicatorRef(id=iid))
    return ParameterSpec(
        id=fixture_parameter_id(SiteKind.MANIFEST_MEANS, owners),
        name=f"manifest_mean_{variable}",
        description=f"Observation intercept for {variable}",
    )


def _center(plan: ModelSpec) -> ParameterSpec:
    from nof1_causal_lab.artifacts.identity import MechanismRef
    from nof1_causal_lab.models.model_mechanisms import default_mechanism_id

    cid = plan.state_order[0]
    owners = (ConstructRef(id=cid), MechanismRef(id=default_mechanism_id(cid, "node_potential")))
    return ParameterSpec(
        id=fixture_parameter_id(SiteKind.DYNAMICS_POTENTIAL_CENTER, owners),
        name="cint_mood",
        description="equilibrium center",
    )


# ═══════════════════════════════════════════════════════════════════════
# Ordered-logistic: free threshold base, no centering
# ═══════════════════════════════════════════════════════════════════════


class TestOrderedThresholds:
    def test_cutpoints_keep_free_base(self):
        """The threshold base shifts the cutpoints instead of cancelling out."""
        spec = model_fixture(
            n_latent=1,
            n_manifest=1,
            dynamics_spec=DynamicsSpec(n_latent=1, components=()),
            manifest_dists=[DistributionFamily.ORDERED_LOGISTIC],
            manifest_level_counts=[3],
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
        model = _structure(["mood"], [_indicator("mood_level", "mood", "ordinal")])
        spec, _ = (
            _with_likelihoods([_likelihood("mood_level", "ordinal")], plan=model),
            numeric.edge_lag_days(
                _with_likelihoods([_likelihood("mood_level", "ordinal")], plan=model)
            ),
        )
        assert numeric.categorical_anchors(spec) is not None
        assert not any(numeric.categorical_anchors(spec))
        assert float(numeric.loading_block(spec).template[0, 0]) == 1.0
        assert not numeric.loading_block(spec).free_support[0, 0]

    def test_manifest_intercept_is_rejected_for_threshold_channel(self):
        model = _structure(["mood"], [_indicator("mood_level", "mood", "ordinal")])
        with pytest.raises(NumericalSupportError, match=r"Observation intercept.*is inactive"):
            numeric.validate_execution(
                _with_likelihoods(
                    [_likelihood("mood_level", "ordinal")],
                    [_manifest_mean("mood_level", plan=model)],
                    plan=model,
                )
            )


# ═══════════════════════════════════════════════════════════════════════
# Location anchors: equilibrium center and static t0 mean
# ═══════════════════════════════════════════════════════════════════════


class TestLocationAnchors:
    def test_manifest_intercept_is_rejected_for_standardized_channel(self):
        model = _structure(["mood"], [_indicator("mood_rating", "mood", "continuous")])
        with pytest.raises(NumericalSupportError, match=r"Observation intercept.*is inactive"):
            numeric.validate_execution(
                _with_likelihoods(
                    [_likelihood("mood_rating", "continuous")],
                    [_manifest_mean("mood_rating", plan=model)],
                    plan=model,
                )
            )

    def test_manifest_intercept_remains_free_for_raw_gaussian_sum_channel(self):
        indicator = _indicator("fill_quantity", "dose", "continuous")
        indicator["aggregation"] = "sum"
        model = _structure(["dose"], [indicator])
        spec, _ = (
            _with_likelihoods(
                [_likelihood("fill_quantity", "continuous")],
                [_manifest_mean("fill_quantity", plan=model)],
                plan=model,
            ),
            numeric.edge_lag_days(
                _with_likelihoods(
                    [_likelihood("fill_quantity", "continuous")],
                    [_manifest_mean("fill_quantity", plan=model)],
                    plan=model,
                )
            ),
        )

        assert numeric.observation_standardized(spec) == [False]
        assert numeric.observation_mean_block(spec).free_support.tolist() == [True]

    def test_manifest_intercept_remains_free_for_binary_channel(self):
        model = _structure(["mood"], [_indicator("mood_flag", "mood", "binary")])
        spec, _ = (
            _with_likelihoods(
                [_likelihood("mood_flag", "binary")],
                [_manifest_mean("mood_flag", plan=model)],
                plan=model,
            ),
            numeric.edge_lag_days(
                _with_likelihoods(
                    [_likelihood("mood_flag", "binary")],
                    [_manifest_mean("mood_flag", plan=model)],
                    plan=model,
                )
            ),
        )
        assert numeric.observation_mean_block(spec).free_support.tolist() == [True]

    def test_free_center_without_standardized_channel_fails(self):
        model = _structure(["mood"], [_indicator("mood_flag", "mood", "binary")])
        spec = _with_likelihoods(
            [_likelihood("mood_flag", "binary")],
            [_center(model)],
            centered_states=(model.state_order[0],),
            plan=model,
        )
        with pytest.raises(StructuralClosureError, match="no location anchor"):
            numeric.validate_execution(spec)

    def test_free_center_with_standardized_channel_compiles(self):
        model = _structure(
            ["mood"],
            [
                _indicator("mood_rating", "mood", "continuous"),
                _indicator("mood_flag", "mood", "binary"),
            ],
        )
        spec, _ = (
            _with_likelihoods(
                [
                    _likelihood("mood_rating", "continuous"),
                    _likelihood("mood_flag", "binary"),
                ],
                [_center(model)],
                centered_states=(model.state_order[0],),
                plan=model,
            ),
            numeric.edge_lag_days(
                _with_likelihoods(
                    [
                        _likelihood("mood_rating", "continuous"),
                        _likelihood("mood_flag", "binary"),
                    ],
                    [_center(model)],
                    centered_states=(model.state_order[0],),
                    plan=model,
                )
            ),
        )
        assert numeric.observation_standardized(spec) is not None
        assert numeric.observation_standardized(spec)[0]

    def test_static_t0_mean_gated_without_standardized_channel(self):
        model = _structure(
            ["mood", "trait"],
            [
                _indicator("mood_rating", "mood", "continuous"),
                _indicator("trait_flag", "trait", "binary"),
            ],
            time_invariant={"trait"},
        )
        spec, _ = (
            _with_likelihoods(
                [
                    _likelihood("mood_rating", "continuous"),
                    _likelihood("trait_flag", "binary"),
                ],
                plan=model,
            ),
            numeric.edge_lag_days(
                _with_likelihoods(
                    [
                        _likelihood("mood_rating", "continuous"),
                        _likelihood("trait_flag", "binary"),
                    ],
                    plan=model,
                )
            ),
        )
        assert numeric.state_names(spec) is not None
        trait_index = numeric.state_names(spec).index("trait")
        assert not numeric.initial_mean_block(spec).free_support[trait_index]

    def test_static_t0_mean_free_with_standardized_channel(self):
        model = _structure(
            ["mood", "trait"],
            [
                _indicator("mood_rating", "mood", "continuous"),
                _indicator("trait_score", "trait", "continuous"),
            ],
            time_invariant={"trait"},
        )
        spec, _ = (
            _with_likelihoods(
                [
                    _likelihood("mood_rating", "continuous"),
                    _likelihood("trait_score", "continuous"),
                ],
                plan=model,
            ),
            numeric.edge_lag_days(
                _with_likelihoods(
                    [
                        _likelihood("mood_rating", "continuous"),
                        _likelihood("trait_score", "continuous"),
                    ],
                    plan=model,
                )
            ),
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
        model = _structure(
            ["mood"],
            [
                _indicator("mood_rating", "mood", "continuous"),
                _indicator("mood_kind", "mood", "categorical"),
            ],
        )
        spec, _ = (
            _with_likelihoods(
                [
                    _likelihood("mood_rating", "continuous"),
                    _likelihood("mood_kind", "categorical"),
                ],
                plan=model,
            ),
            numeric.edge_lag_days(
                _with_likelihoods(
                    [
                        _likelihood("mood_rating", "continuous"),
                        _likelihood("mood_kind", "categorical"),
                    ],
                    plan=model,
                )
            ),
        )
        assert numeric.observation_names(spec) is not None
        assert numeric.categorical_anchors(spec) is not None
        cat_row = numeric.observation_names(spec).index("mood_kind")
        assert float(numeric.loading_block(spec).template[cat_row, 0]) == 1.0
        assert not numeric.loading_block(spec).free_support[cat_row, 0]
        assert not numeric.categorical_anchors(spec)[cat_row]

    def test_all_categorical_construct_gets_anchor_slope(self):
        model = _structure(["mood"], [_indicator("mood_kind", "mood", "categorical")])
        spec, _ = (
            _with_likelihoods([_likelihood("mood_kind", "categorical")], plan=model),
            numeric.edge_lag_days(
                _with_likelihoods([_likelihood("mood_kind", "categorical")], plan=model)
            ),
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
        model = _structure(["mood"], [_indicator("mood_kind", "mood", "categorical")])
        with pytest.raises(NumericalSupportError, match=r"Observation intercept.*is inactive"):
            numeric.validate_execution(
                _with_likelihoods(
                    [_likelihood("mood_kind", "categorical")],
                    [_manifest_mean("mood_kind", plan=model)],
                    plan=model,
                )
            )


# ═══════════════════════════════════════════════════════════════════════
# Reference indicator preference and prior-surface activation
# ═══════════════════════════════════════════════════════════════════════


class TestAnchorSurfaces:
    def test_reference_prefers_continuous_over_ordinal(self):
        model = _structure(
            ["mood"],
            [
                _indicator("mood_level", "mood", "ordinal"),
                _indicator("mood_rating", "mood", "continuous"),
            ],
        )
        spec, _ = (
            _with_likelihoods(
                [
                    _likelihood("mood_level", "ordinal"),
                    _likelihood("mood_rating", "continuous"),
                ],
                plan=model,
            ),
            numeric.edge_lag_days(
                _with_likelihoods(
                    [
                        _likelihood("mood_level", "ordinal"),
                        _likelihood("mood_rating", "continuous"),
                    ],
                    plan=model,
                )
            ),
        )
        assert numeric.observation_names(spec) is not None
        continuous_row = numeric.observation_names(spec).index("mood_rating")
        ordinal_row = numeric.observation_names(spec).index("mood_level")
        assert float(numeric.loading_block(spec).template[continuous_row, 0]) == 1.0
        assert numeric.loading_block(spec).free_support[ordinal_row, 0]

    def test_raw_gaussian_sum_authors_an_intercept_slot(self):
        from nof1_causal_lab.artifacts.coefficient import ParameterCoefficient
        from nof1_causal_lab.models.parameter_planning import complete_component_slots

        plan = _structure(["dose"], [_indicator("fill_quantity", "dose", "continuous")])
        owner = plan.constructs[0]
        indicator = owner.indicators[0].model_copy(
            update={
                "aggregation": "sum",
                "likelihood": LikelihoodSpec(
                    law=observation_law(owner.id, "gaussian", "identity"),
                    standardized=False,
                    reasoning="Total quantity",
                ),
            }
        )
        model = plan.revised(
            edges=replace_constructs(
                plan.edges, (owner.model_copy(update={"indicators": (indicator,)}),)
            )
        )
        completed = complete_component_slots(model)
        likelihood = completed.indicators[0].likelihood
        assert likelihood is not None
        assert isinstance(likelihood.terms.intercept.coefficient, ParameterCoefficient)
