"""Model structure and discrete-time to continuous-time prior compilation.

Checks reference intervals, construct-specific priors, structural support,
and parameter identity.
"""

from pathlib import Path

import math
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import expression_coefficients
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_priors as compile_ssm_priors,
)
from nof1_causal_lab.prior_distributions import prior_reference_value
from tests.helpers import graph_constructs, make_model
from tests.model_fixtures import compile_fit_fixture

pytestmark = pytest.mark.contract


def _compile_structure(payload: dict[str, Any]) -> ModelSpec:

    return ModelSpec.model_validate(payload)


def _compile_priors_for_test(scientific_model: ModelSpec):
    prior_registry, index_maps, _diagnostics = compile_ssm_priors(scientific_model)
    return prior_registry, index_maps


def _prior_reference_value(prior, flat_index: int = 0) -> float:
    return float(np.asarray(prior_reference_value(prior)).reshape(-1)[flat_index])


def _decay_reference_values(spec: ModelSpec, prior_registry) -> np.ndarray:
    values = np.zeros(numeric.n_states(spec), dtype=float)
    for index, component in enumerate(numeric.dynamics_expressions(spec)):
        for _, site in component.parameter_sites(f"vf_{index}"):
            if site.site_kind == SiteKind.DYNAMICS_DECAY:
                values[component.target] += _prior_reference_value(prior_registry[site.name])
    return values


def _linear_edge_weight(spec: ModelSpec, prior_registry, *, source: int, target: int) -> float:
    for index, component in enumerate(numeric.dynamics_expressions(spec)):
        if component.source == source and component.target == target:
            for _, site in component.parameter_sites(f"vf_{index}"):
                if site.site_kind == SiteKind.DYNAMICS_WEIGHT:
                    return _prior_reference_value(prior_registry[site.name])
    raise AssertionError(f"No linear coefficient for source={source}, target={target}")


def _decay_support(spec: ModelSpec) -> np.ndarray:
    mask = np.zeros(numeric.n_states(spec), dtype=bool)
    for component in numeric.dynamics_expressions(spec):
        if any(
            operand.role == "decay" for operand in expression_coefficients(component.expression)
        ):
            mask[component.target] = True
    return mask


def _linear_edge_support(spec: ModelSpec) -> np.ndarray:
    mask = np.zeros((numeric.n_states(spec), numeric.n_states(spec)), dtype=bool)
    for component in numeric.dynamics_expressions(spec):
        if component.source is not None and any(
            operand.role == "weight" for operand in expression_coefficients(component.expression)
        ):
            mask[component.target, component.source] = True
    return mask


def _state_intercept_mask(spec: ModelSpec) -> np.ndarray:
    mask = np.zeros(numeric.n_states(spec), dtype=bool)
    for component in numeric.dynamics_expressions(spec):
        if not component.edge_owned and any(
            operand.role in {"center", "intercept"} for _, operand in component.parameters
        ):
            mask[component.target] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def two_construct_structure() -> ModelSpec:
    """Realistic 2-construct causal design: stress → mood.

    - Both constructs are daily time-varying
    - 3 indicators: mood_rating, stress_self_report, stress_cortisol
    - stress_cortisol is a second indicator for stress (free loading)
    """
    return _compile_structure(
        {
            "default_outcome": "construct:bbc87212909e45b9e6c3",
            "edges": [
                {
                    "cause": {
                        "id": "construct:6b04dc42c531e7091eb8",
                        "name": "stress",
                        "description": "Daily stress level",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                        "indicators": [
                            {
                                "id": "indicator:4ff8be7491bd87d28af4",
                                "name": "stress_self_report",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                                "construct_polarity": "positive",
                            },
                            {
                                "id": "indicator:522342c2385e38d5e750",
                                "name": "stress_cortisol",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                                "construct_polarity": "positive",
                            },
                        ],
                    },
                    "effect": {
                        "id": "construct:bbc87212909e45b9e6c3",
                        "name": "mood",
                        "description": "Daily mood state",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                        "indicators": [
                            {
                                "id": "indicator:e05e217de7f4442abdc5",
                                "name": "mood_rating",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                                "construct_polarity": "positive",
                            }
                        ],
                    },
                    "id": "edge:923689028b6b177617c2",
                    "description": "Stress impairs mood",
                }
            ],
            "measurement_clock": "1d",
        }
    )


@pytest.fixture
def two_construct_model(two_construct_structure) -> ModelSpec:
    return ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'e2e_discretization/two_construct_model_model.json').read_text())


@pytest.fixture
def weekly_study_priors() -> dict[str, dict[str, Any]]:
    """Priors from a weekly-interval study (reference_interval_days=7).

    AR coefficients: Beta(3,2) → E=0.6 (mood), Beta(2,2) → E=0.5 (stress)
    Cross-lag: Normal(0.3, 0.15) at weekly scale
    """
    return {
        "rho_mood": {
            "parameter": "rho_mood",
            "distribution": "Beta",
            "params": {"alpha": 3.0, "beta": 2.0},
            "sources": [
                {
                    "title": "Weekly mood dynamics meta-analysis",
                    "snippet": "AR(1) ≈ 0.6 at weekly interval",
                    "study_interval_days": 7.0,
                }
            ],
            "reasoning": "Meta-analysis of weekly diary studies",
            "reference_interval_days": 7.0,
        },
        "rho_stress": {
            "parameter": "rho_stress",
            "distribution": "Beta",
            "params": {"alpha": 2.0, "beta": 2.0},
            "sources": [],
            "reasoning": "No literature; weakly informative",
            # No reference_interval_days → falls back to dt=1
        },
        "beta_stress_mood": {
            "parameter": "beta_stress_mood",
            "distribution": "Normal",
            "params": {"mu": 0.3, "sigma": 0.15},
            "sources": [
                {
                    "title": "Stress-mood cross-lag study",
                    "snippet": "β = 0.3 at weekly interval",
                    "study_interval_days": 7.0,
                }
            ],
            "reasoning": "Weekly cross-lagged panel study",
            "reference_interval_days": 7.0,
        },
        "sigma_mood": {
            "parameter": "sigma_mood",
            "distribution": "HalfNormal",
            "params": {"sigma": 1.0},
            "sources": [],
            "reasoning": "Weakly informative",
        },
        "sigma_stress": {
            "parameter": "sigma_stress",
            "distribution": "HalfNormal",
            "params": {"sigma": 1.0},
            "sources": [],
            "reasoning": "Weakly informative",
        },
        "lambda_stress_cortisol_stress": {
            "parameter": "lambda_stress_cortisol_stress",
            "distribution": "HalfNormal",
            "params": {"sigma": 0.8},
            "sources": [],
            "reasoning": "Weakly informative free loading",
        },
        "obs_sd_stress_self_report": {
            "parameter": "obs_sd_stress_self_report",
            "distribution": "HalfNormal",
            "params": {"sigma": 0.5},
            "sources": [],
            "reasoning": "Weakly informative measurement error",
        },
        "obs_sd_stress_cortisol": {
            "parameter": "obs_sd_stress_cortisol",
            "distribution": "HalfNormal",
            "params": {"sigma": 0.5},
            "sources": [],
            "reasoning": "Weakly informative measurement error",
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: First-order DT→CT with reference_interval_days
# ═══════════════════════════════════════════════════════════════════════


class TestE2ESpecToDiscretization:
    """End-to-end: ModelSpec → ModelSpec → dict[str, dist.Distribution] → discretize → roundtrip."""

    def test_source_model_structure_from_dag(self, two_construct_structure, two_construct_model):
        """Compilation produces correct ModelSpec from DAG structure."""

        # Dimensions
        assert numeric.n_states(two_construct_model) == 2  # mood, stress
        assert (
            numeric.n_observations(two_construct_model) == 3
        )  # mood_rating, stress_self_report, stress_cortisol
        assert numeric.state_names(two_construct_model) == ["stress", "mood"]

        # Dynamics support: diagonal decay (AR) + stress→mood linear edge.
        np.testing.assert_array_equal(_decay_support(two_construct_model), [True, True])
        edge_support = _linear_edge_support(two_construct_model)
        assert edge_support[1, 0]  # stress→mood coupling (effect=mood row, cause=stress col)
        assert not edge_support[0, 1]  # no mood→stress edge

        # Lambda mask: stress_cortisol has free loading for stress
        assert numeric.loading_block(two_construct_model).free_support is not None
        # mood_rating loads on mood (fixed=1.0), stress_self_report loads on stress (fixed=1.0)
        # stress_cortisol loads on stress (free)
        manifest_names = numeric.observation_names(two_construct_model)
        assert manifest_names is not None
        stress_cortisol_idx = manifest_names.index("stress_cortisol")
        assert numeric.state_names(two_construct_model) is not None
        stress_latent_idx = numeric.state_names(two_construct_model).index("stress")
        assert numeric.loading_block(two_construct_model).free_support[
            stress_cortisol_idx, stress_latent_idx
        ]

    def test_model_owns_latent_identity(self, two_construct_model):

        model = two_construct_model
        renamed = model.revised(
            edges=replace_constructs(
                model.edges,
                tuple(
                    type(c).model_validate({**c.model_dump(), "name": "renamed"})
                    if c.name == "mood"
                    else c
                    for c in model.constructs
                ),
            )
        )
        spec = renamed
        assert numeric.state_names(spec) == ["stress", "renamed"]
        assert numeric.state_ids(spec) == [c.id for c in model.constructs]
        assert [p.id for p in renamed.parameters] == [p.id for p in model.parameters]

    def test_time_invariant_states_drop_static_target_dynamics_and_diffusion_support(self):

        model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'e2e_discretization/teste2espectodiscretization_test_time_invariant_states_drop_static_target_dynamics_and_diffusion_support_complete_test_model.json').read_text())
        spec = model
        assert not model.constructs[0].dynamics
        static_index = numeric.state_names(spec).index("baseline")
        dynamic_index = numeric.state_names(spec).index("mood")
        assert not _decay_support(spec)[static_index]
        assert _decay_support(spec)[dynamic_index]
        assert not numeric.diffusion_block(spec).diffusion_chol_support[static_index, static_index]
        assert numeric.diffusion_block(spec).diffusion_chol_support[dynamic_index, dynamic_index]
        assert not _linear_edge_support(spec)[static_index].any()
        assert not _state_intercept_mask(spec)[static_index]

    def test_model_rejects_mechanism_reference_outside_its_owners(self, two_construct_model):
        payload = two_construct_model.model_dump(mode="json")
        target = graph_constructs(payload)[0]["dynamics"][0]
        target["expression"] = {
            "kind": "coefficient",
            "role": "decay",
            "value": "parameter:foreign",
        }
        with pytest.raises(ValueError, match="parameter"):
            ModelSpec.model_validate(payload)

    def test_compiled_artifact_roundtrips_grounded_structure(
        self,
        two_construct_structure,
        two_construct_model,
        weekly_study_priors,
    ):
        """Compiled artifacts preserve the grounded latent and measurement layout."""
        from nof1_causal_lab.models.model_checks import check_execution
        from nof1_causal_lab.models.ssm.runtime import build_ssm_model
        from nof1_causal_lab.utils.data import pivot_to_wide

        typed_scientific_model = ModelSpec.model_validate(two_construct_model)
        check_execution(
            ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'e2e_discretization/teste2espectodiscretization_test_compiled_artifact_roundtrips_grounded_structure_make_prior_model.json').read_text()),
        )

        assert numeric.state_names(
            ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'e2e_discretization/teste2espectodiscretization_test_compiled_artifact_roundtrips_grounded_structure_make_prior_model_2.json').read_text())
        ) == ["stress", "mood"]
        assert numeric.observation_names(typed_scientific_model) == [
            "stress_self_report",
            "stress_cortisol",
            "mood_rating",
        ]
        binding_rows = [
            {
                "parameter": next(
                    p.name
                    for p in typed_scientific_model.parameters
                    if p.id == binding.parameter_id
                ),
                "site_name": binding.site_name,
                "flat_index": binding.flat_index,
            }
            for binding in parameter_bindings(
                ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'e2e_discretization/teste2espectodiscretization_test_compiled_artifact_roundtrips_grounded_structure_make_prior_model_3.json').read_text())
            )[0]
        ]
        bindings = {item["parameter"]: item for item in binding_rows}
        assert bindings["beta_stress_mood"]["site_name"] == "vf_2_p0"
        assert bindings["rho_mood"]["site_name"] == "vf_1_p0"
        assert bindings["rho_stress"]["site_name"] == "vf_0_p0"
        assert bindings["lambda_stress_cortisol_stress"]["flat_index"] == 0
        assert set(bindings) == {p.name for p in typed_scientific_model.parameters}

        data_for_model = pl.DataFrame(
            {
                "indicator": [
                    "mood_rating",
                    "stress_self_report",
                    "stress_cortisol",
                    "mood_rating",
                    "stress_self_report",
                    "stress_cortisol",
                ],
                "value": [6.0, 4.0, 10.0, 7.0, 5.0, 11.0],
                "anchor_time": [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-02T00:00:00",
                ],
            }
        )

        indicator_ids = {
            item.name: item.id for item in two_construct_structure._indicators.values()
        }
        data_for_model = data_for_model.with_columns(
            pl.col("indicator").replace_strict(indicator_ids).alias("indicator_id")
        ).drop("indicator")
        model = build_ssm_model(
            pivot_to_wide(data_for_model, time_origin=datetime(2024, 1, 1, tzinfo=UTC)).rename(
                {i.id: i.name for i in typed_scientific_model.indicators}
            ),
            inputs=compile_fit_fixture(
                ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'e2e_discretization/teste2espectodiscretization_test_compiled_artifact_roundtrips_grounded_structure_make_prior_model_4.json').read_text())
            ),
        )
        spec = model.spec
        assert numeric.state_names(spec) == ["stress", "mood"]
        edge_support = _linear_edge_support(spec)
        assert edge_support[1, 0]
        assert not edge_support[0, 1]
        assert numeric.loading_block(spec).free_support is not None
        assert numeric.loading_block(spec).free_support[1, 0]
        runtime = model.get_prior_runtime_bundle()
        assert runtime.priors["vf_0_p0"].batch_shape == ()
        assert runtime.priors["vf_1_p0"].batch_shape == ()
        assert model.parameter_bindings == tuple(
            parameter_bindings(ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'e2e_discretization/teste2espectodiscretization_test_compiled_artifact_roundtrips_grounded_structure_make_prior_model_5.json').read_text()))[0]
        )

    def test_residual_sd_priors_are_construct_specific(
        self, two_construct_structure, two_construct_model
    ):
        """Construct-specific sigma priors compile to per-latent diffusion scales."""

        ssm_priors, _idx = _compile_priors_for_test(ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / "e2e_discretization/construct_specific_residual_scales.json").read_text()))

        np.testing.assert_allclose(ssm_priors["diffusion_diag_free"].scale, [0.9, 0.1])

    def test_dt_to_ct_uses_reference_interval_days(
        self,
        two_construct_structure,
        two_construct_model,
        weekly_study_priors,
    ):
        """Priors with reference_interval_days use that dt.

        rho_mood has reference_interval_days=7 → dt=7
        rho_stress has no reference_interval_days → falls back to dt=1
        beta_stress_mood has reference_interval_days=7 → dt=7
        """
        ssm_priors, _idx = _compile_priors_for_test(ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / "e2e_discretization/weekly_reference_intervals.json").read_text()))

        # --- rho_mood: Beta(3,2) → E=0.6, reference_interval_days=7 ---
        # dynamics decay for mood = -ln(0.6) / 7 ≈ 0.073
        mu_ar_mood = 3.0 / 5.0  # E[Beta(3,2)] = 0.6
        expected_dynamics_mood = -math.log(mu_ar_mood) / 7.0
        mu_dynamics = _decay_reference_values(two_construct_model, ssm_priors)
        mu_mood = mu_dynamics[1]
        assert abs(mu_mood - expected_dynamics_mood) < 0.01, (
            f"mood dynamics: got {mu_mood}, expected {expected_dynamics_mood} "
            f"(using reference_interval_days=7)"
        )

        # --- rho_stress: Beta(2,2) → E=0.5, no reference_interval_days → daily dt=1 ---
        # dynamics decay for stress = -ln(0.5) / 1.0 ≈ 0.693
        mu_ar_stress = 0.5
        expected_dynamics_stress = -math.log(mu_ar_stress) / 1.0
        mu_stress = mu_dynamics[0]
        assert abs(mu_stress - expected_dynamics_stress) < 0.01, (
            f"stress dynamics: got {mu_stress}, expected {expected_dynamics_stress} "
            f"(fallback to daily dt=1)"
        )

        # --- beta_stress_mood: Normal(0.3, 0.15), reference_interval_days=7 ---
        # linear-edge weight = 0.3 / 7 ≈ 0.043
        expected_offdiag = 0.3 / 7.0
        mu_offdiag_val = _linear_edge_weight(two_construct_model, ssm_priors, source=0, target=1)
        assert abs(mu_offdiag_val - expected_offdiag) < 0.01, (
            f"stress→mood dynamics: got {mu_offdiag_val}, expected {expected_offdiag} "
            f"(using reference_interval_days=7)"
        )

    def test_different_intervals_produce_different_rates(self, two_construct_model):
        """Same DT beta at different study intervals → different CT rates.

        beta=0.3 from weekly (dt=7) → CT rate ≈ 0.043
        beta=0.3 from daily  (dt=1) → CT rate ≈ 0.300
        This is the Kuiper & Ryan (2018) sign-reversal effect in action.
        """

        scientific_model = two_construct_model

        # Weekly study priors
        # Daily study priors (same beta value, different interval)

        source_model = scientific_model

        ssm_priors_w, _idx = _compile_priors_for_test(ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / "e2e_discretization/weekly_effect_rate.json").read_text()))

        ssm_priors_d, _idx = _compile_priors_for_test(ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / "e2e_discretization/daily_effect_rate.json").read_text()))

        # Weekly: mixed intervals (beta=7d, rho=1d) → first-order: 0.3 / 7 ≈ 0.043
        mu_w_val = _linear_edge_weight(source_model, ssm_priors_w, source=0, target=1)
        assert abs(mu_w_val - 0.3 / 7.0) < 0.01

        # Daily: beta_CT = beta_DT / dt = 0.3 / 1 = 0.3
        mu_d_val = _linear_edge_weight(source_model, ssm_priors_d, source=0, target=1)
        expected_daily = 0.3
        assert abs(mu_d_val - expected_daily) < 0.05, (
            f"Daily case uses beta/dt scaling: got {mu_d_val}, expected {expected_daily}"
        )

        # Rates should differ significantly because beta/dt depends on the interval.
        assert mu_d_val > mu_w_val, "Daily rate should be larger than weekly rate"


# ═══════════════════════════════════════════════════════════════════════
# Prior factorization and reference intervals
# ═══════════════════════════════════════════════════════════════════════


class TestPriorCompilationMetadata:
    """Compiler-owned prior factorization and reference intervals."""

    def test_compile_keeps_elementwise_priors_when_intervals_match(self, two_construct_structure):
        """Compilation keeps factorized DT→CT priors even when dt values match."""
        scientific_model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'e2e_discretization/testpriorcompilationmetadata_test_compile_keeps_elementwise_priors_when_intervals_match_complete_test_model.json').read_text())

        # All parameters at dt=7 (weekly)

        source_model = scientific_model

        ssm_priors, _idx = _compile_priors_for_test(ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / "e2e_discretization/equal_intervals_elementwise_priors.json").read_text()))

        dynamics_decay = _decay_reference_values(source_model, ssm_priors)
        linear_edge_weight = _linear_edge_weight(source_model, ssm_priors, source=0, target=1)

        assert abs(dynamics_decay[1] - (-math.log(0.6) / 7.0)) < 0.01
        assert abs(dynamics_decay[0] - (-math.log(0.5) / 7.0)) < 0.01
        assert abs(linear_edge_weight - (0.3 / 7.0)) < 0.01
