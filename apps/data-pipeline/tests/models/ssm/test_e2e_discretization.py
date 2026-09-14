"""End-to-end tests: ModelSpec -> Prior Conversion -> Discretization.

These tests verify the full chain from a realistic causal design
through DT→CT prior conversion and CT→DT discretization, checking that
the mathematical roundtrip is consistent.

Phase 1 tests:
- reference_interval_days precedence chain for DT→CT conversion
- ModelSpec structure (dynamics_support, lambda_support) from DAG
- First-order DT→CT→DT roundtrip consistency
- Prior predictive produces finite, stable samples

Compilation also preserves factorized priors and checks causal edge-lag metadata.
"""

import math
from typing import Any

import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import expression_coefficients
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_priors as compile_ssm_priors,
)
from nof1_causal_lab.prior_distributions import prior_reference_value
from tests.helpers import (
    complete_test_model,
    graph_constructs,
    make_model,
    model_with_prior_payloads,
    named_prior_payloads,
)
from tests.model_fixtures import (
    affine_test_evolution,
)


def _compile_structure(payload: dict[str, Any]) -> ModelSpec:

    return ModelSpec.model_validate(payload)


def _compile_priors_for_test(
    priors: dict[str, dict[str, Any]],
    scientific_model: ModelSpec,
    *,
    edge_lag_days: dict[tuple[int, int], float] | None = None,
):
    prior_registry, index_maps, _diagnostics = compile_ssm_priors(
        model_with_prior_payloads(
            ModelSpec.model_validate(scientific_model),
            named_prior_payloads(ModelSpec.model_validate(scientific_model), priors),
        ),
        edge_lag_days=edge_lag_days,
    )
    return prior_registry, index_maps


def _prior_law(prior_registry, site_name: str):
    return prior_registry[site_name]


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
            operand.role in {"center", "intercept"} for operand in component.parameters
        ):
            mask[component.target] = True
    return mask


def _reference_dynamics_from_priors(spec: ModelSpec, prior_registry) -> jnp.ndarray:
    dynamics = -np.diag(_decay_reference_values(spec, prior_registry))
    for target, source in np.argwhere(_linear_edge_support(spec)):
        dynamics[target, source] += _linear_edge_weight(
            spec, prior_registry, source=int(source), target=int(target)
        )
    return jnp.asarray(dynamics, dtype=jnp.float32)


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
                                "how_to_measure": "Self-reported stress (1-10)",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                                "construct_polarity": "positive",
                            },
                            {
                                "id": "indicator:522342c2385e38d5e750",
                                "name": "stress_cortisol",
                                "how_to_measure": "Salivary cortisol (nmol/L)",
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
                                "how_to_measure": "Self-reported mood (1-10)",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                                "construct_polarity": "positive",
                            }
                        ],
                    },
                    "id": "edge:923689028b6b177617c2",
                    "description": "Stress impairs mood",
                    "lagged": True,
                }
            ],
            "measurement_clock": "1d",
        }
    )


@pytest.fixture
def two_construct_model(two_construct_structure) -> ModelSpec:
    """Explicit scientific choices and defaults before the pure compiler."""
    return complete_test_model(two_construct_structure)


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
                    c.model_copy(update={"name": "renamed"}) if c.name == "mood" else c
                    for c in model.constructs
                ),
            )
        )
        spec, _ = (renamed, numeric.edge_lag_days(renamed))
        assert numeric.state_names(spec) == ["stress", "renamed"]
        assert numeric.state_ids(spec) == [c.id for c in model.constructs]
        assert [p.id for p in renamed.parameters] == [p.id for p in model.parameters]

    def test_time_invariant_states_drop_static_target_dynamics_and_diffusion_support(self):

        model = make_model(["baseline", "mood"], [("baseline", "mood")])
        model = model.revised(
            edges=replace_constructs(
                model.edges,
                (
                    model.constructs[0].model_copy(
                        update={"role": "exogenous", "temporal_status": "time_invariant"}
                    ),
                    model.constructs[1],
                ),
            )
        )
        model = complete_test_model(model)
        spec, _ = (model, numeric.edge_lag_days(model))
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
        from tests.helpers import make_prior_model

        typed_scientific_model = ModelSpec.model_validate(two_construct_model)
        compiled = check_execution(
            make_prior_model(typed_scientific_model, weekly_study_priors),
        )

        assert len(compiled) == 2
        assert numeric.state_names(
            make_prior_model(typed_scientific_model, weekly_study_priors)
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
                make_prior_model(typed_scientific_model, weekly_study_priors)
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
            pivot_to_wide(data_for_model).rename(
                {i.id: i.name for i in typed_scientific_model.indicators}
            ),
            model_spec=make_prior_model(typed_scientific_model, weekly_study_priors),
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
        assert (
            model.parameter_bindings
            == parameter_bindings(make_prior_model(typed_scientific_model, weekly_study_priors))[0]
        )

    def test_residual_sd_priors_are_construct_specific(
        self, two_construct_structure, two_construct_model
    ):
        """Construct-specific sigma priors compile to per-latent diffusion scales."""
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 3.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.15}},
            "sigma_mood": {"distribution": "HalfNormal", "params": {"sigma": 0.1}},
            "sigma_stress": {"distribution": "HalfNormal", "params": {"sigma": 0.9}},
            "lambda_stress_cortisol_stress": {
                "distribution": "Normal",
                "params": {"mu": 0.8, "sigma": 0.2},
            },
        }

        ssm_priors, _idx = _compile_priors_for_test(
            priors,
            two_construct_model,
        )

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
        ssm_priors, _idx = _compile_priors_for_test(
            weekly_study_priors,
            two_construct_model,
        )

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

    def test_ct_dynamics_is_stable(
        self,
        two_construct_structure,
        two_construct_model,
        weekly_study_priors,
    ):
        """The CT dynamics matrix from converted priors has all eigenvalues with Re < 0."""
        ssm_priors, _idx = _compile_priors_for_test(
            weekly_study_priors,
            two_construct_model,
        )

        dynamics = np.asarray(_reference_dynamics_from_priors(two_construct_model, ssm_priors))

        # All eigenvalues must have negative real parts (stability)
        eigenvalues = np.linalg.eigvals(dynamics)
        max_real = np.max(np.real(eigenvalues))
        assert max_real < 0, f"Dynamics matrix is unstable: max Re(eigenvalue) = {max_real}"

    def test_first_order_roundtrip_ar(
        self,
        two_construct_structure,
        two_construct_model,
        weekly_study_priors,
    ):
        """Resolved AR persistence follows component-owned decay rates."""
        ssm_priors, _idx = _compile_priors_for_test(
            weekly_study_priors,
            two_construct_model,
        )

        dynamics = _reference_dynamics_from_priors(two_construct_model, ssm_priors)

        # Discretize at dt=7 (weekly). The mood prior was authored on the
        # weekly interval, so its diagonal transition recovers that persistence.
        dt_weekly = 7.0
        F_weekly = jla.expm(dynamics * dt_weekly)

        decay_rate = _decay_reference_values(two_construct_model, ssm_priors)
        baseline_ar_mood = 3.0 / 5.0  # Beta(3,2) mean = 0.6
        expected_resolved_mood = math.exp(-decay_rate[1] * dt_weekly)
        recovered_ar_mood = float(F_weekly[1, 1])
        assert recovered_ar_mood == pytest.approx(baseline_ar_mood, abs=0.05)
        assert abs(recovered_ar_mood - expected_resolved_mood) < 0.05, (
            f"Weekly resolved mood AR: got {recovered_ar_mood:.4f}, "
            f"expected ≈{expected_resolved_mood:.4f}"
        )

        # stress: dt=1 for this prior, evaluated over a weekly interval.
        recovered_ar_stress = float(F_weekly[0, 0])
        assert recovered_ar_stress < 0.05, (
            f"Stress AR at weekly interval should be very low (daily-derived rate), "
            f"got {recovered_ar_stress:.4f}"
        )

        # Discretize at dt=1 (daily) for stress resolved persistence.
        F_daily = jla.expm(dynamics * 1.0)
        expected_daily_stress = math.exp(-decay_rate[0])
        recovered_daily_stress = float(F_daily[0, 0])
        assert abs(recovered_daily_stress - expected_daily_stress) < 0.05, (
            f"Daily roundtrip stress AR: got {recovered_daily_stress:.4f}, "
            f"expected ≈{expected_daily_stress:.4f}"
        )

    def test_first_order_roundtrip_cross_lag(
        self,
        two_construct_structure,
        two_construct_model,
        weekly_study_priors,
    ):
        """DT→CT→DT roundtrip for cross-lagged coefficient.

        beta_stress_mood = 0.3 from weekly study
        → CT rate = 0.3/7 → discretize at dt=7 → F[mood,stress] ≈ 0.3
        (first-order approximation; exact requires matrix exponential)
        """
        ssm_priors, _idx = _compile_priors_for_test(
            weekly_study_priors,
            two_construct_model,
        )

        # Build dynamics matrix
        dynamics = _reference_dynamics_from_priors(two_construct_model, ssm_priors)

        # Discretize at weekly interval
        dt_weekly = 7.0
        F_weekly = jla.expm(dynamics * dt_weekly)

        # NOTE: F[mood,stress] ≠ β_DT because the matrix exponential mixes terms.
        # For different diagonal entries, this is not simply A[mood,stress]*dt.
        # The exact DT→CT→DT roundtrip requires the matrix logarithm.
        #
        # What we CAN verify at first order:
        # 1. The CT rate was computed correctly (tested in test_dt_to_ct_uses_reference_interval_days)
        # 2. The coupling direction is preserved (F[mood,stress] > 0).
        # 3. The exact logm(F)/dt recovers the original A.
        recovered_coupling = float(F_weekly[1, 0])
        assert recovered_coupling > 0, (
            f"Coupling direction should be positive (stress→mood), got {recovered_coupling:.4f}"
        )
        # Verify via exact logm roundtrip
        from scipy.linalg import logm

        A_recovered = logm(np.array(F_weekly)).real / dt_weekly
        ct_rate = float(dynamics[1, 0])  # the CT rate we set
        assert abs(A_recovered[1, 0] - ct_rate) < 1e-6, (
            f"Exact logm roundtrip: got {A_recovered[1, 0]:.6f}, expected {ct_rate:.6f}"
        )

    def test_discretize_produces_valid_system(
        self,
        two_construct_structure,
        two_construct_model,
        weekly_study_priors,
    ):
        """discretize_linear_system_exact produces valid F, Q, c from converted priors."""
        ssm_priors, _idx = _compile_priors_for_test(
            weekly_study_priors,
            two_construct_model,
        )

        # Build dynamics and diffusion at prior means
        n = numeric.n_states(two_construct_model)
        dynamics = _reference_dynamics_from_priors(two_construct_model, ssm_priors)

        # Simple diagonal diffusion
        diff_sd = _prior_law(ssm_priors, "diffusion_diag_free").scale
        diff_sd_arr = jnp.asarray(diff_sd, dtype=jnp.float32)
        diffusion_cov = jnp.diag(diff_sd_arr**2)

        # CINT (zeros)
        cint = jnp.zeros(n)

        # Discretize at dt=1 (daily)
        parameters = affine_test_evolution(dynamics, diffusion_cov, cint).params_at(0.0, 1.0)
        F, Q, c = parameters.A, parameters.cov, parameters.bias

        # F should be a valid transition matrix (all eigenvalues < 1 in abs)
        eigs_F = jnp.linalg.eigvals(F)
        assert jnp.all(jnp.abs(eigs_F) < 1.0 + 1e-6), (
            f"F has eigenvalues outside unit circle: {eigs_F}"
        )

        # Q should be symmetric positive semi-definite
        assert jnp.allclose(Q, Q.T, atol=1e-6), "Q is not symmetric"
        eigs_Q = jnp.linalg.eigvalsh(Q)
        assert jnp.all(eigs_Q >= -1e-6), f"Q has negative eigenvalues: {eigs_Q}"

        # No NaN/Inf
        assert jnp.all(jnp.isfinite(F)), "F contains NaN/Inf"
        assert jnp.all(jnp.isfinite(Q)), "Q contains NaN/Inf"
        assert c is not None, "c should not be None when cint is provided"
        assert jnp.all(jnp.isfinite(c)), "c contains NaN/Inf"

    @pytest.mark.predictive
    def test_prior_predictive_produces_finite_samples(
        self,
        two_construct_structure,
        two_construct_model,
        weekly_study_priors,
    ):
        """Prior predictive sampling produces finite, bounded outputs."""
        import polars as pl

        from nof1_causal_lab.models.ssm.inference import prior_predictive

        n_time = 30
        rng = np.random.default_rng(0)
        mock_data = pl.DataFrame(
            {
                "mood_rating": rng.normal(5, 1.5, n_time),
                "stress_self_report": rng.normal(5, 1.5, n_time),
                "stress_cortisol": rng.normal(10, 2, n_time),
                "time": np.arange(n_time, dtype=float),
            }
        )
        from nof1_causal_lab.models.model_checks import check_execution
        from nof1_causal_lab.models.ssm.runtime import build_ssm_model
        from tests.helpers import make_prior_model

        typed_scientific_model = ModelSpec.model_validate(two_construct_model)
        check_execution(
            make_prior_model(typed_scientific_model, weekly_study_priors),
        )
        model = build_ssm_model(
            mock_data, model_spec=make_prior_model(typed_scientific_model, weekly_study_priors)
        )

        # Sample from prior predictive
        times = jnp.arange(n_time, dtype=jnp.float32)
        samples = prior_predictive(model, times, num_samples=20, seed=42)

        # Check key component-owned dynamics sites exist and are finite.
        dynamics_keys = [key for key in samples if key.startswith("vf_")]
        assert dynamics_keys, "Missing component-owned dynamics sites in prior predictive samples"
        for key in dynamics_keys:
            assert jnp.all(jnp.isfinite(samples[key])), f"{key} samples contain NaN/Inf"

        if "diffusion" in samples:
            diff_samples = samples["diffusion"]
            assert jnp.all(jnp.isfinite(diff_samples)), "diffusion samples contain NaN/Inf"

        # Decay rates should stay on positive support, giving negative diagonal dynamics terms.
        for key in [name for name in dynamics_keys if name.endswith("_decay")]:
            assert jnp.all(samples[key] > 0), f"{key} has non-positive decay draws"

    def test_different_intervals_produce_different_rates(self, two_construct_model):
        """Same DT beta at different study intervals → different CT rates.

        beta=0.3 from weekly (dt=7) → CT rate ≈ 0.043
        beta=0.3 from daily  (dt=1) → CT rate ≈ 0.300
        This is the Kuiper & Ryan (2018) sign-reversal effect in action.
        """

        scientific_model = two_construct_model

        # Weekly study priors
        priors_weekly = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
                "reference_interval_days": 7.0,
            },
        }
        # Daily study priors (same beta value, different interval)
        priors_daily = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
                "reference_interval_days": 1.0,
            },
        }

        source_model, _ = (scientific_model, numeric.edge_lag_days(scientific_model))

        ssm_priors_w, _idx = _compile_priors_for_test(
            priors_weekly,
            scientific_model,
        )

        ssm_priors_d, _idx = _compile_priors_for_test(
            priors_daily,
            scientific_model,
        )

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
# Prior factorization and edge-lag metadata
# ═══════════════════════════════════════════════════════════════════════


class TestPriorCompilationMetadata:
    """Compiler-owned prior factorization and edge-lag contracts."""

    def test_compile_keeps_elementwise_priors_when_intervals_match(self, two_construct_structure):
        """Compilation keeps factorized DT→CT priors even when dt values match."""
        scientific_model = complete_test_model(two_construct_structure)

        # All parameters at dt=7 (weekly)
        priors = {
            "rho_mood": {
                "distribution": "Beta",
                "params": {"alpha": 3.0, "beta": 2.0},
                "reference_interval_days": 7.0,
            },
            "rho_stress": {
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "reference_interval_days": 7.0,
            },
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
                "reference_interval_days": 7.0,
            },
        }

        source_model, _ = (scientific_model, numeric.edge_lag_days(scientific_model))

        ssm_priors, _idx = _compile_priors_for_test(
            priors,
            scientific_model,
        )

        dynamics_decay = _decay_reference_values(source_model, ssm_priors)
        linear_edge_weight = _linear_edge_weight(source_model, ssm_priors, source=0, target=1)

        assert abs(dynamics_decay[1] - (-math.log(0.6) / 7.0)) < 0.01
        assert abs(dynamics_decay[0] - (-math.log(0.5) / 7.0)) < 0.01
        assert abs(linear_edge_weight - (0.3 / 7.0)) < 0.01

    def test_edge_lag_days_populated(self, two_construct_structure):
        """Compilation stores edge lag metadata from causal design during support building."""
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_model,
        )

        _dm, _lm, _lmask, _cat, edge_lag_days = build_structural_support_from_model(
            ["mood", "stress"],
            ["mood_rating", "stress_self_report"],
            2,
            2,
            manifest_dists=[DistributionFamily.GAUSSIAN] * 2,
            model=two_construct_structure,
        )

        # stress -> mood edge, both daily, lagged=True: lag = 24h = 1.0 day
        assert len(edge_lag_days) == 1
        # effect_idx=0 (mood), cause_idx=1 (stress)
        assert (0, 1) in edge_lag_days
        assert abs(edge_lag_days[(0, 1)] - 1.0) < 0.01

    def test_dynamics_lag_consistency_warns(self, two_construct_structure, caplog):
        """Compilation warns when CT dynamics implies timescale far from edge lag."""
        import logging

        scientific_model = complete_test_model(two_construct_structure)
        # Very large beta → CT rate implies very fast coupling (short timescale)
        # but edge lag is 1 day → should warn about mismatch
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 6.0, "sigma": 1.0},
            },
        }

        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_model,
        )

        _dm, _lm, _lmask, _cat, edge_lag_days = build_structural_support_from_model(
            ["mood", "stress"],
            ["mood_rating", "stress_self_report"],
            2,
            2,
            manifest_dists=[DistributionFamily.GAUSSIAN] * 2,
            model=two_construct_structure,
        )
        with caplog.at_level(logging.WARNING, logger="nof1_causal_lab.models.ssm.compile.inputs"):
            _compile_priors_for_test(
                priors,
                scientific_model,
                edge_lag_days=edge_lag_days,
            )

        # Large beta_CT → implied timescale << 1 day, edge lag = 1 day → warning
        lag_warnings = [r for r in caplog.records if "mismatch" in r.message.lower()]
        assert len(lag_warnings) >= 1, (
            f"Expected lag mismatch warning, got: {[r.message for r in caplog.records]}"
        )
