"""Tests for SSM runtime preparation helpers.

Covers: semantic prior binding and fit-input preparation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
import pytest

from nof1_causal_lab.artifacts.compiled_ssm import CompiledSSMArtifact
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.artifacts.statistical_model_spec import (
    DistributionFamily,
    LinkFunction,
    StatisticalModelSpec,
)
from nof1_causal_lab.models.ssm.compile.inputs import (
    compile_priors,
)
from nof1_causal_lab.models.ssm.dynamics.spec import (
    DiagonalDecaySpec,
    DynamicsSpec,
    HillEdgeSpec,
    LinearEdgeSpec,
    MultiplicativeEdgeSpec,
)
from nof1_causal_lab.models.ssm.model import SSMModel, SSMSpec
from nof1_causal_lab.models.ssm.runtime import (
    build_ssm_model,
    prepare_fit_inputs,
    prepare_model_runtime,
    prepare_transition_inputs,
    sample_prior_predictive,
)
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    SparseMatrixBlockSpec,
    T0CholBlockSpec,
)
from tests.helpers import model_with_prior_payloads, named_prior_payloads, native_axis_metadata
from tests.ssm_spec_fixtures import (
    default_diffusion_block,
    default_input_effect_block,
    default_lambda_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
    dense_matrix_dynamics_spec,
    full_dense_matrix_dynamics_spec,
    full_diagonal_support,
)

if TYPE_CHECKING:
    from nof1_causal_lab.sampler_config import SamplerConfigOverride

# =============================================================================
# normalize_prior_params
# =============================================================================


def _make_spec(
    *,
    n_latent: int = 1,
    n_manifest: int = 1,
    dynamics_spec=None,
    diffusion_block=None,
    lambda_block=None,
    manifest_means_block=None,
    manifest_chol_block=None,
    t0_means_block=None,
    t0_chol_block=None,
    input_effect_block=None,
    static_state_sd_block=None,
    **kwargs,
) -> SSMSpec:
    """Build an SSMSpec from explicit block specs for tests."""
    if dynamics_spec is None:
        dynamics_spec = full_dense_matrix_dynamics_spec(n_latent)
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dynamics_spec,
        diffusion_block=diffusion_block or default_diffusion_block(n_latent),
        lambda_block=lambda_block or default_lambda_block(n_manifest, n_latent),
        manifest_means_block=manifest_means_block or default_manifest_means_block(n_manifest),
        manifest_chol_block=manifest_chol_block or default_manifest_chol_block(n_manifest),
        t0_means_block=t0_means_block or default_t0_means_block(n_latent),
        t0_chol_block=t0_chol_block or default_t0_chol_block(n_latent),
        input_effect_block=input_effect_block or default_input_effect_block(n_latent),
        static_state_sd_block=static_state_sd_block or default_static_state_sd_block(),
        **native_axis_metadata(n_latent, n_manifest, kwargs),
    )


class TestBuilderPriorConversion:
    def test_ar_prior_rejects_negative_support(self):
        """AR priors must stay on the DT persistence scale in (0, 1)."""
        statistical_model_spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:bbc87212909e45b9e6c3",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:fb33dbedf43eb15e324c86fa97201278e306cb48aa9752104361309d61215122",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                }
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:869e1d0209fb25a7fc06",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                }
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:fb33dbedf43eb15e324c86fa97201278e306cb48aa9752104361309d61215122",
                    "owners": [{"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                }
            ],
        }
        priors = {
            "rho_mood": {
                "distribution": "Uniform",
                "params": {"lower": -1.0, "upper": 1.0},
            }
        }
        ssm_spec = _make_spec(n_latent=1, n_manifest=1, latent_names=["mood"])

        with pytest.raises(ValueError, match="support within"):
            compile_priors(
                model_with_prior_payloads(
                    StatisticalModelSpec.model_validate(statistical_model_spec),
                    named_prior_payloads(
                        StatisticalModelSpec.model_validate(statistical_model_spec), priors
                    ),
                ),
                ssm_spec=ssm_spec,
            )

    def test_initial_state_correlation_priors_are_bounded_to_correlation_scale(self):
        """Initial-state correlations should compile to bounded correlation priors."""
        statistical_model_spec = {
            "mechanisms": [],
            "likelihoods": [
                {
                    "indicator_id": "indicator:869e1d0209fb25a7fc06",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:ea4ac346f9793711d3de",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "prior_transform": "initial_state_correlation",
                    "id": "parameter:9db0b63984f73620cea7336b8efb56bfb3765270a1a417f7958d9fe811f82907",
                    "owners": [
                        {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
                        {"kind": "construct", "id": "construct:cdc0b2958a9512b2abad"},
                    ],
                    "quantity": "t0_var_lower",
                    "name": "cor0_mood_sleep",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "",
                }
            ],
        }
        priors = {
            "cor0_mood_sleep": {
                "distribution": "Normal",
                "params": {"mu": 0.2, "sigma": 0.8},
            }
        }
        t0_mask = np.zeros((2, 2), dtype=bool)
        t0_mask[1, 0] = True
        ssm_spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "sleep"],
            manifest_names=["mood", "sleep"],
            t0_chol_block=T0CholBlockSpec(
                n_latent=2,
                diag_support=full_diagonal_support(2),
                correlation_support=t0_mask,
                template=jnp.eye(2),
            ),
        )

        prior_registry, _index_maps, _diagnostics = compile_priors(
            model_with_prior_payloads(
                StatisticalModelSpec.model_validate(statistical_model_spec),
                named_prior_payloads(
                    StatisticalModelSpec.model_validate(statistical_model_spec), priors
                ),
            ),
            ssm_spec=ssm_spec,
        )
        t0_corr_prior = prior_registry["t0_var_lower_free"]

        np.testing.assert_allclose(t0_corr_prior.base_dist.loc, [0.2])
        np.testing.assert_allclose(t0_corr_prior.base_dist.scale, [0.8])
        np.testing.assert_allclose(t0_corr_prior.low, [-1.0])
        np.testing.assert_allclose(t0_corr_prior.high, [1.0])

    def test_initial_state_mean_and_sd_priors_bind_to_t0_sites(self):
        """Authored initial-state priors should compile to the t0 mean/diag sites."""
        statistical_model_spec = {
            "mechanisms": [],
            "likelihoods": [
                {
                    "indicator_id": "indicator:869e1d0209fb25a7fc06",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:ea4ac346f9793711d3de",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "id": "parameter:dfbf9958e6c73f2233582b25872fd924a82ee5c04e55adb8acf0b44765b7b277",
                    "owners": [{"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"}],
                    "quantity": "t0_means",
                    "name": "t0_mean_mood",
                    "role": "initial_state_mean",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "id": "parameter:46608d1b7349d1939222ae192e6f6205f686d1837de6685cf34f12f590bcdb01",
                    "owners": [{"kind": "construct", "id": "construct:cdc0b2958a9512b2abad"}],
                    "quantity": "t0_means",
                    "name": "t0_mean_sleep",
                    "role": "initial_state_mean",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "id": "parameter:af08d5a5b9bc3378fe708023f1b98bb9cbe557ab23a3a147e084f76fc331cca9",
                    "owners": [{"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"}],
                    "quantity": "t0_var_diag",
                    "name": "t0_sd_mood",
                    "role": "initial_state_sd",
                    "constraint": "positive",
                    "description": "",
                },
                {
                    "id": "parameter:ebf0bf3a45ddac90be77292343d5ad604cdab82c43449a1c05601511575d3712",
                    "owners": [{"kind": "construct", "id": "construct:cdc0b2958a9512b2abad"}],
                    "quantity": "t0_var_diag",
                    "name": "t0_sd_sleep",
                    "role": "initial_state_sd",
                    "constraint": "positive",
                    "description": "",
                },
            ],
        }
        priors = {
            "t0_mean_mood": {
                "distribution": "Normal",
                "params": {"mu": 0.1, "sigma": 0.2},
            },
            "t0_mean_sleep": {
                "distribution": "Normal",
                "params": {"mu": -0.3, "sigma": 0.4},
            },
            "t0_sd_mood": {
                "distribution": "HalfNormal",
                "params": {"sigma": 0.7},
            },
            "t0_sd_sleep": {
                "distribution": "HalfNormal",
                "params": {"sigma": 0.9},
            },
        }
        ssm_spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "sleep"],
            manifest_names=["mood", "sleep"],
            t0_chol_block=T0CholBlockSpec(
                n_latent=2,
                diag_support=full_diagonal_support(2),
                correlation_support=np.zeros((2, 2), dtype=bool),
                template=jnp.eye(2),
            ),
        )

        prior_registry, index_maps, _diagnostics = compile_priors(
            model_with_prior_payloads(
                StatisticalModelSpec.model_validate(statistical_model_spec),
                named_prior_payloads(
                    StatisticalModelSpec.model_validate(statistical_model_spec), priors
                ),
            ),
            ssm_spec=ssm_spec,
        )
        t0_mean_prior = prior_registry["t0_means_free"]
        t0_diag_prior = prior_registry["t0_var_diag_free"]

        np.testing.assert_allclose(t0_mean_prior.loc, [0.1, -0.3])
        np.testing.assert_allclose(t0_mean_prior.scale, [0.2, 0.4])
        np.testing.assert_allclose(t0_diag_prior.scale, [0.7, 0.9])
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_mean_mood"
            ).site_name
            == "t0_means_free"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_mean_mood"
            ).prior_field
            == "t0_means"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_mean_mood"
            ).flat_index
            == 0
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_mean_sleep"
            ).site_name
            == "t0_means_free"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_mean_sleep"
            ).prior_field
            == "t0_means"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_mean_sleep"
            ).flat_index
            == 1
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_sd_mood"
            ).site_name
            == "t0_var_diag_free"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_sd_mood"
            ).prior_field
            == "t0_var_diag"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_sd_mood"
            ).flat_index
            == 0
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_sd_sleep"
            ).site_name
            == "t0_var_diag_free"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_sd_sleep"
            ).prior_field
            == "t0_var_diag"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "t0_sd_sleep"
            ).flat_index
            == 1
        )

    def test_initial_state_correlation_prior_indices_are_dense_after_mask_filtering(self):
        """Filtered initial-state pairs should not leave holes in prior arrays."""
        statistical_model_spec = {
            "mechanisms": [],
            "likelihoods": [
                {
                    "indicator_id": "indicator:47ff20b4fb0dc419ad4a",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:397bec1953295d24c291",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:413477ae6b4978c5c2fc",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "prior_transform": "initial_state_correlation",
                    "id": "parameter:2346e535ebf15e1a19af34d4fe9f6160b29a0a8e86fffe03c5066ffdb41ada7e",
                    "owners": [
                        {"kind": "construct", "id": "construct:c2061f241aafa2387cc9"},
                        {"kind": "construct", "id": "construct:efca9bd7f243a46c0d06"},
                    ],
                    "quantity": "t0_var_lower",
                    "name": "cor0_A_B",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "",
                },
                {
                    "prior_transform": "initial_state_correlation",
                    "id": "parameter:7c67bf719e1fd28a047bd9fee3468ca419fba5dfa1090e85c478afe79d5a1f91",
                    "owners": [
                        {"kind": "construct", "id": "construct:614026f43322f0daebd5"},
                        {"kind": "construct", "id": "construct:efca9bd7f243a46c0d06"},
                    ],
                    "quantity": "t0_var_lower",
                    "name": "cor0_C_B",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "",
                },
            ],
        }
        priors = {
            "cor0_C_B": {
                "distribution": "Normal",
                "params": {"mu": 0.1, "sigma": 0.2},
            }
        }
        t0_mask = np.zeros((3, 3), dtype=bool)
        t0_mask[2, 1] = True
        ssm_spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            latent_names=["A", "B", "C"],
            manifest_names=["a", "b", "c"],
            t0_chol_block=T0CholBlockSpec(
                n_latent=3,
                diag_support=full_diagonal_support(3),
                correlation_support=t0_mask,
                template=jnp.eye(3),
            ),
        )

        # Only the retained free covariance coordinate is a scientific parameter.
        statistical_model_spec["parameters"] = statistical_model_spec["parameters"][1:]
        prior_registry, index_maps, _diagnostics = compile_priors(
            model_with_prior_payloads(
                StatisticalModelSpec.model_validate(statistical_model_spec),
                named_prior_payloads(
                    StatisticalModelSpec.model_validate(statistical_model_spec), priors
                ),
            ),
            ssm_spec=ssm_spec,
        )
        t0_corr_prior = prior_registry["t0_var_lower_free"]

        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "cor0_C_B"
            ).site_name
            == "t0_var_lower_free"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "cor0_C_B"
            ).prior_field
            == "t0_var_offdiag"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "cor0_C_B"
            ).flat_index
            == 0
        )
        np.testing.assert_allclose(t0_corr_prior.base_dist.loc, [0.1])
        np.testing.assert_allclose(t0_corr_prior.base_dist.scale, [0.2])
        np.testing.assert_allclose(t0_corr_prior.low, [-1.0])
        np.testing.assert_allclose(t0_corr_prior.high, [1.0])

    def test_component_dynamics_parameters_bind_to_component_sites(self):
        """Semantic priors should bind to component-owned dynamics sites."""
        statistical_model_spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:d109b56ae5a2d4eadc3e",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:f0aaf80847be94589405e83538407c2b5964a7877fd3952ab64a8188724ccc7a",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                }
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:10cfd825f8eacc872e49",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:9d6c9ae51b644196d9d2",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:f0aaf80847be94589405e83538407c2b5964a7877fd3952ab64a8188724ccc7a",
                    "owners": [{"kind": "construct", "id": "construct:d109b56ae5a2d4eadc3e"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_response",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "prior_transform": "dt_effect_to_ct_rate",
                    "id": "parameter:14a490479a1936aba219f86e583ac50d843460fb85a58d39264755087678bb97",
                    "owners": [
                        {"kind": "construct", "id": "construct:16176a18c25802dee8a1"},
                        {"kind": "construct", "id": "construct:d109b56ae5a2d4eadc3e"},
                    ],
                    "quantity": "dynamics_weight",
                    "name": "beta_dose_response",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "id": "parameter:dcb114718cbe240842a4ff61487f162194aa5824b97e6b6ed2f9c6eb3b498da7",
                    "owners": [
                        {"kind": "construct", "id": "construct:16176a18c25802dee8a1"},
                        {"kind": "construct", "id": "construct:d109b56ae5a2d4eadc3e"},
                    ],
                    "quantity": "hill_emax",
                    "name": "hill_emax_dose_response",
                    "role": "dynamics_parameter_positive",
                    "constraint": "positive",
                    "description": "",
                },
                {
                    "id": "parameter:af64bcd906834b25bcd6e15fd71e54434132fa8b5ad9d7a50cd46b928f063dd8",
                    "owners": [
                        {"kind": "construct", "id": "construct:16176a18c25802dee8a1"},
                        {"kind": "construct", "id": "construct:d109b56ae5a2d4eadc3e"},
                    ],
                    "quantity": "hill_n",
                    "name": "hill_n_dose_response",
                    "role": "dynamics_parameter",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_response": {
                "distribution": "Beta",
                "params": {"alpha": 4.0, "beta": 1.0},
            },
            "beta_dose_response": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.2},
                "reference_interval_days": 2.0,
            },
            "hill_emax_dose_response": {
                "distribution": "HalfNormal",
                "params": {"sigma": 1.5},
            },
            "hill_n_dose_response": {
                "distribution": "TruncatedNormal",
                "params": {"mu": 2.0, "sigma": 0.3, "lower": 1.0, "upper": 4.0},
            },
        }
        ssm_spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["dose", "response"],
            manifest_names=["dose", "response"],
            dynamics_spec=DynamicsSpec(
                n_latent=2,
                components=(
                    DiagonalDecaySpec(),
                    LinearEdgeSpec(source=0, target=1),
                    HillEdgeSpec(source=0, target=1),
                ),
            ),
        )

        prior_registry, index_maps, _diagnostics = compile_priors(
            model_with_prior_payloads(
                StatisticalModelSpec.model_validate(statistical_model_spec),
                named_prior_payloads(
                    StatisticalModelSpec.model_validate(statistical_model_spec), priors
                ),
            ),
            ssm_spec=ssm_spec,
        )

        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "rho_response"
            ).site_name
            == "vf_0_decay"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "rho_response"
            ).prior_field
            == "dynamics_decay"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "rho_response"
            ).flat_index
            == 1
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "beta_dose_response"
            ).site_name
            == "vf_1_weight"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "beta_dose_response"
            ).prior_field
            == "linear_edge_weight"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "hill_emax_dose_response"
            ).site_name
            == "vf_2_Emax"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "hill_emax_dose_response"
            ).prior_field
            == "hill_emax"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "hill_n_dose_response"
            ).site_name
            == "vf_2_n"
        )
        assert (
            next(
                binding
                for binding in index_maps.by_parameter.values()
                if binding.parameter_name == "hill_n_dose_response"
            ).prior_field
            == "hill_n"
        )

        linear_prior = prior_registry["vf_1_weight"]
        hill_emax_prior = prior_registry["vf_2_Emax"]
        hill_n_prior = prior_registry["vf_2_n"]

        assert float(linear_prior.log_prob(0.2)) == pytest.approx(
            float(dist.Normal(0.15, 0.1).log_prob(0.2))
        )
        assert hill_emax_prior.scale == pytest.approx(1.5)
        assert hill_n_prior.base_dist.loc == pytest.approx(2.0)
        assert hill_n_prior.base_dist.scale == pytest.approx(0.3)
        assert hill_n_prior.low == pytest.approx(1.0)
        assert hill_n_prior.high == pytest.approx(4.0)

    def test_cross_lag_prior_requires_resolved_interval_metadata(self):
        """Cross-lag priors should fail instead of silently defaulting to 1 day."""
        statistical_model_spec = {
            "mechanisms": [],
            "likelihoods": [
                {
                    "indicator_id": "indicator:869e1d0209fb25a7fc06",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:78b974765eabb79741bb",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "prior_transform": "dt_effect_to_ct_rate",
                    "id": "parameter:570f5281d7fd14af134bd2c51b2100db631b6e326c46011e2cd0235805a4ac46",
                    "owners": [
                        {"kind": "construct", "id": "construct:6b04dc42c531e7091eb8"},
                        {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
                    ],
                    "quantity": "dynamics_weight",
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                }
            ],
        }
        priors = {
            "beta_stress_mood": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
            }
        }
        edge_support = np.zeros((2, 2), dtype=bool)
        edge_support[0, 1] = True
        ssm_spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            manifest_names=["mood", "stress"],
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=2,
                decay_support=full_diagonal_support(2),
                edge_support=edge_support,
                coupling_template=jnp.zeros((2, 2)),
                intercept_support=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
        )

        with pytest.raises(ValueError, match="could not resolve an authoring interval"):
            compile_priors(
                model_with_prior_payloads(
                    StatisticalModelSpec.model_validate(statistical_model_spec),
                    named_prior_payloads(
                        StatisticalModelSpec.model_validate(statistical_model_spec), priors
                    ),
                ),
                ssm_spec=ssm_spec,
            )

    @pytest.mark.cpu_expensive
    def test_prior_predictive_supports_hill_edge_spec(self):
        """The prior predictive path should accept nonlinear component dynamics."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["dose", "response"],
            manifest_names=["dose", "response"],
            dynamics_spec=DynamicsSpec(
                n_latent=2,
                components=(
                    DiagonalDecaySpec(),
                    HillEdgeSpec(
                        source=0,
                        target=1,
                    ),
                ),
            ),
        )
        model = build_ssm_model(
            pl.DataFrame({"time": [0.0], "dose": [0.0], "response": [0.0]}),
            ssm_spec=spec,
        )

        samples = sample_prior_predictive(
            model,
            samples=3,
            times=jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32),
        )

        assert samples["latents"].shape == (3, 4, 2)
        assert samples["linear_predictors"].shape == (3, 4, 2)
        assert samples["observations"].shape == (3, 4, 2)
        assert bool(jnp.isfinite(samples["observations"]).all())

    @pytest.mark.cpu_expensive
    def test_prior_predictive_observation_shape_matches_affine_and_nonlinear_specs(self):
        """Affine and nonlinear specs should use the same public predictive shape."""
        times = jnp.linspace(0.0, 1.0, 4, dtype=jnp.float32)
        affine_spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["a", "b"],
            manifest_names=["a", "b"],
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=2,
                decay_support=full_diagonal_support(2),
                edge_support=np.zeros((2, 2), dtype=bool),
                coupling_template=jnp.zeros((2, 2), dtype=jnp.float32),
                intercept_support=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2, dtype=jnp.float32),
            ),
        )
        nonlinear_spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["a", "b"],
            manifest_names=["a", "b"],
            dynamics_spec=DynamicsSpec(
                n_latent=2,
                components=(
                    DiagonalDecaySpec(),
                    MultiplicativeEdgeSpec(
                        source_a=0,
                        source_b=1,
                        target=1,
                    ),
                ),
            ),
        )

        affine_model = build_ssm_model(
            pl.DataFrame({"time": [0.0], "a": [0.0], "b": [0.0]}),
            ssm_spec=affine_spec,
        )
        nonlinear_model = build_ssm_model(
            pl.DataFrame({"time": [0.0], "a": [0.0], "b": [0.0]}),
            ssm_spec=nonlinear_spec,
        )
        affine_samples = sample_prior_predictive(
            affine_model,
            samples=2,
            times=times,
        )
        nonlinear_samples = sample_prior_predictive(
            nonlinear_model,
            samples=2,
            times=times,
        )

        assert affine_samples["observations"].shape == nonlinear_samples["observations"].shape
        assert affine_samples["observations"].shape == (2, 4, 2)


class TestObservationSupportValidation:
    def test_gamma_emission_rejects_zero_observations(self):
        """Gamma likelihoods must fail early when observed data include zeros."""
        X = pl.DataFrame({"time": [0, 1, 2], "screen_gap": [0.0, 1.0, 2.0]})
        spec = _make_spec(
            n_latent=1,
            n_manifest=1,
            latent_names=["screen_gap"],
            manifest_names=["screen_gap"],
            manifest_dists=[DistributionFamily.GAMMA],
            manifest_links=[LinkFunction.LOG],
        )

        with pytest.raises(ValueError, match="Observation support check failed"):
            build_ssm_model(X, ssm_spec=spec)


class TestPrepareFitInputs:
    def test_sparse_wide_nulls_become_nan_without_fill_forward(self):
        """Sparse wide cells should stay missing and never broadcast across ticks."""
        spec = _make_spec(n_latent=2, n_manifest=2, manifest_names=["x", "y"])
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0],
                "x": [10.0, None],
                "y": [None, 30.0],
            }
        )

        observations, times, manifest_names, _wide = prepare_fit_inputs(spec, wide)

        assert manifest_names == ["x", "y"]
        assert jnp.allclose(times, jnp.array([0.0, 1.0], dtype=jnp.float32))
        assert jnp.isclose(observations[0, 0], 10.0)
        assert jnp.isnan(observations[0, 1])
        assert jnp.isnan(observations[1, 0])
        assert jnp.isclose(observations[1, 1], 30.0)

    def test_manifest_standardization_applies_only_to_standardized_channels(self):
        """prepare_fit_inputs should deterministically standardize only marked manifests."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            manifest_names=["x", "y"],
            manifest_standardized=[True, False],
        )
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "x": [10.0, 12.0, 14.0],
                "y": [5.0, 6.0, 7.0],
            }
        )

        observations, times, manifest_names, _wide = prepare_fit_inputs(spec, wide)

        assert manifest_names == ["x", "y"]
        np.testing.assert_allclose(np.asarray(times), np.array([0.0, 1.0, 2.0]))
        np.testing.assert_allclose(
            np.asarray(observations[:, 0]), np.array([-1.0, 0.0, 1.0]), rtol=1e-6
        )
        np.testing.assert_allclose(np.asarray(observations[:, 1]), np.array([5.0, 6.0, 7.0]))

    def test_manifest_standardization_of_constant_column_centers_without_scaling(self):
        """A zero-variance standardized column becomes exactly zero (divisor 1)."""
        spec = _make_spec(
            n_latent=1,
            n_manifest=1,
            manifest_names=["x"],
            manifest_standardized=[True],
        )
        wide = pl.DataFrame({"time": [0.0, 1.0], "x": [4.2, 4.2]})

        observations, _times, _names, _wide = prepare_fit_inputs(spec, wide)

        np.testing.assert_allclose(np.asarray(observations[:, 0]), np.array([0.0, 0.0]))

    def test_transition_inputs_are_scaled_filled_and_shifted_to_interval_start(self):
        """Known inputs are deterministic transition covariates aligned to interval starts."""
        spec = _make_spec(
            n_latent=1,
            n_manifest=1,
            input_effect_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=1,
                free_support=np.array([[True]]),
                template=jnp.zeros((1, 1)),
                free_site_name="input_effect_free",
                det_site_name="input_effect",
                support=SupportClass.REAL,
                site_kind=SiteKind.INPUT_EFFECT,
                assembly_group="input_effect",
                fixed_spec_field="input_effect",
                priors_field="input_effect",
            ),
            input_names=["dose"],
            input_source_indicators=["dose_mg"],
            input_scales=[10.0],
            input_missing_policies=["forward_fill"],
            input_lagged=[True],
        )
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0],
                "dose_mg": [0.0, 20.0, None, 30.0],
                "mood_rating": [1.0, 2.0, 3.0, 4.0],
            }
        )

        transition_inputs = prepare_transition_inputs(spec, wide)

        assert transition_inputs is not None
        np.testing.assert_allclose(
            np.asarray(transition_inputs),
            np.array([[0.0], [0.0], [2.0], [2.0]], dtype=np.float32),
        )

    def test_contemporaneous_transition_inputs_are_not_shifted(self):
        spec = _make_spec(
            n_latent=1,
            n_manifest=1,
            input_effect_block=SparseMatrixBlockSpec(
                n_rows=1,
                n_cols=1,
                free_support=np.array([[True]]),
                template=jnp.zeros((1, 1)),
                free_site_name="input_effect_free",
                det_site_name="input_effect",
                support=SupportClass.REAL,
                site_kind=SiteKind.INPUT_EFFECT,
                assembly_group="input_effect",
                fixed_spec_field="input_effect",
                priors_field="input_effect",
            ),
            input_names=["dose"],
            input_source_indicators=["dose_mg"],
            input_scales=[10.0],
            input_missing_policies=["forward_fill"],
            input_lagged=[False],
        )
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0],
                "dose_mg": [0.0, 20.0, None, 30.0],
                "mood_rating": [1.0, 2.0, 3.0, 4.0],
            }
        )

        transition_inputs = prepare_transition_inputs(spec, wide)

        assert transition_inputs is not None
        np.testing.assert_allclose(
            np.asarray(transition_inputs),
            np.array([[0.0], [2.0], [2.0], [3.0]], dtype=np.float32),
        )


class TestPrepareModelRuntime:
    def test_support_boundary_rows_preserve_known_input_columns(self):
        data_for_model = pl.DataFrame(
            {
                "indicator_id": ["indicator:stress", "indicator:dose"],
                "value": [1.0, 20.0],
                "anchor_time": [
                    "2024-02-01T00:00:00",
                    "2024-02-01T00:00:00",
                ],
                "support_kind": ["interval", "point"],
                "summary_operator": ["mean", "last"],
                "anchor_policy": ["support_end", "support_end"],
                "observation_window": ["1mo", None],
                "support_start": [
                    "2024-01-01T00:00:00",
                    "2024-02-01T00:00:00",
                ],
                "support_end": [
                    "2024-02-01T00:00:00",
                    "2024-02-01T00:00:00",
                ],
            }
        )

        class StubModel:
            def __init__(self):
                self.spec = _make_spec(
                    n_latent=1,
                    n_manifest=1,
                    manifest_names=["stress_score"],
                    input_effect_block=SparseMatrixBlockSpec(
                        n_rows=1,
                        n_cols=1,
                        free_support=np.array([[True]]),
                        template=jnp.zeros((1, 1)),
                        free_site_name="input_effect_free",
                        det_site_name="input_effect",
                        support=SupportClass.REAL,
                        site_kind=SiteKind.INPUT_EFFECT,
                        assembly_group="input_effect",
                        fixed_spec_field="input_effect",
                        priors_field="input_effect",
                    ),
                    input_names=["dose"],
                    input_source_indicators=["dose_mg"],
                    input_scales=[10.0],
                    input_missing_policies=["forward_fill"],
                    input_lagged=[False],
                )
                self.parameter_layout = object()

            def set_observation_support(self, observation_support):
                self.observation_support = observation_support

            def set_transition_inputs(self, transition_inputs):
                self.transition_inputs = transition_inputs

        runtime = prepare_model_runtime(
            data_for_model,
            compiled_ssm=CompiledSSMArtifact.model_construct(
                observation_bindings={
                    "indicator:stress": "stress_score",
                    "indicator:dose": "dose_mg",
                }
            ),
            model=cast("SSMModel", StubModel()),
            sampler_config=cast(
                "SamplerConfigOverride",
                {"method": "marginal_particle_gibbs"},
            ),
        )

        assert runtime.wide_data.columns == ["time", "stress_score", "dose_mg"]
        assert runtime.wide_data["time"].to_list() == [-31.0, 0.0]
        assert runtime.wide_data["dose_mg"].to_list() == [None, 20.0]
        assert runtime.transition_inputs is not None
        np.testing.assert_allclose(
            np.asarray(runtime.transition_inputs),
            np.array([[0.0], [2.0]], dtype=np.float32),
        )

    def test_preserves_long_observation_metadata_and_augments_support_boundaries(self, caplog):
        data_for_model = pl.DataFrame(
            {
                "indicator_id": ["indicator:stress"],
                "value": [1.0],
                "anchor_time": ["2024-02-01T00:00:00"],
                "support_kind": ["interval"],
                "summary_operator": ["mean"],
                "anchor_policy": ["support_end"],
                "observation_window": ["1mo"],
                "support_start": ["2024-01-01T00:00:00"],
                "support_end": ["2024-02-01T00:00:00"],
            }
        )

        class StubModel:
            def __init__(self):
                self.observation_support = None
                self.spec = _make_spec(
                    n_latent=1,
                    n_manifest=1,
                    manifest_names=["stress_score"],
                )
                self.parameter_layout = object()

            def set_observation_support(self, observation_support):
                self.observation_support = observation_support

            def set_transition_inputs(self, transition_inputs):
                self.transition_inputs = transition_inputs

        with caplog.at_level("INFO"):
            runtime = prepare_model_runtime(
                data_for_model,
                compiled_ssm=CompiledSSMArtifact.model_construct(
                    observation_bindings={
                        "indicator:stress": "stress_score",
                        "indicator:dose": "dose_mg",
                    }
                ),
                model=cast("SSMModel", StubModel()),
                sampler_config=cast(
                    "SamplerConfigOverride",
                    {"method": "marginal_particle_gibbs"},
                ),
            )

        assert runtime.observation_data is not None
        assert (
            runtime.observation_data.columns
            == data_for_model.rename({"indicator_id": "indicator"}).columns
        )
        assert runtime.observation_data["observation_window"][0] == "1mo"
        assert runtime.observation_data["support_end"][0] == "2024-02-01T00:00:00"
        assert runtime.observation_data["anchor_time"][0] == "2024-02-01T00:00:00"
        assert runtime.wide_data["time"].to_list() == [-31.0, 0.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.manifest_names == ["stress_score"]
        assert runtime.observation_support.support_kinds == ["interval"]
        assert runtime.observation_support.summary_operators == ["mean"]
        assert runtime.observation_support.anchor_policies == ["support_end"]
        assert runtime.observation_support.observation_windows == ["1mo"]
        assert runtime.observation_support.requires_interval_summary_handling is True
        assert runtime.observation_support.interval_summary_manifest_names == ["stress_score"]
        assert runtime.observation_support.support_start_times.shape == (2, 1)
        assert runtime.observation_support.support_end_times.shape == (2, 1)
        assert runtime.observation_support.support_start_times[1, 0] == pytest.approx(-31.0)
        assert runtime.observation_support.support_end_times[1, 0] == pytest.approx(0.0)
        assert runtime.observation_support.interval_prev_coeffs.shape == (2, 1, 1)
        assert runtime.observation_support.interval_curr_coeffs.shape == (2, 1, 1)
        assert runtime.observation_support.interval_weights.shape == (2, 1, 1)
        assert runtime.observation_support.emission_slot_indices.tolist() == [[-1], [0]]
        assert runtime.observation_support.interval_prev_coeffs[1, 0, 0] == pytest.approx(15.5)
        assert runtime.observation_support.interval_curr_coeffs[1, 0, 0] == pytest.approx(15.5)
        assert runtime.observation_support.interval_weights[1, 0, 0] == pytest.approx(31.0)
        assert runtime.manifest_names == ["stress_score"]
        assert runtime.model.observation_support is runtime.observation_support
        assert runtime.inference_structure.structural_backend == "laplace"
        assert runtime.inference_structure.resolved_method == "marginal_particle_gibbs"
        assert runtime.inference_structure.method_override == "marginal_particle_gibbs"
        assert "support-aware observation semantics" in caplog.text

    def test_compiles_overlapping_interval_windows_into_concurrent_slots(self):
        data_for_model = pl.DataFrame(
            {
                "indicator_id": ["indicator:stress", "indicator:stress"],
                "value": [3.0, 5.0],
                "anchor_time": ["2024-01-03T00:00:00", "2024-01-04T00:00:00"],
                "support_kind": ["interval", "interval"],
                "summary_operator": ["mean", "mean"],
                "anchor_policy": ["support_end", "support_end"],
                "observation_window": ["2d", "2d"],
                "support_start": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
                "support_end": ["2024-01-03T00:00:00", "2024-01-04T00:00:00"],
            }
        )

        class StubModel:
            def __init__(self):
                self.observation_support = None
                self.spec = _make_spec(
                    n_latent=1,
                    n_manifest=1,
                    manifest_names=["stress_score"],
                )
                self.parameter_layout = object()

            def set_observation_support(self, observation_support):
                self.observation_support = observation_support

            def set_transition_inputs(self, transition_inputs):
                self.transition_inputs = transition_inputs

        runtime = prepare_model_runtime(
            data_for_model,
            compiled_ssm=CompiledSSMArtifact.model_construct(
                observation_bindings={
                    "indicator:stress": "stress_score",
                    "indicator:dose": "dose_mg",
                }
            ),
            model=cast("SSMModel", StubModel()),
            sampler_config=cast(
                "SamplerConfigOverride",
                {"method": "marginal_particle_gibbs"},
            ),
        )

        assert runtime.wide_data["time"].to_list() == [-2.0, -1.0, 0.0, 1.0]
        assert runtime.observation_support is not None
        assert runtime.observation_support.max_active_windows == 2
        assert runtime.inference_structure.structural_backend == "laplace"
        assert runtime.inference_structure.resolved_method == "marginal_particle_gibbs"
        assert runtime.observation_support.emission_slot_indices.tolist() == [[-1], [-1], [0], [1]]
        assert runtime.observation_support.interval_weights.shape == (4, 1, 2)
        assert runtime.observation_support.interval_weights[1, 0, 0] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[2, 0, 0] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[2, 0, 1] == pytest.approx(1.0)
        assert runtime.observation_support.interval_weights[3, 0, 1] == pytest.approx(1.0)

    @pytest.mark.cpu_expensive
    def test_prior_predictive_reuses_prepared_support_schedule(self):
        data_for_model = pl.DataFrame(
            {
                "indicator_id": ["indicator:stress"],
                "value": [1.0],
                "anchor_time": ["2024-02-01T00:00:00"],
                "support_kind": ["interval"],
                "summary_operator": ["mean"],
                "anchor_policy": ["support_end"],
                "observation_window": ["1mo"],
                "support_start": ["2024-01-01T00:00:00"],
                "support_end": ["2024-02-01T00:00:00"],
            }
        )
        model = build_ssm_model(
            pl.DataFrame({"time": [0.0], "stress_score": [1.0]}),
            ssm_spec=_make_spec(
                n_latent=1,
                n_manifest=1,
                diffusion_block=DiffusionBlockSpec(
                    n_latent=1,
                    diffusion_chol_support=np.diag(full_diagonal_support(1)),
                    diffusion_chol_template=jnp.eye(1, dtype=jnp.float32),
                ),
                manifest_names=["stress_score"],
            ),
        )
        runtime = prepare_model_runtime(
            data_for_model,
            compiled_ssm=CompiledSSMArtifact.model_construct(
                observation_bindings={
                    "indicator:stress": "stress_score",
                    "indicator:dose": "dose_mg",
                }
            ),
            model=model,
            sampler_config=cast(
                "SamplerConfigOverride",
                {"method": "marginal_particle_gibbs"},
            ),
        )

        samples = sample_prior_predictive(
            runtime.model,
            samples=3,
            times=runtime.times,
            observation_support=runtime.observation_support,
            observation_mask=~jnp.isnan(runtime.observations),
            transition_inputs=runtime.transition_inputs,
        )

        assert samples["observations"].shape == (3, 2, 1)
        assert samples["observations_mask"].shape == (3, 2, 1)
        assert jnp.isnan(samples["observations"][:, 0, 0]).all()
        assert jnp.isfinite(samples["observations"][:, 1, 0]).all()
        assert (~samples["observations_mask"][:, 0, 0]).all()
        assert samples["observations_mask"][:, 1, 0].all()
