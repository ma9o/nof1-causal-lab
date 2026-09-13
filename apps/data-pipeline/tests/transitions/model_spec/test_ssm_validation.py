"""Model-spec assembly and SSM compilation tests."""

from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.compiled_ssm import (
    CompiledSSMArtifact,
)
from nof1_causal_lab.artifacts.statistical_model_spec import StatisticalModelSpec
from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
from nof1_causal_lab.models.ssm import SSMSpec
from nof1_causal_lab.prior_distributions import (
    deserialize_distribution,
    prior_reference_value,
    serialize_distribution,
)
from tests.helpers import (
    make_prior_model,
    make_structural_plan,
    model_with_prior_payloads,
    named_prior_payloads,
    native_axis_metadata,
)
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
)
from tests.transitions.model_spec._support import (
    PriorValidationResult,
    _make_polars_data,
    _with_positive_indicator_polarity,
    compile_ssm_inputs_from_statistical_model_spec,
    compile_ssm_priors,
    np,
    patch,
    pl,
    pytest,
)


def _compile_structural_plan(causal_design: dict[str, Any]) -> StructuralPlan:
    from nof1_causal_lab.models.structural import build_structural_plan

    normalized = _with_positive_indicator_polarity(causal_design)
    constructs = normalized["latent"]["constructs"]
    for construct in constructs:
        construct.setdefault("description", construct["name"])
        construct.setdefault("role", "endogenous")
        construct.setdefault("temporal_status", "time_varying")
    for edge in normalized["latent"].get("edges", []):
        edge.setdefault(
            "description",
            f"{edge['cause_id']} causes {edge['effect_id']}",
        )
    for indicator in normalized["measurement"].get("indicators", []):
        indicator.setdefault("how_to_measure", f"Measure {indicator['name']}")
    return build_structural_plan(CausalDesign.model_validate(normalized))


def _typed_statistical_model_spec(payload: dict[str, Any]) -> StatisticalModelSpec:
    return StatisticalModelSpec.model_validate(payload)


def _mood_structural_plan() -> StructuralPlan:
    from nof1_causal_lab.models.structural import build_structural_plan

    return build_structural_plan(
        CausalDesign.model_validate(
            {
                "latent": {
                    "constructs": [
                        {
                            "id": "construct:bbc87212909e45b9e6c3",
                            "name": "mood",
                            "description": "Mood",
                            "role": "exogenous",
                            "temporal_status": "time_varying",
                        }
                    ],
                    "edges": [],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "id": "indicator:45f78731e3e0c6f3efe1",
                            "construct_id": "construct:bbc87212909e45b9e6c3",
                            "name": "mood_score",
                            "how_to_measure": "Mood score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "construct_polarity": "positive",
                        }
                    ],
                },
                "default_outcome": None,
            }
        )
    )


def _prior_for_parameter(priors, index_maps, parameter):
    binding = next(
        item for item in index_maps.by_parameter.values() if item.parameter_name == parameter
    )
    return deserialize_distribution(
        serialize_distribution(priors[binding.site_name])[binding.flat_index]
    )


def _prior_reference_for_parameter(priors, index_maps, parameter) -> float:
    return float(prior_reference_value(_prior_for_parameter(priors, index_maps, parameter)))


def _default_ssm_spec(
    *,
    n_latent: int,
    n_manifest: int,
    latent_names: list[str] | None = None,
    latent_ids: list[str] | None = None,
    edge_support=None,
) -> SSMSpec:
    """Build a SSMSpec with all default blocks plus optional dynamics support + names."""
    if edge_support is not None:
        dynamics_spec = dense_matrix_dynamics_spec(
            n_latent=n_latent,
            decay_support=np.ones(n_latent, dtype=bool),
            edge_support=np.asarray(edge_support, dtype=bool),
            coupling_template=jnp.zeros((n_latent, n_latent)),
            intercept_support=np.zeros(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent),
        )
    else:
        dynamics_spec = full_dense_matrix_dynamics_spec(n_latent)
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dynamics_spec,
        diffusion_block=default_diffusion_block(n_latent),
        lambda_block=default_lambda_block(n_manifest, n_latent),
        manifest_means_block=default_manifest_means_block(n_manifest),
        manifest_chol_block=default_manifest_chol_block(n_manifest),
        t0_means_block=default_t0_means_block(n_latent),
        t0_chol_block=default_t0_chol_block(n_latent),
        input_effect_block=default_input_effect_block(n_latent),
        static_state_sd_block=default_static_state_sd_block(),
        **native_axis_metadata(
            n_latent,
            n_manifest,
            {
                "latent_names": latent_names,
                **({"latent_ids": latent_ids} if latent_ids is not None else {}),
            },
        ),
    )


def _require_text(value: str | None) -> str:
    """Assert an optional diagnostic field is present before string matching."""
    assert value is not None
    return value


# --- SSM model construction tests ---


class TestSSMModelConstruction:
    """Test SSM model building."""

    def test_build_ssm_model_creates_model(
        self, simple_statistical_model_spec, simple_priors, simple_data
    ):
        """Runtime construction creates an SSMModel with correct dimensions."""
        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
        from nof1_causal_lab.models.ssm.runtime import hydrate_compiled_model

        statistical_model_spec = _typed_statistical_model_spec(simple_statistical_model_spec)
        compiled = compile_ssm_artifact(
            make_prior_model(statistical_model_spec, simple_priors),
            _mood_structural_plan(),
        )
        model = hydrate_compiled_model(compiled, pl.from_pandas(simple_data))
        assert model.spec.n_manifest == 1  # mood_score only
        assert model.spec.n_latent >= 1
        # Lambda should map latent to manifest
        assert model.spec.lambda_block.template.shape == (
            model.spec.n_manifest,
            model.spec.n_latent,
        )


# --- Model-spec assembly tests ---


class TestModelSpecAssembly:
    """Test compile-only assembly and final materialization."""

    def test_materialize_model_spec_result_persists_validation_warnings(
        self,
        simple_statistical_model_spec,
        simple_priors,
    ):
        """Final model-spec artifacts should carry non-fatal validation warnings."""
        from nof1_causal_lab.flows.transitions.model_spec.assembly import (
            AssemblyValidation,
            materialize_model_spec_result,
        )

        validation = AssemblyValidation(
            normalized_statistical_model_spec=simple_statistical_model_spec,
            compile_ok=True,
            diagnostics=[
                PriorValidationResult(
                    parameter="beta_stress_sleep",
                    is_valid=True,
                    code="interval_reference_missing",
                    origin="compile",
                    severity="warning",
                    issue="Weekly evidence is being interpreted on the daily model interval.",
                    suggested_adjustment=(
                        "Set `reference_interval_days` if that weekly interval is intended."
                    ),
                )
            ],
            compiled_ssm=CompiledSSMArtifact.model_construct(),
        )

        with (
            patch(
                "nof1_causal_lab.flows.transitions.model_spec.assembly.compile_model_artifact",
                return_value={
                    "model_built": True,
                    "model_type": "test",
                    "version": "0",
                    "compiled_ssm": SimpleNamespace(parameters=[]),
                },
            ),
            patch(
                "nof1_causal_lab.flows.transitions.model_spec.assembly.build_exact_prior_predictive_samples",
                return_value={},
            ),
        ):
            result = materialize_model_spec_result(
                statistical_model_spec=simple_statistical_model_spec,
                data_for_model=_make_polars_data(),
                indicator_audits=None,
                structural_plan=_mood_structural_plan(),
                validation=validation,
            )

        assert result["validation_warnings"] == [
            "Weekly evidence is being interpreted on the daily model interval."
        ]

    def test_validate_assembly_compiles_once(
        self,
        simple_statistical_model_spec,
        simple_priors,
    ):
        """Assembly compiles once and retains that artifact for materialization."""
        from nof1_causal_lab.flows.transitions.model_spec.assembly import validate_assembly

        compiled_artifact = SimpleNamespace(compile_diagnostics=[], parameters=[])
        with patch(
            "nof1_causal_lab.models.ssm.compile.artifact.compile_ssm_artifact",
            return_value=compiled_artifact,
        ) as compile_mock:
            validation = validate_assembly(
                simple_statistical_model_spec,
                _mood_structural_plan(),
            )

        assert compile_mock.call_count == 1
        assert validation.compiled_ssm == compiled_artifact

    def test_validate_assembly_keeps_lagged_prior_mismatches_as_warnings(
        self,
        simple_statistical_model_spec,
        simple_priors,
    ):
        """Lagged DT/CT heuristics should surface as warnings, not compile errors."""
        from nof1_causal_lab.flows.transitions.model_spec.assembly import validate_assembly

        compiled_artifact = SimpleNamespace(compile_diagnostics=[], parameters=[])

        with (
            patch(
                "nof1_causal_lab.models.ssm.compile.artifact.compile_ssm_artifact",
                return_value=compiled_artifact,
            ),
            patch(
                "nof1_causal_lab.flows.transitions.model_spec.assembly._collect_compile_diagnostics",
                return_value=[
                    PriorValidationResult(
                        parameter="beta_stress_sleep",
                        is_valid=True,
                        code="lagged_response_weak",
                        origin="compile",
                        severity="warning",
                        issue="Median one-lag response is much slower than the nominal lag.",
                        suggested_adjustment="Confirm that this slow response is intended.",
                    )
                ],
            ),
        ):
            validation = validate_assembly(
                simple_statistical_model_spec,
                _mood_structural_plan(),
            )

        assert validation.compile_ok is True
        assert [
            warning.model_dump()
            for warning in validation.compile_diagnostics
            if warning.severity == "warning"
        ] == [
            PriorValidationResult(
                parameter="beta_stress_sleep",
                is_valid=True,
                code="lagged_response_weak",
                origin="compile",
                severity="warning",
                issue="Median one-lag response is much slower than the nominal lag.",
                suggested_adjustment="Confirm that this slow response is intended.",
            ).model_dump()
        ]


# --- SSM Prior Conversion Tests ---


class TestSSMPriorConversion:
    """Test that priors with non-Normal distributions convert correctly."""

    def test_beta_persistence_compiles_to_its_exact_pushforward(
        self, simple_statistical_model_spec
    ):
        """Beta(2,2) AR prior converts via AR-to-dynamics transform."""
        import math

        priors = {
            "rho_mood": {
                "parameter": "rho_mood",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            },
        }
        ssm_spec = _default_ssm_spec(n_latent=1, n_manifest=1, latent_names=["mood"])
        ssm_priors, index_maps, _diagnostics = compile_ssm_priors(
            model_with_prior_payloads(
                _typed_statistical_model_spec(simple_statistical_model_spec),
                named_prior_payloads(
                    _typed_statistical_model_spec(simple_statistical_model_spec), priors
                ),
            ),
            ssm_spec=ssm_spec,
        )

        law = _prior_for_parameter(ssm_priors, index_maps, "rho_mood")
        decay = 0.7
        rho = math.exp(-decay)
        # Exact change of variables, including |d rho / d decay| = rho.
        expected = math.log(6.0) + 2.0 * math.log(rho) + math.log1p(-rho)
        assert float(law.log_prob(decay)) == pytest.approx(expected, abs=2e-6)
        key = jax.random.PRNGKey(6)
        np.testing.assert_array_equal(law.sample(key), -jnp.log(dist.Beta(2.0, 2.0).sample(key)))

    def test_structured_prior_requires_structural_binding_for_residual_sd(
        self, simple_statistical_model_spec
    ):
        """Structured priors should fail without a translated SSM binding."""
        priors = {
            "sigma_mood": {
                "parameter_id": "parameter:146688c9f8e2c980c9e7963be61deb23225a81f828c204339c2164d1f51d441e",
                "distribution": "HalfNormal",
                "params": {"sigma": 0.5},
                "sources": [],
                "reasoning": "test",
            },
        }
        with pytest.raises(ValueError, match="requires a translated SSMSpec"):
            compile_ssm_priors(
                model_with_prior_payloads(
                    _typed_statistical_model_spec(simple_statistical_model_spec),
                    named_prior_payloads(
                        _typed_statistical_model_spec(simple_statistical_model_spec), priors
                    ),
                ),
                ssm_spec=None,
            )

    def test_compile_ssm_inputs_accepts_validated_spec_without_revalidation(
        self, simple_statistical_model_spec, simple_priors
    ):
        """Typed compilation should consume an already validated spec without reparsing it."""
        statistical_model_spec = _typed_statistical_model_spec(simple_statistical_model_spec)

        with patch.object(
            StatisticalModelSpec, "model_validate", wraps=StatisticalModelSpec.model_validate
        ) as validate:
            compile_ssm_inputs_from_statistical_model_spec(
                make_prior_model(statistical_model_spec, simple_priors),
                structural_plan=_mood_structural_plan(),
            )

        assert validate.call_count == 0

    def test_structured_prior_requires_structural_binding_for_loading(
        self, simple_statistical_model_spec
    ):
        """Loading priors should fail without a translated SSM binding."""
        spec = dict(simple_statistical_model_spec)
        spec["mechanisms"] = []
        spec["parameters"] = [
            {
                "id": "parameter:98e602bee17aefa3acdb6d1dc6b3fa919680153289ded2a6ae2a677d163c31c2",
                "owners": [
                    {"kind": "indicator", "id": "indicator:869e1d0209fb25a7fc06"},
                    {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
                ],
                "quantity": "loading",
                "name": "lambda_mood",
                "role": "loading",
                "constraint": "positive",
                "description": "Factor loading",
            },
        ]
        priors = {
            "lambda_mood": {
                "parameter": "lambda_mood",
                "distribution": "HalfNormal",
                "params": {"sigma": 0.8},
                "sources": [],
                "reasoning": "test",
            },
        }
        with pytest.raises(ValueError, match="requires a translated SSMSpec"):
            compile_ssm_priors(
                model_with_prior_payloads(
                    _typed_statistical_model_spec(spec),
                    named_prior_payloads(_typed_statistical_model_spec(spec), priors),
                ),
                ssm_spec=None,
            )

    def test_compile_priors_rejects_invalid_persistence_support(self):
        """Independent prior compile failures should be reported together."""

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
                "params": {"lower": -1.0, "upper": 2.0},
            },
            "parameter:unknown": {
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 1.0},
            },
        }
        ssm_spec = _default_ssm_spec(n_latent=1, n_manifest=1, latent_names=["mood"])

        with pytest.raises(ValueError, match=r"support within \[0, 1\]") as exc_info:
            compile_ssm_priors(
                model_with_prior_payloads(
                    _typed_statistical_model_spec(statistical_model_spec),
                    named_prior_payloads(
                        _typed_statistical_model_spec(statistical_model_spec),
                        {"rho_mood": priors["rho_mood"]},
                    ),
                ),
                ssm_spec=ssm_spec,
            )

        message = str(exc_info.value)
        assert "support within [0, 1]" in message
        assert "support within [0, 1]" in message

    def test_compile_ssm_artifact_rejects_unknown_scientific_owners(self):
        """Invalid scientific ownership is rejected before compilation."""
        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact

        causal_design = _compile_structural_plan(
            {
                "latent": {
                    "default_outcome": {
                        "kind": "construct",
                        "id": "construct:bbc87212909e45b9e6c3",
                    },
                    "constructs": [
                        {
                            "id": "construct:bbc87212909e45b9e6c3",
                            "name": "mood",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                        {
                            "id": "construct:6b04dc42c531e7091eb8",
                            "name": "stress",
                            "role": "exogenous",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [
                        {
                            "id": "edge:923689028b6b177617c2",
                            "cause_id": "construct:6b04dc42c531e7091eb8",
                            "effect_id": "construct:bbc87212909e45b9e6c3",
                        }
                    ],
                },
                "estimation": {
                    "state_order": ["mood", "stress"],
                    "edges": [{"cause": "stress", "effect": "mood"}],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "id": "indicator:45f78731e3e0c6f3efe1",
                            "construct_id": "construct:bbc87212909e45b9e6c3",
                            "name": "mood_score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use mood_score directly",
                        },
                        {
                            "id": "indicator:3696aef3ff6f446744e5",
                            "construct_id": "construct:6b04dc42c531e7091eb8",
                            "name": "stress_score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use stress_score directly",
                        },
                    ],
                },
            }
        )
        statistical_model_spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:05cde79d2d36a58cfb68",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:06d9cf69961c5af677a82e143f7992004a8b9b27c8878ed5e7390deb65160dcc",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
                {
                    "kind": "node_potential",
                    "target_id": "construct:6b04dc42c531e7091eb8",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:aa675637a0e802f0cb93b867b6112c3e017a59da1b5c5c51af028a0f8671a86c",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:45f78731e3e0c6f3efe1",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "Continuous score",
                },
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "Continuous score",
                },
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:06d9cf69961c5af677a82e143f7992004a8b9b27c8878ed5e7390deb65160dcc",
                    "owners": [{"kind": "construct", "id": "construct:05cde79d2d36a58cfb68"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_affect",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "Invalid AR name",
                },
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:aa675637a0e802f0cb93b867b6112c3e017a59da1b5c5c51af028a0f8671a86c",
                    "owners": [{"kind": "construct", "id": "construct:6b04dc42c531e7091eb8"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "Valid AR name",
                },
                {
                    "prior_transform": "dt_effect_to_ct_rate",
                    "id": "parameter:570f5281d7fd14af134bd2c51b2100db631b6e326c46011e2cd0235805a4ac46",
                    "owners": [
                        {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
                        {"kind": "construct", "id": "construct:6b04dc42c531e7091eb8"},
                    ],
                    "quantity": "dynamics_weight",
                    "name": "beta_mood_stress",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "Wrong causal direction",
                },
            ],
        }
        priors = {
            "rho_affect": {
                "parameter": "rho_affect",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            },
            "rho_stress": {
                "parameter": "rho_stress",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "test",
            },
            "beta_mood_stress": {
                "parameter": "beta_mood_stress",
                "distribution": "Normal",
                "params": {"mu": 0.0, "sigma": 0.5},
                "sources": [],
                "reasoning": "test",
            },
        }

        with pytest.raises(ValueError, match="unknown scientific owner") as exc_info:
            typed_statistical_model_spec = _typed_statistical_model_spec(statistical_model_spec)
            compile_ssm_artifact(
                make_prior_model(typed_statistical_model_spec, priors),
                structural_plan=causal_design,
            )

        message = str(exc_info.value)
        assert "unknown scientific owner" in message
        assert "rho_affect" in message

    def test_multiple_ar_params_produce_per_element_decay_rate(self):
        """Multiple AR params map to separate dynamics-decay entries."""
        import math

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
                },
                {
                    "kind": "node_potential",
                    "target_id": "construct:6b04dc42c531e7091eb8",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:aa675637a0e802f0cb93b867b6112c3e017a59da1b5c5c51af028a0f8671a86c",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:45f78731e3e0c6f3efe1",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
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
                },
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:aa675637a0e802f0cb93b867b6112c3e017a59da1b5c5c51af028a0f8671a86c",
                    "owners": [{"kind": "construct", "id": "construct:6b04dc42c531e7091eb8"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 5.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 5.0}},
        }
        ssm_spec = _default_ssm_spec(n_latent=2, n_manifest=2, latent_names=["mood", "stress"])
        ssm_priors, index_maps, _diagnostics = compile_ssm_priors(
            model_with_prior_payloads(
                _typed_statistical_model_spec(statistical_model_spec),
                named_prior_payloads(_typed_statistical_model_spec(statistical_model_spec), priors),
            ),
            ssm_spec=ssm_spec,
        )

        # Beta(5,2) → E=5/7≈0.714, Beta(2,5) → E=2/7≈0.286
        mu_ar_mood = 5.0 / 7.0
        mu_ar_stress = 2.0 / 7.0
        expected_mood = -math.log(mu_ar_mood) / 1.0
        expected_stress = -math.log(mu_ar_stress) / 1.0
        assert (
            abs(_prior_reference_for_parameter(ssm_priors, index_maps, "rho_mood") - expected_mood)
            < 0.01
        )
        assert (
            abs(
                _prior_reference_for_parameter(ssm_priors, index_maps, "rho_stress")
                - expected_stress
            )
            < 0.01
        )

    def test_ar_transform_respects_granularity(self):
        """Hourly construct → dt=1/24, producing larger dynamics magnitude."""
        import math

        statistical_model_spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:4fe59e1f780aa9e8b426",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:8c3c4cb5c945ed0af65483504f2eaf8fe782152b4cfd2aee8d2a38e9bd17533b",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                }
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:752d1cddd25bc5570a84",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:8c3c4cb5c945ed0af65483504f2eaf8fe782152b4cfd2aee8d2a38e9bd17533b",
                    "owners": [{"kind": "construct", "id": "construct:4fe59e1f780aa9e8b426"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
        }
        causal_design = make_structural_plan(["heart_rate"], [])
        causal_design["semantics"]["model_clock"] = "1h"
        ssm_spec = _default_ssm_spec(n_latent=1, n_manifest=1, latent_names=["heart_rate"])
        ssm_priors, index_maps, _diagnostics = compile_ssm_priors(
            model_with_prior_payloads(
                _typed_statistical_model_spec(statistical_model_spec),
                named_prior_payloads(_typed_statistical_model_spec(statistical_model_spec), priors),
            ),
            ssm_spec=ssm_spec,
            structural_plan=StructuralPlan.model_validate(causal_design),
        )

        # Beta(2,2) → E=0.5; hourly dt = 1/24
        # dynamics-decay mean = -ln(0.5) / (1/24) = 0.693 * 24 ≈ 16.64
        dt_hourly = 1.0 / 24.0
        expected_mu = -math.log(0.5) / dt_hourly
        mu_val = _prior_reference_for_parameter(ssm_priors, index_maps, "rho_heart_rate")
        assert abs(mu_val - expected_mu) < 0.1

    def test_beta_prior_dt_to_ct_transform(self):
        """FIXED_EFFECT beta priors are converted via element-wise beta/dt scaling."""

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
                },
                {
                    "kind": "node_potential",
                    "target_id": "construct:6b04dc42c531e7091eb8",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:aa675637a0e802f0cb93b867b6112c3e017a59da1b5c5c51af028a0f8671a86c",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
                {
                    "kind": "linear",
                    "edge_id": "edge:923689028b6b177617c2",
                    "weight": {
                        "kind": "estimated",
                        "parameter_id": "parameter:9ea4b19b64aca6b21b54f4ba461f15339de78aea7d4decc3b5c6a011a0103508",
                    },
                },
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:45f78731e3e0c6f3efe1",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
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
                },
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:aa675637a0e802f0cb93b867b6112c3e017a59da1b5c5c51af028a0f8671a86c",
                    "owners": [{"kind": "construct", "id": "construct:6b04dc42c531e7091eb8"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "prior_transform": "dt_effect_to_ct_rate",
                    "id": "parameter:9ea4b19b64aca6b21b54f4ba461f15339de78aea7d4decc3b5c6a011a0103508",
                    "owners": [
                        {"kind": "construct", "id": "construct:6b04dc42c531e7091eb8"},
                        {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
                        {"kind": "edge", "id": "edge:923689028b6b177617c2"},
                    ],
                    "quantity": "dynamics_weight",
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.15}},
        }
        # off-diagonal support enables [mood, stress].
        edge_support = np.array([[False, True], [False, False]])
        ssm_spec = _default_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            edge_support=edge_support,
        )
        ssm_priors, index_maps, _diagnostics = compile_ssm_priors(
            model_with_prior_payloads(
                _typed_statistical_model_spec(statistical_model_spec),
                named_prior_payloads(_typed_statistical_model_spec(statistical_model_spec), priors),
            ),
            ssm_spec=ssm_spec,
            edge_lag_days={(0, 1): 1.0},
        )

        # Resolved 1d lag metadata: beta_CT = beta_DT / dt = 0.3 / 1 = 0.3
        mu_val = _prior_reference_for_parameter(ssm_priors, index_maps, "beta_stress_mood")
        assert abs(mu_val - 0.3) < 0.01

    def test_dt_ct_warning_uses_full_matrix_logm(self):
        """Cross-lag diagnostics should use logm(Phi)/dt, not beta/dt."""

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
                },
                {
                    "kind": "node_potential",
                    "target_id": "construct:6b04dc42c531e7091eb8",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:aa675637a0e802f0cb93b867b6112c3e017a59da1b5c5c51af028a0f8671a86c",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
                {
                    "kind": "linear",
                    "edge_id": "edge:923689028b6b177617c2",
                    "weight": {
                        "kind": "estimated",
                        "parameter_id": "parameter:9ea4b19b64aca6b21b54f4ba461f15339de78aea7d4decc3b5c6a011a0103508",
                    },
                },
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:45f78731e3e0c6f3efe1",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:3696aef3ff6f446744e5",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
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
                },
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:aa675637a0e802f0cb93b867b6112c3e017a59da1b5c5c51af028a0f8671a86c",
                    "owners": [{"kind": "construct", "id": "construct:6b04dc42c531e7091eb8"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "prior_transform": "dt_effect_to_ct_rate",
                    "id": "parameter:9ea4b19b64aca6b21b54f4ba461f15339de78aea7d4decc3b5c6a011a0103508",
                    "owners": [
                        {"kind": "construct", "id": "construct:6b04dc42c531e7091eb8"},
                        {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
                        {"kind": "edge", "id": "edge:923689028b6b177617c2"},
                    ],
                    "quantity": "dynamics_weight",
                    "name": "beta_stress_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_stress": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_stress_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.15}},
        }
        ssm_spec = _default_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["mood", "stress"],
            edge_support=np.array([[False, True], [False, False]]),
        )

        _ssm_priors, _idx, diagnostics = compile_ssm_priors(
            model_with_prior_payloads(
                _typed_statistical_model_spec(statistical_model_spec),
                named_prior_payloads(_typed_statistical_model_spec(statistical_model_spec), priors),
            ),
            ssm_spec=ssm_spec,
            edge_lag_days={(0, 1): 1.0},
        )

        warning = next(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.code == "dt_ct_approximation_warning"
        )
        assert "matrix-log mismatch; exact CT coupling" in _require_text(warning.issue)
        assert "0.600 1/day" in _require_text(warning.issue)
        assert "beta/dt value 0.300 1/day" in _require_text(warning.issue)
        assert warning.pathology_certificate is not None
        assert warning.pathology_certificate.primary_score == pytest.approx(0.5, abs=0.001)

    def test_lagged_beta_diagnostics_explain_default_authored_interval(self):
        """Lagged-edge diagnostics should mention the default authored interval semantics."""

        statistical_model_spec = {
            "mechanisms": [],
            "likelihoods": [
                {
                    "indicator_id": "indicator:ea4ac346f9793711d3de",
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
                    "id": "parameter:9e6fbc798c5b59ce541a7cbb11486933125b473c0880869fd3fc04e02fbdc94d",
                    "owners": [
                        {"kind": "construct", "id": "construct:6b04dc42c531e7091eb8"},
                        {"kind": "construct", "id": "construct:cdc0b2958a9512b2abad"},
                    ],
                    "quantity": "dynamics_weight",
                    "name": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "beta_stress_sleep": {
                "distribution": "Normal",
                "params": {"mu": 0.1, "sigma": 0.05},
                "sources": [
                    {
                        "title": "Weekly study",
                        "snippet": "Observed at weekly intervals.",
                        "study_interval_days": 7.0,
                    }
                ],
            },
        }
        ssm_spec = _default_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["stress", "sleep"],
            edge_support=np.array([[False, False], [True, False]]),
        )

        _priors, _idx, diagnostics = compile_ssm_priors(
            model_with_prior_payloads(
                _typed_statistical_model_spec(statistical_model_spec),
                named_prior_payloads(_typed_statistical_model_spec(statistical_model_spec), priors),
            ),
            ssm_spec=ssm_spec,
            edge_lag_days={(1, 0): 1.0},
        )

        warning = next(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.code == "interval_reference_missing"
        )
        assert "`reference_interval_days` is omitted" in _require_text(warning.issue)
        assert "default model interval (1.0d)" in _require_text(warning.issue)
        assert "`reference_interval_days`" in _require_text(warning.suggested_adjustment)

    def test_lagged_beta_diagnostics_preserve_reference_interval_language(self):
        """Lagged-edge diagnostics should talk about the authored reference interval."""

        statistical_model_spec = {
            "mechanisms": [],
            "likelihoods": [
                {
                    "indicator_id": "indicator:ea4ac346f9793711d3de",
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
                    "id": "parameter:9e6fbc798c5b59ce541a7cbb11486933125b473c0880869fd3fc04e02fbdc94d",
                    "owners": [
                        {"kind": "construct", "id": "construct:6b04dc42c531e7091eb8"},
                        {"kind": "construct", "id": "construct:cdc0b2958a9512b2abad"},
                    ],
                    "quantity": "dynamics_weight",
                    "name": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "beta_stress_sleep": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
                "reference_interval_days": 7.0,
                "sources": [
                    {
                        "title": "Daily study",
                        "snippet": "Observed at daily intervals.",
                        "study_interval_days": 1.0,
                    }
                ],
            },
        }
        ssm_spec = _default_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["stress", "sleep"],
            edge_support=np.array([[False, False], [True, False]]),
        )

        _priors, _idx, diagnostics = compile_ssm_priors(
            model_with_prior_payloads(
                _typed_statistical_model_spec(statistical_model_spec),
                named_prior_payloads(_typed_statistical_model_spec(statistical_model_spec), priors),
            ),
            ssm_spec=ssm_spec,
            edge_lag_days={(1, 0): 1.0},
        )

        warning = next(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.code == "interval_reference_mismatch"
        )
        assert "`reference_interval_days`" in _require_text(warning.issue)
        assert "7.0d" in _require_text(warning.issue)

    def test_beta_prior_dt_to_ct_respects_granularity(self):
        """FIXED_EFFECT beta transform uses effect construct's granularity."""

        statistical_model_spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:0000",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:8c3c4cb5c945ed0af65483504f2eaf8fe782152b4cfd2aee8d2a38e9bd17533b",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
                {
                    "kind": "node_potential",
                    "target_id": "construct:0001",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:27959fb1e678a58bc009181a7e723e9f9847fd61cda2ee8a26d13b2d584906ce",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
                {
                    "kind": "linear",
                    "edge_id": "edge:0000",
                    "weight": {
                        "kind": "estimated",
                        "parameter_id": "parameter:027ed95579d2da1b3011ba038d1870f7de748c596e65a7b898b3e6c978043c99",
                    },
                },
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:752d1cddd25bc5570a84",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:c33d5f1180ede8de27a3",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:8c3c4cb5c945ed0af65483504f2eaf8fe782152b4cfd2aee8d2a38e9bd17533b",
                    "owners": [{"kind": "construct", "id": "construct:0000"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:27959fb1e678a58bc009181a7e723e9f9847fd61cda2ee8a26d13b2d584906ce",
                    "owners": [{"kind": "construct", "id": "construct:0001"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_activity",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "prior_transform": "dt_effect_to_ct_rate",
                    "id": "parameter:027ed95579d2da1b3011ba038d1870f7de748c596e65a7b898b3e6c978043c99",
                    "owners": [
                        {"kind": "construct", "id": "construct:0001"},
                        {"kind": "construct", "id": "construct:0000"},
                        {"kind": "edge", "id": "edge:0000"},
                    ],
                    "quantity": "dynamics_weight",
                    "name": "beta_activity_heart_rate",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_activity": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_activity_heart_rate": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
            },
        }
        causal_design = make_structural_plan(
            ["heart_rate", "activity"],
            [("activity", "heart_rate")],
        )
        causal_design["semantics"]["model_clock"] = "1h"
        edge_support = np.array([[False, True], [False, False]])
        ssm_spec = _default_ssm_spec(
            n_latent=2,
            n_manifest=2,
            latent_names=["heart_rate", "activity"],
            latent_ids=["construct:0000", "construct:0001"],
            edge_support=edge_support,
        )
        ssm_priors, index_maps, _diagnostics = compile_ssm_priors(
            model_with_prior_payloads(
                _typed_statistical_model_spec(statistical_model_spec),
                named_prior_payloads(_typed_statistical_model_spec(statistical_model_spec), priors),
            ),
            ssm_spec=ssm_spec,
            structural_plan=StructuralPlan.model_validate(causal_design),
        )

        # Hourly dt = 1/24 → beta_CT = 0.3 / (1/24) = 7.2
        dt_hourly = 1.0 / 24.0
        expected_mu = 0.3 / dt_hourly  # 7.2
        mu_val = _prior_reference_for_parameter(
            ssm_priors,
            index_maps,
            "beta_activity_heart_rate",
        )
        assert abs(mu_val - expected_mu) < 0.5

    def test_compile_ssm_inputs_attaches_direct_writer_to_dt_ct_warning(self):
        statistical_model_spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:4fe59e1f780aa9e8b426",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:8c3c4cb5c945ed0af65483504f2eaf8fe782152b4cfd2aee8d2a38e9bd17533b",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
                {
                    "kind": "node_potential",
                    "target_id": "construct:42739985076ec8fdcd0a",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:27959fb1e678a58bc009181a7e723e9f9847fd61cda2ee8a26d13b2d584906ce",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
                {
                    "kind": "linear",
                    "edge_id": "edge:5c7f0a368b667aff4cfe",
                    "weight": {
                        "kind": "estimated",
                        "parameter_id": "parameter:027ed95579d2da1b3011ba038d1870f7de748c596e65a7b898b3e6c978043c99",
                    },
                },
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:752d1cddd25bc5570a84",
                    "distribution": "gaussian",
                    "link": "identity",
                    "standardized": True,
                    "reasoning": "",
                },
                {
                    "indicator_id": "indicator:c33d5f1180ede8de27a3",
                    "distribution": "gaussian",
                    "link": "identity",
                    "standardized": True,
                    "reasoning": "",
                },
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:8c3c4cb5c945ed0af65483504f2eaf8fe782152b4cfd2aee8d2a38e9bd17533b",
                    "owners": [{"kind": "construct", "id": "construct:4fe59e1f780aa9e8b426"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_heart_rate",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:27959fb1e678a58bc009181a7e723e9f9847fd61cda2ee8a26d13b2d584906ce",
                    "owners": [{"kind": "construct", "id": "construct:42739985076ec8fdcd0a"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_activity",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "prior_transform": "dt_effect_to_ct_rate",
                    "id": "parameter:027ed95579d2da1b3011ba038d1870f7de748c596e65a7b898b3e6c978043c99",
                    "owners": [
                        {"kind": "construct", "id": "construct:42739985076ec8fdcd0a"},
                        {"kind": "construct", "id": "construct:4fe59e1f780aa9e8b426"},
                        {"kind": "edge", "id": "edge:5c7f0a368b667aff4cfe"},
                    ],
                    "quantity": "dynamics_weight",
                    "name": "beta_activity_heart_rate",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "id": "parameter:c0f537d7729456b7acaac380b6f11394ba95ff13961642214337b1477f61654c",
                    "owners": [{"kind": "construct", "id": "construct:4fe59e1f780aa9e8b426"}],
                    "quantity": "diffusion_diag",
                    "name": "sigma_heart_rate",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "",
                },
                {
                    "id": "parameter:c98186a022e71240eba0b05cb7b2148c99e8d603bd482cf9bbc42c32ef7598c8",
                    "owners": [{"kind": "construct", "id": "construct:42739985076ec8fdcd0a"}],
                    "quantity": "diffusion_diag",
                    "name": "sigma_activity",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "",
                },
            ],
            "initialization_policy": "stationary",
            "observation_intercept_policy": "free",
        }
        priors = {
            "rho_heart_rate": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "rho_activity": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_activity_heart_rate": {
                "distribution": "Normal",
                "params": {"mu": 0.3, "sigma": 0.15},
            },
            "sigma_heart_rate": {"distribution": "HalfNormal", "params": {"sigma": 1.0}},
            "sigma_activity": {"distribution": "HalfNormal", "params": {"sigma": 1.0}},
        }
        causal_design = _compile_structural_plan(
            {
                "latent": {
                    "constructs": [
                        {
                            "id": "construct:4fe59e1f780aa9e8b426",
                            "name": "heart_rate",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                        {
                            "id": "construct:42739985076ec8fdcd0a",
                            "name": "activity",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [
                        {
                            "id": "edge:5c7f0a368b667aff4cfe",
                            "cause_id": "construct:42739985076ec8fdcd0a",
                            "effect_id": "construct:4fe59e1f780aa9e8b426",
                        }
                    ],
                },
                "estimation": {
                    "state_order": ["heart_rate", "activity"],
                    "edges": [{"cause": "activity", "effect": "heart_rate"}],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1h",
                    "indicators": [
                        {
                            "id": "indicator:752d1cddd25bc5570a84",
                            "construct_id": "construct:4fe59e1f780aa9e8b426",
                            "name": "hr",
                            "measurement_dtype": "continuous",
                        },
                        {
                            "id": "indicator:c33d5f1180ede8de27a3",
                            "construct_id": "construct:42739985076ec8fdcd0a",
                            "name": "act",
                            "measurement_dtype": "continuous",
                        },
                    ],
                },
            }
        )

        typed_statistical_model_spec = _typed_statistical_model_spec(statistical_model_spec)
        _ssm_spec, _ssm_priors, _bindings, diagnostics, _edge_lag_days, _parameters, _aux = (
            compile_ssm_inputs_from_statistical_model_spec(
                make_prior_model(typed_statistical_model_spec, priors),
                structural_plan=causal_design,
            )
        )

        dt_ct_warning = next(
            diagnostic
            for diagnostic in diagnostics
            if diagnostic.code == "dt_ct_approximation_warning"
        )
        assert dt_ct_warning.parameter == "linear_edge_weight"
        assert dt_ct_warning.related_parameters == [
            next(
                p.id
                for p in typed_statistical_model_spec.parameters
                if p.name == "beta_activity_heart_rate"
            )
        ]


# --- Trial Compile Tests ---


class TestTrialCompile:
    """Test trial_compile_statistical_model_spec catches structural errors early."""

    def test_valid_spec_returns_none(self, simple_statistical_model_spec):
        """A well-formed spec compiles successfully with default priors."""
        from tests.helpers import (
            trial_compile_statistical_model_spec,
        )

        result = trial_compile_statistical_model_spec(
            _typed_statistical_model_spec(simple_statistical_model_spec),
            _mood_structural_plan(),
        )
        assert result is None

    def test_compile_failure_returns_error(self):
        """When compilation raises, trial_compile returns the error string."""
        from tests.helpers import (
            trial_compile_statistical_model_spec,
        )

        spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:0000",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:af1c225bc2068c08ec33c4ee2ecc82c364224dfa00564d84148007a4999ff859",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                }
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:0000",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:af1c225bc2068c08ec33c4ee2ecc82c364224dfa00564d84148007a4999ff859",
                    "owners": [{"kind": "construct", "id": "construct:0000"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                }
            ],
        }
        with patch(
            "nof1_causal_lab.models.ssm.compile.artifact._compile_validated_ssm_artifact",
            side_effect=ValueError("dimension mismatch in dynamics matrix"),
        ):
            plan = make_structural_plan(["x"], [])
            plan["semantics"]["indicators"]["indicator:0000"]["name"] = "x"
            result = trial_compile_statistical_model_spec(
                _typed_statistical_model_spec(spec),
                StructuralPlan.model_validate(plan),
            )
        assert result is not None
        assert "dimension mismatch" in result

    def test_role_constraint_mismatch_returns_error(self):
        """Raw spec validation should reject parameter-role constraint mismatches."""
        from nof1_causal_lab.artifacts.statistical_model_spec import (
            validate_statistical_model_spec_dict,
        )

        spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:0000",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:af1c225bc2068c08ec33c4ee2ecc82c364224dfa00564d84148007a4999ff859",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                }
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:0000",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:af1c225bc2068c08ec33c4ee2ecc82c364224dfa00564d84148007a4999ff859",
                    "owners": [{"kind": "construct", "id": "construct:0000"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_x",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                },
                {
                    "id": "parameter:fdf941044f6f150fb049484a48adf7305f23e5ac44ef0cd0381cf618c212138d",
                    "owners": [{"kind": "construct", "id": "construct:0000"}],
                    "quantity": "diffusion_diag",
                    "name": "sigma_x",
                    "role": "residual_sd",
                    "constraint": "none",
                    "description": "test",
                },
            ],
        }

        validated, errors = validate_statistical_model_spec_dict(spec)

        assert validated is None
        assert any(
            "constraint 'none' unexpected for role 'residual_sd'" in error for error in errors
        )

    def test_missing_ar_parameters_returns_error(self):
        """Compiler should reject StatisticalModelSpecs with no latent dimensionality signal."""
        from tests.helpers import (
            trial_compile_statistical_model_spec,
        )

        spec = {
            "mechanisms": [],
            "likelihoods": [
                {
                    "indicator_id": "indicator:0000",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "id": "parameter:fdf941044f6f150fb049484a48adf7305f23e5ac44ef0cd0381cf618c212138d",
                    "owners": [{"kind": "construct", "id": "construct:0000"}],
                    "quantity": "diffusion_diag",
                    "name": "sigma_x",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "test",
                }
            ],
        }

        plan = make_structural_plan(["x"], [])
        plan["semantics"]["indicators"]["indicator:0000"]["name"] = "x"
        result = trial_compile_statistical_model_spec(
            _typed_statistical_model_spec(spec),
            StructuralPlan.model_validate(plan),
        )

        assert result is not None
        assert "Mechanisms must cover the retained dynamic states and edges" in result

    def test_unmanifested_retained_state_returns_error(self):
        """Structural planning rejects a retained state with no manifest."""
        spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:a1cddfa8657e4a8cb3ae",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:e85559567232d781cd1072f1b296e44db358805cbbdc816c3e93c1ad52f26915",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                }
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:54a6e8d16cbce15b4427",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                }
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:e85559567232d781cd1072f1b296e44db358805cbbdc816c3e93c1ad52f26915",
                    "owners": [{"kind": "construct", "id": "construct:a1cddfa8657e4a8cb3ae"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_outcome",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                }
            ],
        }
        causal_design = make_structural_plan(["treatment", "outcome"], [])
        causal_design["semantics"]["indicators"]["indicator:0001"]["name"] = "outcome_score"
        causal_design["manifest_indicator_order"] = ["indicator:0001"]
        causal_design["dispositions"][2]["disposition"] = "excluded_indicator"

        _typed_statistical_model_spec(spec)
        with pytest.raises(ValueError, match="retained states lack manifest indicators"):
            StructuralPlan.model_validate(causal_design)

    def test_trial_compile_aggregates_initial_state_translation_errors(self):
        """Translation should report multiple initial-state correlation errors together."""
        from tests.helpers import (
            trial_compile_statistical_model_spec,
        )

        causal_design = _compile_structural_plan(
            {
                "latent": {
                    "default_outcome": {
                        "kind": "construct",
                        "id": "construct:d90c52e59b79004188dc",
                    },
                    "constructs": [
                        {
                            "id": "construct:311c9047b5ede16a8f26",
                            "name": "X",
                            "role": "exogenous",
                            "description": "X",
                            "temporal_status": "time_varying",
                        },
                        {
                            "id": "construct:d90c52e59b79004188dc",
                            "name": "Y",
                            "role": "endogenous",
                            "description": "Y",
                            "temporal_status": "time_varying",
                        },
                    ],
                    "edges": [
                        {
                            "id": "edge:39ba80b774e02c409662",
                            "cause_id": "construct:311c9047b5ede16a8f26",
                            "effect_id": "construct:d90c52e59b79004188dc",
                        }
                    ],
                },
                "estimation": {
                    "state_order": ["X", "Y"],
                    "edges": [{"cause": "X", "effect": "Y"}],
                    "induced_dependencies": [],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "id": "indicator:0954266d4cc513fa17ad",
                            "construct_id": "construct:311c9047b5ede16a8f26",
                            "name": "x_score",
                            "how_to_measure": "Use x_score directly",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                        {
                            "id": "indicator:60fdff0597f5fa0f7194",
                            "construct_id": "construct:d90c52e59b79004188dc",
                            "name": "y_score",
                            "how_to_measure": "Use y_score directly",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                    ],
                },
            }
        )
        spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:311c9047b5ede16a8f26",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:d4f5f53b1e587883d9cd30cc9393726d9e06dc90b63b7cae47ce252f25f4b42a",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
                {
                    "kind": "node_potential",
                    "target_id": "construct:d90c52e59b79004188dc",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:f6c297044713e7312c77a790e9a018f4881e7382620dec264eec8546bdd9c56a",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:0954266d4cc513fa17ad",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                },
                {
                    "indicator_id": "indicator:60fdff0597f5fa0f7194",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "test",
                },
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:d4f5f53b1e587883d9cd30cc9393726d9e06dc90b63b7cae47ce252f25f4b42a",
                    "owners": [{"kind": "construct", "id": "construct:311c9047b5ede16a8f26"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_X",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                },
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:f6c297044713e7312c77a790e9a018f4881e7382620dec264eec8546bdd9c56a",
                    "owners": [{"kind": "construct", "id": "construct:d90c52e59b79004188dc"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_Y",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "test",
                },
                {
                    "prior_transform": "initial_state_correlation",
                    "id": "parameter:6efa548cc7ef957857274e64c9693e3bb393bed5fa3c20a78b572ec53aa65963",
                    "owners": [
                        {"kind": "construct", "id": "construct:311c9047b5ede16a8f26"},
                        {"kind": "construct", "id": "construct:311c9047b5ede16a8f26"},
                    ],
                    "quantity": "t0_var_lower",
                    "name": "cor0_X_X",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "invalid self correlation",
                },
                {
                    "prior_transform": "initial_state_correlation",
                    "id": "parameter:7eb095e23e64163bbb15e0d8f92b9163eb47c44d31da095c617d0e0efad15a78",
                    "owners": [
                        {"kind": "construct", "id": "construct:697a57ed87711fa1868c"},
                        {"kind": "construct", "id": "construct:4c7e768a743b41accdc7"},
                    ],
                    "quantity": "t0_var_lower",
                    "name": "cor0_unknown_pair",
                    "role": "initial_state_correlation",
                    "constraint": "correlation",
                    "description": "invalid parse",
                },
            ],
        }

        result = trial_compile_statistical_model_spec(
            _typed_statistical_model_spec(spec),
            causal_design,
        )

        assert result is not None
        assert "unknown scientific owner" in result
        assert "cor0_unknown_pair" in result
