"""Small analytic checks for native prior laws and their JSON boundary."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.distributions import constraints

from nof1_causal_lab.artifacts.distribution import CompiledDistribution
from nof1_causal_lab.prior_distributions import (
    batch_prior_distributions,
    deserialize_distribution,
    distribution_from_params,
    interval_effect_to_rate,
    persistence_to_decay,
    prior_reference_value,
    serialize_distribution,
)
from tests.helpers import model_with_prior_payloads, named_prior_payloads


def test_beta_persistence_has_exact_density_and_jacobian():
    prior = persistence_to_decay(dist.Beta(2.0, 3.0), 7.0)
    decay = 0.2
    rho = math.exp(-7.0 * decay)
    expected = math.log(12.0) + math.log(rho) + 2.0 * math.log1p(-rho) + math.log(7.0 * rho)
    assert float(prior.log_prob(decay)) == pytest.approx(expected, abs=2e-6)
    gradient = jax.grad(prior.log_prob)(decay)
    assert float(gradient) == pytest.approx(-14.0 + 14.0 * rho / (1.0 - rho), abs=2e-5)
    assert not bool(prior.support(-0.1))
    assert bool(prior.support(decay))


def test_persistence_draw_is_the_exact_pushforward():
    base = dist.Beta(2.0, 3.0)
    prior = persistence_to_decay(base, 3.0)
    key = jax.random.PRNGKey(7)
    expected = -jnp.log(base.sample(key, (3,))) / 3.0
    np.testing.assert_array_equal(prior.sample(key, (3,)), expected)


@pytest.mark.parametrize(
    "base", [dist.Normal(0.5, 0.1), dist.Uniform(-0.1, 1.0), dist.Uniform(0.0, 1.1)]
)
def test_persistence_requires_an_authored_law_on_the_unit_interval(base):
    with pytest.raises(ValueError, match="support within"):
        persistence_to_decay(base, 1.0)


def test_interval_effect_preserves_uniform_family_and_bounds():
    prior = interval_effect_to_rate(dist.Uniform(-2.0, 4.0), 2.0)
    assert float(prior.support.lower_bound) == -1.0
    assert float(prior.support.upper_bound) == 2.0
    assert float(prior.log_prob(0.5)) == pytest.approx(-math.log(3.0))


def test_transformed_vector_prior_roundtrip_preserves_density_and_positive_support():
    coordinates = [persistence_to_decay(dist.Beta(2.0, 3.0), interval) for interval in [1.0, 7.0]]
    prior = batch_prior_distributions(coordinates, (2,), support=constraints.positive)
    recipes = [
        CompiledDistribution.model_validate_json(recipe.model_dump_json())
        for recipe in serialize_distribution(prior)
    ]
    restored = batch_prior_distributions(
        [deserialize_distribution(recipe) for recipe in recipes], (2,), support=constraints.positive
    )
    values = jnp.array([0.4, 0.1])
    np.testing.assert_allclose(prior.log_prob(values), restored.log_prob(values))
    assert np.all(np.asarray(restored.support(values)))
    assert not np.any(np.asarray(restored.support(-values)))
    abstract = jax.eval_shape(lambda law, value: law.log_prob(value), restored, values)
    assert abstract.shape == (2,)


def test_different_coordinate_families_have_no_mixture_uncertainty():
    coordinates = [dist.Normal(0.0, 1.0), dist.Uniform(-2.0, 2.0)]
    prior = batch_prior_distributions(coordinates, (2,), support=constraints.real)
    values = jnp.array([0.2, 0.3])
    expected = jnp.array([coordinates[0].log_prob(values[0]), coordinates[1].log_prob(values[1])])
    np.testing.assert_allclose(prior.log_prob(values), expected)
    assert jnp.isneginf(prior.log_prob(jnp.array([0.2, 3.0]))[1])
    recipes = serialize_distribution(prior)
    assert [recipe.distribution.value for recipe in recipes] == ["Normal", "Uniform"]


def test_distribution_arguments_are_complete_and_native_validated():
    with pytest.raises(ValueError, match="requires exactly"):
        distribution_from_params("Gamma", {"concentration": 2.0})
    with pytest.raises(ValueError, match="invalid rate"):
        distribution_from_params("Gamma", {"concentration": 2.0, "rate": -1.0})
    with pytest.raises(ValueError, match="lower < upper"):
        distribution_from_params("Uniform", {"lower": 2.0, "upper": 1.0})


@pytest.mark.parametrize(
    ("coordinates", "values", "support"),
    [
        ([dist.Normal(0.0, 1.0), dist.LogNormal(0.0, 0.5)], [-2.0, 0.3], constraints.real),
        (
            [dist.Gamma(2.0, 3.0), persistence_to_decay(dist.Beta(2.0, 3.0), 7.0)],
            [0.4, 0.1],
            constraints.positive,
        ),
    ],
)
def test_mixed_coordinate_gradients_and_roundtrip_ignore_inactive_laws(
    coordinates, values, support
):
    prior = batch_prior_distributions(coordinates, (2,), support=support)
    values = jnp.array(values)
    expected = jnp.array(
        [jax.grad(law.log_prob)(values[index]) for index, law in enumerate(coordinates)]
    )
    np.testing.assert_allclose(
        jax.grad(lambda x: prior.log_prob(x).sum())(values), expected, atol=2e-6
    )
    recipes = serialize_distribution(prior)
    restored = batch_prior_distributions(
        [deserialize_distribution(recipe) for recipe in recipes], (2,), support=support
    )
    np.testing.assert_allclose(restored.log_prob(values), prior.log_prob(values), atol=2e-6)
    np.testing.assert_allclose(
        prior_reference_value(prior), [prior_reference_value(law) for law in coordinates]
    )


def test_expanded_transformed_reference_is_an_anchor_not_a_claimed_mean():
    law = persistence_to_decay(dist.Beta(2.0, 2.0), 1.0).expand((3,))
    np.testing.assert_allclose(prior_reference_value(law), jnp.full(3, math.log(2.0)))


def test_compiler_and_dynestyx_parameter_trace_use_the_exact_persistence_law():
    from dataclasses import replace

    from numpyro import handlers

    from nof1_causal_lab.artifacts.compiled_ssm import CompiledSSMArtifact, CompiledStructure
    from nof1_causal_lab.artifacts.identity import ConstructRef
    from nof1_causal_lab.artifacts.parameter import SiteKind
    from nof1_causal_lab.artifacts.statistical_model_spec import StatisticalModelSpec
    from nof1_causal_lab.models.ssm.compile.artifact import (
        serialize_ssm_spec,
    )
    from nof1_causal_lab.models.ssm.compile.parameter_identity import parameter_identity
    from nof1_causal_lab.models.ssm.compile.prior_compilation import bind_parameters, compile_priors
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec, NodePotentialSpec
    from nof1_causal_lab.models.ssm.model import SSMModel
    from nof1_causal_lab.models.ssm.parameterization import (
        compile_prior_semantics,
        load_prior_runtime_bundle,
    )
    from nof1_causal_lab.models.ssm.structure.parameters import Fixed
    from tests.ssm_spec_fixtures import block_ssm_spec

    spec = block_ssm_spec(
        n_latent=1,
        latent_names=["mood"],
        dynamics_spec=DynamicsSpec(
            1, (NodePotentialSpec(target=0, center=Fixed(0.0), quartic=Fixed(0.2)),)
        ),
    )
    spec = replace(
        spec,
        diffusion_block=replace(
            spec.diffusion_block, diffusion_chol_support=np.zeros((1, 1), dtype=bool)
        ),
        manifest_chol_block=replace(
            spec.manifest_chol_block, diag_support=np.zeros(1, dtype=bool), template=jnp.eye(1)
        ),
        t0_means_block=replace(spec.t0_means_block, free_support=np.zeros(1, dtype=bool)),
        t0_chol_block=replace(spec.t0_chol_block, diag_support=np.zeros(1, dtype=bool)),
    )
    authored = StatisticalModelSpec.model_validate(
        {
            "mechanisms": [],
            "likelihoods": [],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": parameter_identity(
                        SiteKind.DYNAMICS_DECAY, [ConstructRef(id="construct:bbc87212909e45b9e6c3")]
                    ),
                    "owners": [{"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "Persistence over seven days",
                }
            ],
        }
    )
    authored = model_with_prior_payloads(
        authored,
        named_prior_payloads(
            authored,
            {
                "rho_mood": {
                    "distribution": "Beta",
                    "params": {"alpha": 2.0, "beta": 3.0},
                    "reference_interval_days": 7.0,
                }
            },
        ),
    )
    priors, semantic_bindings, _ = compile_priors(authored, spec)
    semantics = compile_prior_semantics(spec, priors)
    semantics = type(semantics).model_validate_json(semantics.model_dump_json())
    definitions, bindings, auxiliary = bind_parameters(
        semantic_bindings, spec, None, authored.parameters
    )
    assert spec.manifest_ids is not None
    assert spec.manifest_names is not None
    artifact = CompiledSSMArtifact(
        schema_version=2,
        structure=CompiledStructure(
            spec=serialize_ssm_spec(spec), edge_lag_days=[], bindings=[], anchor_certificates=[]
        ),
        compiled_prior_semantics=semantics,
        observation_bindings=dict(zip(spec.manifest_ids, spec.manifest_names, strict=True)),
        parameters=definitions,
        parameter_bindings=bindings,
        auxiliary_coordinates=auxiliary,
        compile_diagnostics=[],
    )
    artifact = CompiledSSMArtifact.model_validate_json(artifact.model_dump_json())
    recovered = next(
        parameter for parameter in artifact.parameters if parameter.id == authored.parameters[0].id
    )
    assert isinstance(recovered.prior, dist.Beta)
    np.testing.assert_allclose(recovered.prior.concentration1, 2.0)
    np.testing.assert_allclose(recovered.prior.concentration0, 3.0)
    assert recovered.reference_interval_days == 7.0
    restored = load_prior_runtime_bundle(semantics)
    model = SSMModel(spec, priors=restored.priors)
    value = jnp.array(0.2)
    with handlers.substitute(data={"vf_0_decay": value}):
        trace = handlers.trace(model._sample_runtime_dynamics).get_trace(
            jnp.eye(1), jnp.zeros((1, 0))
        )
    law = trace["vf_0_decay"]["fn"]
    expected = dist.Beta(2.0, 3.0).log_prob(jnp.exp(-7.0 * value)) + jnp.log(7.0) - 7.0 * value
    np.testing.assert_allclose(law.log_prob(value), expected, atol=2e-6)
    assert np.isfinite(jax.grad(law.log_prob)(value))
    assert semantics.priors["vf_0_decay"][0].transforms[0].kind == "persistence_to_decay"


@pytest.mark.parametrize(
    ("family", "params", "value"),
    [
        ("Normal", {"mu": 0.2, "sigma": 0.4}, 0.3),
        ("HalfNormal", {"sigma": 0.4}, 0.3),
        ("Gamma", {"concentration": 2.0, "rate": 4.0}, 0.3),
        ("LogNormal", {"mu": 0.2, "sigma": 0.4}, 0.3),
        ("Exponential", {"rate": 4.0}, 0.3),
        ("Beta", {"alpha": 2.0, "beta": 4.0}, 0.3),
        ("Uniform", {"lower": -1.0, "upper": 2.0}, 0.3),
        ("TruncatedNormal", {"mu": 0.2, "sigma": 0.4, "lower": -1.0, "upper": 2.0}, 0.3),
        ("Delta", {"value": 0.3}, 0.3),
    ],
)
def test_approved_family_json_roundtrip_keeps_its_native_density(family, params, value):
    law = distribution_from_params(family, params)
    recipe = serialize_distribution(law)[0]
    restored = deserialize_distribution(
        CompiledDistribution.model_validate_json(recipe.model_dump_json())
    )
    np.testing.assert_allclose(law.log_prob(value), restored.log_prob(value), atol=1e-6)
