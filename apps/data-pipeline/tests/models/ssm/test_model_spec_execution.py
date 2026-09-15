"""ModelSpec execution and stored identity checks without inference or simulation."""

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.models.model_structure import model_for_constructs
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.dynamical_model import (
    HeterogeneousObservation,
    build_dynamical_model,
)
from nof1_causal_lab.models.ssm.inference.persistence import condition_model, model_draws
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws, ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
)
from nof1_causal_lab.recipes.construct_authoring import AdmissionState
from tests.dynamics_fixtures import decay_term, interaction_term, linear_term
from tests.helpers import complete_test_model, make_model


@pytest.fixture(scope="module")
def model():
    value = make_model(["A", "B"], [("A", "B")])
    return complete_test_model(
        value, self_limiting=[value.constructs[0].id], hill_edges=[value.edges[0].id]
    )


def test_conditioning_revises_the_same_type_and_retains_joint_uncertainty(
    model, tmp_path, monkeypatch
):
    from functools import cache

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.machine.derivations import read_model
    from nof1_causal_lab.machine.store import ArtifactStore
    from nof1_causal_lab.models.model_inputs import input_fingerprints
    from nof1_causal_lab.numpyro_json import empirical_atoms
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store = ArtifactStore("TEST")
    count = 3
    samples = {
        site.name: jnp.arange(count, dtype=float).reshape((count,) + (1,) * len(site.shape))
        + jnp.full((count, *site.shape), 0.5)
        for site in build_site_registry(model)
    }
    samples.update(assemble_deterministics_from_registry(samples, model))
    paths = jnp.arange(count * 4 * 2, dtype=float).reshape(count, 4, 2)
    result = ParticleMCMCPosterior(
        JointPosteriorDraws(samples, paths), diagnostics={"likelihood_backend": lambda: None}
    )
    conditioned = condition_model(
        model,
        result,
        times=jnp.arange(4),
        array_writer=store.write_array,
        array_loader=cache(store.read_array),
    )
    assert type(conditioned) is ModelSpec
    assert model.time_points == ()
    assert len(model.distributions) == len(model.parameters)
    assert len(conditioned.distributions) == 1
    assert all(
        p.distribution == next(iter(conditioned.distributions))
        for p in conditioned.parameters
        if p.value is None
    )
    payload = conditioned.model_dump(mode="json")
    assert not {"posterior", "prior", "provenance", "diagnostics", "result"} & payload.keys()
    assert "array_ref" in conditioned.model_dump_json()
    info = store.write_version(
        "model",
        provenance="computed",
        derived_from={},
        produced_by="run:posterior",
        json_files={"model.json": payload},
    )
    loaded = read_model(store, info.version)
    assert loaded == conditioned
    assert input_fingerprints(model)["compilation"] == input_fingerprints(loaded)["compilation"]
    restored = model_draws(loaded)
    for name, values in samples.items():
        np.testing.assert_array_equal(restored.parameters[name], values)
    np.testing.assert_array_equal(restored.latent_paths, paths)
    # Entity/parameter list order carries no joint distribution coordinates.
    reordered = loaded.revised(parameters=tuple(reversed(loaded.parameters)))
    for name, values in restored.parameters.items():
        np.testing.assert_array_equal(model_draws(reordered).parameters[name], values)
    assert (
        loaded.revised(edges=replace_constructs(loaded.edges, tuple(reversed(loaded.constructs))))
        == loaded
    )
    np.testing.assert_array_equal(
        empirical_atoms(next(iter(loaded.distributions.values()))),
        empirical_atoms(next(iter(conditioned.distributions.values()))),
    )


def test_numerical_function_constructs_dynestyx_model(model):
    samples = {site.name: jnp.full((1, *site.shape), 0.5) for site in build_site_registry(model)}
    samples.update(assemble_deterministics_from_registry(samples, model))
    native = build_dynamical_model(
        model, {key: value[0] for key, value in samples.items()}, t0=jnp.asarray(0.0)
    )
    assert isinstance(native, dsx.DynamicalModel)
    assert native.initial_condition.event_shape == (2,)
    assert isinstance(native.observation_model, HeterogeneousObservation)
    assert isinstance(native.state_evolution, dsx.StochasticContinuousTimeStateEvolution)
    assert native.state_evolution.drift is not None
    assert native.observation_model.families == tuple(numeric.observation_families(model))
    assert native.state_evolution.drift(jnp.ones(2), jnp.empty(0), 0.0).shape == (2,)


def test_predictive_runtime_uses_native_initial_and_observation_laws(model):
    """Exercise model batching and prediction at one time point, without a trajectory solve."""
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
        sample_predictive_emissions,
        simulate_predictive_latents,
    )

    samples = {site.name: jnp.full((2, *site.shape), 0.5) for site in build_site_registry(model)}
    samples.update(assemble_deterministics_from_registry(samples, model))
    times = jnp.array([2.0])
    key = jax.random.PRNGKey(14)
    latents, predictors = simulate_predictive_latents(model, samples, times, rng_key=key)
    native = build_dynamical_model(
        model, {name: values[0] for name, values in samples.items()}, t0=times[0]
    )
    assert isinstance(native.observation_model, HeterogeneousObservation)
    init_key, _ = jax.random.split(jax.random.split(key, 2)[0])
    np.testing.assert_allclose(latents[0, 0], native.initial_condition.sample(init_key), atol=1e-6)
    np.testing.assert_allclose(
        predictors[0, 0], native.observation_model.linear_predictor(latents[0, 0]), atol=1e-6
    )
    observations, mask, means = sample_predictive_emissions(
        model,
        samples,
        predictors,
        times,
        observation_support=None,
        observation_mask=jnp.array([[True, False]]),
        num_samples=2,
        rng_key=jax.random.PRNGKey(15),
    )
    law = native.observation_model(latents[0, 0], None, times[0])
    assert observations.shape == means.shape == mask.shape == (2, 1, 2)
    np.testing.assert_allclose(means[0, 0, 0], law.mean[0], atol=1e-6)
    assert np.all(np.isfinite(observations[:, :, 0]))
    assert np.all(np.isnan(observations[:, :, 1]))
    assert np.all(mask[:, :, 0])
    assert not np.any(mask[:, :, 1])


def test_predictive_edge_off_reaches_the_native_state_evolution(model, monkeypatch):
    """Inspect the declared derivative at the solver boundary, without integrating a path."""
    from dataclasses import replace

    import equinox as eqx

    from nof1_causal_lab.artifacts.expressions import LiteralExpression
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from nof1_causal_lab.models.ssm.predictive import registry_runtime

    samples = {site.name: jnp.full((1, *site.shape), 0.5) for site in build_site_registry(model)}
    samples.update(assemble_deterministics_from_registry(samples, model))
    terms = numeric.dynamics_expressions(model)
    edge = next(term for term in terms if term.edge_owned)
    dynamics = DynamicsSpec(
        2,
        tuple(
            replace(term, expression=LiteralExpression(value=0)) if term.edge_owned else term
            for term in terms
        ),
    )
    state = jnp.ones(2)

    def inspect_drift(models, times, *_args):
        return eqx.filter_vmap(
            lambda native: native.state_evolution.total_drift(state, None, times[0])[None, :]
        )(models)

    monkeypatch.setattr(registry_runtime, "_simulate_model_predictive_draws", inspect_drift)
    derivatives, _ = registry_runtime._simulate_vector_field_predictive_latents(
        model,
        samples,
        jnp.array([3.0]),
        rng_key=jax.random.PRNGKey(21),
        dynamics=dynamics,
    )
    natural = build_dynamical_model(
        model, {name: values[0] for name, values in samples.items()}, t0=jnp.array(3.0)
    )
    assert isinstance(natural.state_evolution, dsx.StochasticContinuousTimeStateEvolution)
    # This model's Hill edge vanishes at zero source, while the target's own
    # nonlinear restoring dynamics stay the same.
    source_zero = state.at[edge.source].set(0.0)
    expected = natural.state_evolution.total_drift(source_zero, None, 3.0)[edge.target]
    np.testing.assert_allclose(derivatives[0, 0, edge.target], expected, atol=1e-6)
    assert (
        derivatives[0, 0, edge.target]
        != natural.state_evolution.total_drift(state, None, 3.0)[edge.target]
    )


def test_admission_scope_is_derived_from_the_scientific_value(model):
    model_for_constructs(model, {"A"})
    partial = AdmissionState(model=model, names=("A",)).completed_model(restrict=True)
    assert numeric.state_names(partial) == ["A"]
    assert numeric.observation_names(partial) == ["A_obs"]
    assert numeric.loading_block(partial).template.shape == (1, 1)
    partial.check_execution()
    assert numeric.state_names(model) == ["A", "B"]


def test_model_equality_does_not_depend_on_execution_cache(model):
    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    restored = ModelSpec.model_validate_json(model.model_dump_json())
    (model).require_execution_structure()
    (restored).require_execution_structure()
    assert restored == model


def test_nonlinear_fixture_declares_the_same_drift_and_measurements():
    """Compare a single true drift evaluation; no simulator or inference is run."""
    from evaluation.fixtures import synthetic_nonlinear as fixture

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.execution.parameters import assemble_model_matrices

    source = fixture.build_synthetic_nonlinear_spec()
    source = ModelSpec.model_validate_json(source.model_dump_json())
    samples = {name: jnp.asarray(value) for name, value in fixture.SCALAR_RECOVERY_TARGETS.items()}
    samples.update(
        diffusion_diag_free=jnp.asarray(fixture.TRUE_DIFFUSION_SD),
        lambda_free=jnp.asarray(
            [
                fixture.TRUE_LOADINGS[row, col]
                for row, col in fixture.MEASUREMENT_LOADINGS_FREE_POSITIONS
            ]
        ),
        manifest_var_diag_free=jnp.asarray(fixture.TRUE_MANIFEST_SD)[
            numeric.observation_noise_block(source).diag_support[:-2]
        ],
        manifest_means_free=jnp.asarray(fixture.TRUE_MANIFEST_MEANS)[
            numeric.observation_mean_block(source).free_support[:-2]
        ],
    )
    matrices, _ = assemble_model_matrices(source, samples)
    np.testing.assert_allclose(matrices["lambda"][:-2, :-2], fixture.TRUE_LOADINGS)
    np.testing.assert_allclose(matrices["manifest_means"][:-2], fixture.TRUE_MANIFEST_MEANS)
    native = build_dynamical_model(source, {**samples, **matrices}, t0=jnp.asarray(0.0))
    assert isinstance(native.state_evolution, dsx.StochasticContinuousTimeStateEvolution)
    assert native.state_evolution.drift is not None
    state = jnp.asarray([0.8, 0.5, 1.2])
    controls = jnp.asarray([0.2, -0.4])
    np.testing.assert_allclose(
        native.state_evolution.drift(jnp.concatenate([state, controls]), None, 0.0)[:3],
        fixture._synthetic_nonlinear_drift(np.asarray(state), np.asarray(controls)),
        atol=1e-7,
    )


def test_fixed_quantities_and_interactions_remain_effective_in_edge_off_checks(monkeypatch):
    from nof1_causal_lab.artifacts.expressions import LiteralExpression, linear_coefficient
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.parameter import SiteKind
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from nof1_causal_lab.models.ssm.predictive import registry_runtime
    from nof1_causal_lab.models.ssm.simulation_checks import (
        _incoming_edge_off_target,
        _resimulate_edge_off,
    )
    from nof1_causal_lab.recipes.construct_authoring import ConstructContribution
    from tests.model_fixtures import model_fixture, parameter_draws

    source = model_fixture(
        n_latent=3,
        latent_names=["A", "B", "C"],
        dynamics_spec=DynamicsSpec(
            3,
            (
                *(decay_term(target=i) for i in range(3)),
                linear_term(0, 2),
                interaction_term(0, 1, 2, 0.8),
            ),
        ),
    )
    source = source.revised(
        distributions={
            k: v
            for k, v in source.distributions.items()
            if k
            not in {
                p.distribution
                for p in source.parameters
                if source.parameter_context(p.id).quantity == SiteKind.DYNAMICS_WEIGHT
            }
        },
        parameters=tuple(
            p.model_copy(update={"value": 0.7, "distribution": None})
            if source.parameter_context(p.id).quantity == SiteKind.DYNAMICS_WEIGHT
            else p
            for p in source.parameters
        ),
    )
    source = ModelSpec.model_validate_json(source.model_dump_json())
    terms = numeric.dynamics_expressions(source)
    assert linear_coefficient(terms[3].expression, source.state_order[0]) == 0.7
    assert not terms[3].parameters
    target = _incoming_edge_off_target(
        source,
        ConstructContribution(construct=source.constructs[2], edge_parents=("A", "B")),
        numeric.state_names(source),
        2,
    )
    assert target.components == (3, 4)
    calls = []
    original_samples = {**parameter_draws(source, 2), "latents": jnp.ones((2, 3, 3))}

    def capture(model, samples, times, *, dynamics, **_kwargs):
        assert model is source
        for index in (3, 4):
            assert dynamics.components[index].expression == LiteralExpression(value=0)
        for name, value in original_samples.items():
            np.testing.assert_array_equal(samples[name], value)
        calls.append(dynamics)
        return jnp.zeros((2, len(times), 3)), jnp.zeros((2, len(times), 3))

    monkeypatch.setattr(registry_runtime, "_simulate_vector_field_predictive_latents", capture)
    _resimulate_edge_off(
        source,
        original_samples,
        jnp.arange(3),
        target,
        seed=0,
    )
    assert len(calls) == 1
    assert numeric.dynamics_expressions(source) == terms
