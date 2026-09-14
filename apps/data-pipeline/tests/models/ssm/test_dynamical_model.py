"""Small model-boundary checks; no fitting or trajectory simulation."""

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dynestyx.inference.particle_runtime import Parameterization

from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.inference import problem as problem_module
from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import _prepare_linearized_path
from tests.dynamics_fixtures import potential_term


@pytest.fixture
def runtime(monkeypatch):
    # Only parameter discovery is stubbed. The declared drift, observation model,
    # model partition, discretizer, and particle target are the actual code paths.
    times = jnp.array([2.0, 2.1, 2.35])
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from nof1_causal_lab.models.ssm.execution.parameters import assemble_model_matrices
    from nof1_causal_lab.models.ssm.structure import DiffusionBlockSpec
    from tests.model_fixtures import (
        default_input_effect_block,
        default_lambda_block,
        default_manifest_chol_block,
        default_t0_chol_block,
        default_t0_means_block,
        model_fixture,
    )

    spec = model_fixture(
        n_latent=1,
        n_manifest=2,
        dynamics_spec=DynamicsSpec(
            1,
            (
                potential_term(
                    target=0,
                    center=None,
                    stiffness=FixedCoefficient(value=0.4),
                    quartic=FixedCoefficient(value=0.2),
                ),
            ),
        ),
        input_names=["forcing"],
        manifest_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.POISSON],
        manifest_links=[LinkFunction.IDENTITY, LinkFunction.LOG],
        diffusion_block=DiffusionBlockSpec(
            1, np.zeros((1, 1), dtype=bool), jnp.sqrt(jnp.array([[0.3]]))
        ),
        lambda_block=replace(default_lambda_block(2, 1), template=jnp.ones((2, 1))),
        manifest_chol_block=replace(
            default_manifest_chol_block(2),
            diag_support=np.zeros(2, dtype=bool),
            template=jnp.eye(2),
        ),
        t0_means_block=replace(
            default_t0_means_block(1),
            free_support=np.zeros(1, dtype=bool),
            template=jnp.array([0.7]),
        ),
        t0_chol_block=replace(
            default_t0_chol_block(1),
            diag_support=np.zeros(1, dtype=bool),
            template=jnp.sqrt(jnp.array([[0.8]])),
        ),
        input_effect_block=replace(
            default_input_effect_block(1),
            n_cols=1,
            free_support=np.zeros((1, 1), dtype=bool),
            template=jnp.array([[0.5]]),
        ),
    )
    model = SimpleNamespace(
        spec=spec,
        observation_support=None,
        transition_inputs=jnp.array([[99.0], [2.0], [-1.0]]),
    )

    def constrain(z):
        samples = {"vf_0_p0": z[0]}
        matrices, _ = assemble_model_matrices(spec, samples)
        return {**samples, **matrices}

    parameters = Parameterization(
        initial_position=jnp.array([0.2]),
        unravel=lambda z: {"vf_0_p0": z[0]},
        constrain=constrain,
        log_prior=lambda z: -jnp.sum(z**2),
    )
    monkeypatch.setattr(
        problem_module, "prepare_model_parameters", lambda *_: (parameters, {}, {"vf_0_p0"})
    )
    return problem_module.build_particle_problem(
        model,
        jnp.array([[0.2, 1.0], [jnp.nan, 2.0], [jnp.nan, jnp.nan]]),
        times,
        scheme="euler_maruyama",
        trace_key=jax.random.key(0),
        reparam=None,
    ).runtime


def test_sampler_context_is_a_dynestyx_model_pytree(runtime):
    context = runtime.context(runtime.initial_position, runtime.times)
    assert isinstance(context[0], dsx.DynamicalModel)
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(context))
    shapes = jax.eval_shape(
        jax.vmap(runtime.context, in_axes=(0, None)), jnp.zeros((2, 1)), runtime.times
    )
    assert shapes[0].initial_condition.loc.shape == (2, 1)
    assert shapes[1].shape == (2, 3)


@pytest.mark.parametrize("supplied_path", [False, True])
def test_warmup_gaussian_view_traces_with_scalar_state_and_known_inputs(runtime, supplied_path):
    """Trace the real library discretizer without executing matrix exponentials."""
    context = runtime.context(runtime.initial_position, runtime.times)
    continuous = context[0].state_evolution
    schedule = runtime.schedule(context)
    init_mean = jnp.array([0.7])
    intervals = jnp.array([1e-4, 0.1, 0.25])

    def prepare(path):
        transitions_at, seed = _prepare_linearized_path(
            continuous,
            intervals,
            init_mean,
            transition_inputs=schedule.transition_controls,
            z_init=path if supplied_path else None,
            dtype=path.dtype,
        )
        return seed, transitions_at(seed)

    shapes = jax.eval_shape(prepare, jnp.zeros((3, 1)))
    seed, (transition, covariance, bias) = shapes
    assert seed.shape == (3, 1)
    assert transition.shape == (3, 1, 1)
    assert covariance.shape == (3, 1, 1)
    assert bias.shape == (3, 1)


def test_model_keeps_nonlinear_drift_and_destination_indexed_controls(runtime):
    context = runtime.context(runtime.initial_position, runtime.times)
    declared = runtime.model(context)
    state = jnp.array([0.8])
    schedule = runtime.schedule(context)
    # Inspect a single distribution, without invoking a solver or sampler.
    with jax.disable_jit():
        law = declared.state_evolution(
            state, schedule.transition_controls[1], runtime.times[0], runtime.times[1]
        )
    displacement = np.asarray(state) - 0.2
    dt = float(runtime.times[1] - runtime.times[0])
    expected_drift = -0.4 * displacement - 0.2 * displacement**3 + 0.5 * 2.0
    np.testing.assert_allclose(law.mean, state + dt * expected_drift, atol=1e-6)
    np.testing.assert_allclose(law.covariance_matrix, [[dt * 0.3 + 1e-8]], atol=1e-7)
    np.testing.assert_allclose(declared.initial_condition.mean, [0.7])
    np.testing.assert_allclose(declared.initial_condition.covariance_matrix, [[0.8]], atol=2e-6)


def test_exact_target_and_parameter_gradient_trace_through_the_model(runtime):
    path = jnp.array([[0.1], [0.3], [-0.2]])

    def evaluate(position):
        context = runtime.context(position, runtime.times)
        return runtime.path_log_prob(context, path, runtime.observations)

    density, gradient = jax.eval_shape(jax.value_and_grad(evaluate), runtime.initial_position)
    assert density.shape == ()
    assert gradient.shape == (1,)
    factors = jax.eval_shape(
        lambda position: runtime.observation_log_probs(
            runtime.context(position, runtime.times), path, runtime.observations
        ),
        runtime.initial_position,
    )
    assert factors.shape == (3,)


def test_parameter_gradient_remains_dynamic_after_model_partition(runtime):
    previous, current = jnp.array([0.8]), jnp.array([0.9])

    def transition(position):
        context = runtime.context(position, runtime.times)
        return runtime.transition_log_prob(context, previous, current, jnp.array(1))

    # One scalar derivative with JIT disabled; no sampler or trajectory solve.
    with jax.disable_jit():
        gradient = jax.grad(transition)(runtime.initial_position)
    dt = float(runtime.times[1] - runtime.times[0])
    displacement = 0.8 - 0.2
    mean = 0.8 + dt * (-0.4 * displacement - 0.2 * displacement**3 + 1.0)
    mean_gradient = dt * (0.4 + 0.6 * displacement**2)
    expected = (0.9 - mean) * mean_gradient / (dt * 0.3 + 1e-8)
    np.testing.assert_allclose(gradient, [expected], rtol=1e-5, atol=1e-6)


def test_observation_model_keeps_partial_and_complete_missingness(runtime):
    context = runtime.context(runtime.initial_position, runtime.times)
    observation = runtime.model(context).observation_model(jnp.array([0.0]), None, runtime.times[0])
    with jax.disable_jit():
        partial = observation.log_prob(jnp.array([jnp.nan, 2.0]))
        absent = observation.log_prob(jnp.array([jnp.nan, jnp.nan]))
    assert float(partial) == pytest.approx(-1.0 - np.log(2.0), abs=1e-6)
    assert float(absent) == 0.0


def test_native_discrete_law_samples_categories_from_predictors():
    from nof1_causal_lab.models.ssm.execution.contracts import MeasurementParams
    from nof1_causal_lab.models.ssm.execution.dynamical_model import HeterogeneousObservation

    observation = HeterogeneousObservation(
        MeasurementParams(jnp.eye(2), jnp.zeros(2), jnp.eye(2)),
        (DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL),
        (LinkFunction.CUMULATIVE_LOGIT, LinkFunction.SOFTMAX),
        {
            "obs_level_counts": jnp.array([3, 3]),
            "obs_ordered_cutpoints": jnp.array([[-1.0, 1.0], [-1.0, 1.0]]),
            "obs_cat_intercepts": jnp.zeros((2, 2)),
            "obs_cat_slopes": jnp.array([[0.0, 0.0], [-1.0, 1.0]]),
        },
    )
    # At this predictor both declared laws concentrate on their final category.
    # Reconstructing a law from the response mean would lose that information.
    law = observation(jnp.array([100.0, 100.0]), None, jnp.array(0.0))
    for key in (jax.random.key(2), jax.random.PRNGKey(2)):
        draws = law.sample(key, sample_shape=(2, 3))
        np.testing.assert_array_equal(draws, np.full((2, 3, 2), 2))
    np.testing.assert_allclose(law.mean, [2.0, 2.0], atol=1e-6)
    assert np.isfinite(law.log_prob(jnp.array([2.0, 2.0])))


def test_forward_simulation_passes_the_declared_model_to_the_ode_solver(monkeypatch):
    simulator = import_module("nof1_causal_lab.models.ssm.dynamics.simulator")
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec, compile_dynamics
    

    field = compile_dynamics(
        DynamicsSpec(
            1, (potential_term(0, center=FixedCoefficient(value=0), stiffness=FixedCoefficient(value=0.4), quartic=FixedCoefficient(value=0.2)),)
        )
    ).vector_field
    params = ({},)
    grid = jnp.array([2.0, 2.2, 2.5])
    initial = jnp.array([0.8])
    paths = jnp.zeros((3, 1))

    def solve(
        model, *, initial_state, t0, path_times, ctrl_times, ctrl_values, diffeqsolve_settings
    ):
        assert isinstance(model, dsx.DynamicalModel)
        assert ctrl_times is None
        assert ctrl_values is None
        np.testing.assert_allclose(
            model.state_evolution.total_drift(x=initial, u=None, t=t0),
            -0.4 * initial - 0.2 * initial**3,
        )
        np.testing.assert_array_equal(initial_state, initial)
        np.testing.assert_array_equal(path_times, grid)
        assert diffeqsolve_settings["throw"] is False
        assert diffeqsolve_settings["max_steps"] == 4096
        assert float(diffeqsolve_settings["dt0"]) == pytest.approx(0.5 / 256)
        return paths

    monkeypatch.setattr(simulator, "solve_ode_state_path", solve)
    actual = simulator.simulate(field, params, Intervention.none(), initial, grid)
    assert actual is paths


def test_indexed_sde_keeps_its_brownian_path_and_uses_dynestyx_evolution(monkeypatch):
    simulator = import_module("nof1_causal_lab.models.ssm.dynamics.simulator")
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec, compile_dynamics
    

    field = compile_dynamics(
        DynamicsSpec(
            1, (potential_term(0, center=FixedCoefficient(value=0), stiffness=FixedCoefficient(value=0.4), quartic=FixedCoefficient(value=0.2)),)
        )
    ).vector_field
    params = ({},)
    grid = jnp.array([2.0, 2.2, 2.5])
    initial = jnp.array([0.8])
    covariance = jnp.array([[0.3]])
    key = jax.random.key(5)
    paths = jnp.zeros((3, 1))

    def solve(terms, _solver, **settings):
        evolution = settings["args"][0]
        assert isinstance(evolution, dsx.StochasticContinuousTimeStateEvolution)
        brownian = terms.terms[1].control
        assert isinstance(brownian, simulator._IndexedBrownianPath)
        np.testing.assert_array_equal(jax.random.key_data(brownian.key), jax.random.key_data(key))
        np.testing.assert_allclose(
            evolution.diffusion.gram_matrix(x=initial, u=None, t=grid[0], state_dim=1),
            covariance,
            atol=1e-7,
        )
        base_drift = -0.4 * initial - 0.2 * initial**3
        for time, forcing in [(grid[0], 1.0), (grid[1], -0.5), (grid[-1], -0.5)]:
            np.testing.assert_allclose(
                terms.terms[0].vf(time, initial, settings["args"]), base_drift + forcing
            )
        assert float(settings["dt0"]) == pytest.approx(0.01)
        assert isinstance(settings["adjoint"], simulator.dfx.ForwardMode)
        return SimpleNamespace(ys=paths)

    monkeypatch.setattr(simulator.dfx, "diffeqsolve", solve)
    actual = simulator.simulate(
        field,
        params,
        Intervention.none(),
        initial,
        grid,
        config=simulator.SimulationConfig(sde_dt=0.01, use_indexed_brownian_path=True),
        key=key,
        diffusion_cov=covariance,
        input_effect=jnp.array([[0.5]]),
        transition_inputs=jnp.array([[99.0], [2.0], [-1.0]]),
    )
    assert actual is paths
