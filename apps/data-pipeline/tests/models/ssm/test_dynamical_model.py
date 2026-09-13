"""Small model-boundary checks; no fitting or trajectory simulation."""

from importlib import import_module
from types import SimpleNamespace

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from dynestyx.inference.particle_runtime import Parameterization

from nof1_causal_lab.artifacts.statistical_model_spec import LinkFunction
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.dynamics.edges import NodePotential
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField
from nof1_causal_lab.models.ssm.execution.contracts import MeasurementParams
from nof1_causal_lab.models.ssm.execution.dynamical_model import continuous_state_evolution
from nof1_causal_lab.models.ssm.inference import problem as problem_module
from nof1_causal_lab.models.ssm.inference.targets.laplace.shared import _prepare_linearized_path


@pytest.fixture
def runtime(monkeypatch):
    # Only parameter discovery is stubbed. The declared drift, observation model,
    # model partition, discretizer, and particle target are the actual code paths.
    times = jnp.array([2.0, 2.1, 2.35])
    spec = SimpleNamespace(
        n_manifest=2,
        input_names=["forcing"],
        diffusion_dists=[DistributionFamily.GAUSSIAN],
        manifest_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.POISSON],
        manifest_links=[LinkFunction.IDENTITY, LinkFunction.LOG],
    )
    model = SimpleNamespace(
        spec=spec,
        observation_support=None,
        transition_inputs=jnp.array([[99.0], [2.0], [-1.0]]),
    )
    parameters = Parameterization(
        initial_position=jnp.array([0.2]),
        unravel=lambda z: {"center": z[0]},
        constrain=lambda z: {"center": z[0]},
        log_prior=lambda z: -jnp.sum(z**2),
    )
    monkeypatch.setattr(
        problem_module, "prepare_model_parameters", lambda *_: (parameters, {}, {"center"})
    )
    monkeypatch.setattr(problem_module, "build_site_registry", lambda _: None)
    field = VectorField(n_latent=1, components=(NodePotential(target=0),))

    def assemble(samples, _spec, *, registry):
        del registry
        evolution = continuous_state_evolution(
            field,
            (
                {
                    "center": samples["center"],
                    "stiffness": jnp.array(0.4),
                    "quartic": jnp.array(0.2),
                },
            ),
            jnp.array([[0.3]]),
            jnp.array([[0.5]]),
        )
        measurement = MeasurementParams(jnp.ones((2, 1)), jnp.zeros(2), jnp.eye(2))
        initial = dist.MultivariateNormal(jnp.array([0.7]), covariance_matrix=jnp.array([[0.8]]))
        return evolution, measurement, initial, None

    monkeypatch.setattr(problem_module, "_assemble_likelihood_inputs", assemble)
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
    np.testing.assert_allclose(declared.initial_condition.covariance_matrix, [[0.8]])


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


def test_forward_simulation_passes_the_declared_model_to_the_ode_solver(monkeypatch):
    simulator = import_module("nof1_causal_lab.models.ssm.dynamics.simulator")
    field = VectorField(n_latent=1, components=(NodePotential(target=0),))
    params = ({"center": jnp.array(0.0), "stiffness": jnp.array(0.4), "quartic": jnp.array(0.2)},)
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
    field = VectorField(n_latent=1, components=(NodePotential(target=0),))
    params = ({"center": jnp.array(0.0), "stiffness": jnp.array(0.4), "quartic": jnp.array(0.2)},)
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
            covariance + 1e-8,
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
