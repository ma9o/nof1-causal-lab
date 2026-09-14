"""Exact measurement semantics across authoring, compilation, and observation execution."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import state
from nof1_causal_lab.artifacts.identity import ConstructId
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec, ObservationLawSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.equations import observation_equations
from nof1_causal_lab.models.likelihoods import observation_law
from nof1_causal_lab.models.model_mechanisms import default_model
from nof1_causal_lab.models.model_parameters import iter_coefficient_uses
from nof1_causal_lab.models.prior_planning import complete_model
from nof1_causal_lab.models.ssm import SSMModel
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.observation_model import compile_observation_model
from nof1_causal_lab.models.ssm.inference import problem as problem_module
from nof1_causal_lab.models.ssm.inference.backend_factory import build_laplace_backend
from nof1_causal_lab.models.ssm.inference.conditioning import compile_exact_state_constraints
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._math import (
    _masked_mean,
    _masked_normal_log_prob,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.runner import (
    _initialize_chain_state,
)
from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime
from tests.helpers import complete_test_model, make_model


@pytest.fixture
def exact_model():
    model = make_model(["setting", "response"], [("setting", "response")])
    owner = model.constructs[0]
    indicator = owner.indicators[0].model_copy(
        update={
            "aggregation": "last",
            "likelihood": LikelihoodSpec(
                law=observation_law(owner.id, "delta", "identity"),
                reasoning="The recorded setting is exact at its observation anchor.",
            ),
        }
    )
    return complete_test_model(
        model.revised(
            edges=replace_constructs(
                model.edges, (owner.model_copy(update={"indicators": (indicator,)}),)
            )
        )
    )


def test_exact_binding_roundtrips_without_authored_measurement_parameters(exact_model):
    model = ModelSpec.model_validate_json(exact_model.model_dump_json())
    owner = model.constructs[0]
    indicator = owner.indicators[0]
    likelihood = indicator.likelihood
    assert likelihood is not None
    assert likelihood.law.arguments == {"v": state(owner.id)}
    assert likelihood.terms.loadings[owner.id].value == 1
    assert likelihood.terms.intercept.value == 0
    assert not likelihood.terms.auxiliary
    assert all(
        isinstance(use.value, (int, float))
        for use in iter_coefficient_uses(model)
        if any(ref.id == indicator.id for ref in use.owners)
    )
    np.testing.assert_array_equal(numeric.loading_block(model).template, np.eye(2))
    assert numeric.observation_mean_block(model).template[0] == 0
    assert not numeric.observation_noise_block(model).diag_support[0]
    np.testing.assert_array_equal(numeric.observation_noise_block(model).template[0], [0, 0])
    assert owner.id in model.state_order
    assert "usage" not in type(owner).model_fields
    anchor = next(item for item in model.check_execution() if item.construct_id == owner.id)
    assert anchor.location_anchor == "exact_state_observation"
    assert r"\operatorname{Delta}" in observation_equations(model)[indicator.id]


def test_delta_constructor_requires_only_its_exact_value():
    for arguments in (
        {},
        {"loc": state(ConstructId("construct:x"))},
        {"v": state(ConstructId("construct:x")), "scale": 0},
    ):
        with pytest.raises(ValidationError):
            ObservationLawSpec(distribution="Delta", arguments=arguments)


def test_authored_affine_delta_keeps_its_calibration_coefficients(exact_model):
    owner = exact_model.constructs[0]
    predictor = observation_law(owner.id, "gaussian", "identity").arguments["loc"]
    likelihood = LikelihoodSpec(
        law=ObservationLawSpec(distribution="Delta", arguments={"v": predictor}),
        reasoning="Exact measurement with an unknown calibration offset.",
    )
    indicator = owner.indicators[0].model_copy(update={"likelihood": likelihood})
    model = complete_model(
        exact_model.revised(
            edges=replace_constructs(
                exact_model.edges, (owner.model_copy(update={"indicators": (indicator,)}),)
            )
        )
    )
    completed = model.indicator(indicator.id).likelihood
    assert completed is not None
    assert isinstance(completed.terms.intercept.value, str)
    assert not completed.terms.auxiliary
    model.check_execution()


@pytest.mark.parametrize(
    ("dtype", "default_family"),
    [
        ("continuous", "gaussian"),
        ("binary", "bernoulli"),
        ("count", "poisson"),
        ("ordinal", "ordered_logistic"),
        ("categorical", "categorical"),
    ],
)
def test_exact_measurement_is_available_but_never_selected_by_default(dtype, default_family):
    model = make_model(["setting", "response"], [("setting", "response")])
    owner = model.constructs[0]
    updates = {"measurement_dtype": dtype, "aggregation": "last"}
    if dtype in {"ordinal", "categorical"}:
        updates[f"{dtype}_levels"] = ("low", "high")
    indicator = owner.indicators[0].model_copy(update=updates)
    model = model.revised(
        edges=replace_constructs(
            model.edges, (owner.model_copy(update={"indicators": (indicator,)}),)
        )
    )
    proposed = default_model(model).indicator(indicator.id).likelihood
    assert proposed is not None
    assert proposed.law.family == default_family
    exact = LikelihoodSpec(
        law=observation_law(owner.id, "delta", "identity"), reasoning="Explicit exact measurement"
    )
    model.revised(
        edges=replace_constructs(
            model.edges,
            (
                owner.model_copy(
                    update={"indicators": (indicator.model_copy(update={"likelihood": exact}),)}
                ),
            ),
        )
    )


def test_mixed_delta_density_draws_and_missingness_remain_exact():
    covariance = jnp.eye(2) * 100.0
    compiled = compile_observation_model(["delta", "poisson"], manifest_cov=covariance)
    kernel = compiled.kernel
    predictor = jnp.array([-2.0, jnp.log(3.0)])
    observed = jnp.array([-2.0, 2.0])
    expected = dist.Poisson(3.0).log_prob(2.0)
    density = jax.jit(kernel.log_prob_fn)
    np.testing.assert_allclose(density(observed, predictor, covariance, jnp.ones(2)), expected)
    mismatch = observed.at[0].set(jnp.nextafter(observed[0], 0.0))
    assert jnp.isneginf(density(mismatch, predictor, covariance, jnp.ones(2)))
    np.testing.assert_allclose(
        density(observed.at[0].set(jnp.nan), predictor, covariance, jnp.array([0.0, 1.0])),
        expected,
    )
    assert density(jnp.full(2, jnp.nan), predictor, covariance, jnp.zeros(2)) == 0
    np.testing.assert_array_equal(kernel.variance_fn(kernel.response_fn(predictor))[0], [0, 0])
    draw = compiled.point_sampler.sample_point(jax.random.key(2), predictor)
    assert draw[0] == predictor[0]
    with pytest.raises(ValueError, match="no smooth log-density"):
        kernel.latent_grad_hess_fn(
            observed, predictor, jnp.eye(2), jnp.zeros(2), covariance, jnp.ones(2)
        )


def test_exact_window_mean_constrains_the_summary_without_pinning_the_path():
    support = ObservationSupportRuntime(
        anchor_times=np.array([0.0, 2.0]),
        manifest_names=["setting_mean"],
        support_kinds=["interval"],
        summary_operators=["mean"],
        anchor_policies=["support_end"],
        observation_windows=["2d"],
        support_start_times=np.array([[np.nan], [0.0]]),
        support_end_times=np.array([[np.nan], [2.0]]),
        interval_prev_coeffs=np.array([[[0.0]], [[1.0]]]),
        interval_curr_coeffs=np.array([[[0.0]], [[1.0]]]),
        interval_weights=np.array([[[0.0]], [[2.0]]]),
        emission_slot_indices=np.array([[-1], [0]]),
    )
    covariance = jnp.zeros((1, 1))
    compiled = compile_observation_model(
        ["delta"], manifest_cov=covariance, observation_support=support
    )
    assert compiled.mean_log_prob_fn is not None
    assert compiled.interval_summary_sampler is not None
    for path in ([[0.0], [4.0]], [[4.0], [0.0]], [[2.0], [2.0]]):
        means, mask = compiled.observation_operator.project_response_trajectory(jnp.array(path))
        np.testing.assert_array_equal(mask, [[0.0], [1.0]])
        assert means[-1, 0] == 2.0
        assert compiled.mean_log_prob_fn(jnp.array([2.0]), means[-1], covariance, mask[-1]) == 0
        assert jnp.isneginf(
            compiled.mean_log_prob_fn(jnp.array([3.0]), means[-1], covariance, mask[-1])
        )
        draws = compiled.interval_summary_sampler.sample_mean_trajectory(jax.random.key(0), means)
        np.testing.assert_array_equal(draws, means)


@pytest.mark.parametrize("unsupported", ["affine", "interval"])
def test_unsupported_delta_constraints_fail_before_parameter_initialization(
    exact_model, monkeypatch, unsupported
):
    owner = exact_model.constructs[0]
    indicator = owner.indicators[0]
    if unsupported == "interval":
        indicator = indicator.model_copy(update={"aggregation": "mean"})
    else:
        indicator = indicator.model_copy(
            update={
                "likelihood": LikelihoodSpec(
                    law=ObservationLawSpec(
                        distribution="Delta",
                        arguments={
                            "v": observation_law(owner.id, "gaussian", "identity").arguments["loc"]
                        },
                    ),
                    reasoning="An affine equality needs a different constraint parameterization.",
                )
            }
        )
    exact_model = exact_model.revised(
        edges=replace_constructs(
            exact_model.edges, (owner.model_copy(update={"indicators": (indicator,)}),)
        )
    )

    def unexpected_initialization(*_args):
        pytest.fail("An unsupported exact constraint must fail before numerical initialization")

    monkeypatch.setattr(problem_module, "prepare_model_parameters", unexpected_initialization)
    with pytest.raises(ValueError, match=r"setting_obs.*direct point binding"):
        problem_module.build_particle_problem(
            SimpleNamespace(spec=exact_model, observation_support=None),
            jnp.array([[1.0, 0.0], [jnp.nan, 0.5]]),
            jnp.array([0.0, 1.0]),
            scheme="euler_maruyama",
            trace_key=jax.random.key(0),
            reparam=None,
        )


def test_sparse_exact_observations_leave_missing_coordinates_free(exact_model):
    constraints = compile_exact_state_constraints(
        exact_model, jnp.array([[1.0, 0.0], [jnp.nan, 0.5], [2.0, 1.0]])
    )
    assert constraints is not None
    np.testing.assert_array_equal(
        constraints.free_mask, [[False, True], [True, True], [False, True]]
    )
    path = jnp.arange(6.0).reshape(3, 2)
    projected = constraints.project(path)
    np.testing.assert_array_equal(projected, [[1.0, 1.0], [2.0, 3.0], [2.0, 5.0]])
    np.testing.assert_array_equal(constraints.project(jnp.stack([path, path]))[1], projected)
    with pytest.raises(ValueError, match="finite or missing"):
        compile_exact_state_constraints(exact_model, jnp.array([[jnp.inf, 0.0]]))


def test_multiple_exact_indicators_must_agree_at_shared_times(exact_model):
    from nof1_causal_lab.artifacts.identity import scientific_id

    owner = exact_model.constructs[0]
    original = owner.indicators[0]
    duplicate = original.model_copy(
        update={"id": scientific_id("indicator", "second_recording"), "name": "second_recording"}
    )
    model = exact_model.revised(
        edges=replace_constructs(
            exact_model.edges, (owner.model_copy(update={"indicators": (original, duplicate)}),)
        )
    )
    columns = {identity: column for column, identity in enumerate(model.manifest_indicator_order)}
    observations = jnp.full((2, 3), jnp.nan)
    observations = observations.at[:, columns[original.id]].set(jnp.array([1.0, jnp.nan]))
    observations = observations.at[:, columns[duplicate.id]].set(jnp.array([1.0, 4.0]))
    constraints = compile_exact_state_constraints(model, observations)
    assert constraints is not None
    np.testing.assert_array_equal(constraints.values[:, 0], [1.0, 4.0])
    with pytest.raises(ValueError, match="Conflicting exact observations"):
        compile_exact_state_constraints(model, observations.at[0, columns[duplicate.id]].set(2.0))


@pytest.mark.parametrize("mask", [[True, False], [False, True], [True, True], [False, False]])
def test_proposal_density_uses_the_free_coordinate_measure(mask):
    values, mean, variance = jnp.array([1.5, -2.0]), jnp.array([0.2, 0.4]), jnp.array([0.3, 2.0])
    free = np.asarray(mask)
    expected = dist.Normal(mean[free], jnp.sqrt(variance[free])).log_prob(values[free]).sum()
    np.testing.assert_allclose(
        _masked_normal_log_prob(values, mean, variance, free), expected, atol=1e-6
    )
    grad = jax.grad(lambda x: _masked_normal_log_prob(x, mean, variance, free))(values)
    np.testing.assert_array_equal(grad[~free], jnp.zeros((~free).sum()))


def test_fixed_coordinates_are_excluded_from_sampler_freeze_diagnostics():
    free = jnp.array([[False, True], [False, False], [False, True]])
    frozen = jnp.array([[True, False], [True, True], [True, True]])
    assert _masked_mean(frozen, free) == 0.5
    np.testing.assert_array_equal(_masked_mean(frozen, free, axis=0), [0.0, 0.5])
    np.testing.assert_array_equal(_masked_mean(frozen, free, axis=-1), [0.0, 0.0, 1.0])


@pytest.fixture
def point_problem(exact_model):
    return problem_module.build_particle_problem(
        SSMModel(exact_model),
        jnp.array([[1.0, jnp.nan], [jnp.nan, jnp.nan], [2.0, jnp.nan]]),
        jnp.array([0.0, 0.4, 1.0]),
        scheme="euler_maruyama",
        trace_key=jax.random.key(0),
        reparam=None,
    )


def _conditioned_initial_state(problem):
    runtime = problem.runtime
    return _initialize_chain_state(
        runtime.initial_position,
        observations=runtime.observations,
        times=runtime.times,
        target=runtime,
        initial_latent_delta=jnp.full(runtime.times.shape, 0.2),
        param_step_size=0.01,
        param_min_scale=1e-6,
        param_max_scale=1e3,
        param_target_accept=0.35,
        initial_latent_trajectory=jnp.full((3, 2), 0.25),
        exact_constraints=problem.exact_constraints,
    )


def test_conditioned_target_preserves_initial_and_transition_evidence(point_problem):
    runtime = point_problem.runtime
    state = _conditioned_initial_state(point_problem)
    path = state.latent_trajectory
    np.testing.assert_array_equal(path[:, 0], [1.0, 0.25, 2.0])
    dynamics = runtime.model(state.latent_context)
    expected = dynamics.initial_condition.log_prob(path[0])
    for index in (1, 2):
        expected += dynamics.state_evolution(
            path[index - 1], None, runtime.times[index - 1], runtime.times[index]
        ).log_prob(path[index])
    np.testing.assert_allclose(state.trajectory_log_prob, expected, rtol=1e-6)
    assert jnp.isfinite(state.complete_log_posterior)
    assert jnp.isneginf(
        runtime.path_log_prob(state.latent_context, path.at[0, 0].set(0.0), runtime.observations)
    )
    derivative = jax.grad(
        lambda z: runtime.log_posterior(z, path, runtime.observations, runtime.times)
    )(runtime.initial_position)
    assert jnp.isfinite(derivative).all()


def test_gaussian_view_is_confined_to_warmup(exact_model):
    warmup = build_laplace_backend(exact_model, n_ieks_iters=1)
    assert warmup.manifest_dists == ["gaussian", "gaussian"]
    assert numeric.observation_families(exact_model) == ["delta", "gaussian"]


@pytest.mark.parametrize("proposal", ["amala_exact", "paid_mix"])
def test_conditioned_smoother_traces_with_missing_and_observed_coordinates(point_problem, proposal):
    runtime = point_problem.runtime
    state = _conditioned_initial_state(point_problem)
    path = state.latent_trajectory
    kernel = build_marginal_particle_gibbs_kernel(
        runtime,
        num_particles=4,
        num_parameter_particles=2,
        param_step_size=0.01,
        exact_constraints=point_problem.exact_constraints,
        dsmc_leaf_proposal=proposal,
        latent_block_coords=1,
        pilot_means=path,
        pilot_vars=jnp.ones_like(path),
        pilot_wide_vars=jnp.ones_like(path) * 4,
    )
    shape, _ = jax.eval_shape(kernel.step_fn, state, jax.random.key(1))
    assert shape.latent_trajectory.shape == (3, 2)


@pytest.mark.inference
@pytest.mark.parametrize("proposal", ["amala_exact", "paid_mix"])
def test_particle_updates_keep_exact_readings_and_move_missing_states(point_problem, proposal):
    runtime = point_problem.runtime
    state = _conditioned_initial_state(point_problem)
    path = state.latent_trajectory
    kernel = build_marginal_particle_gibbs_kernel(
        runtime,
        num_particles=16,
        num_parameter_particles=2,
        param_step_size=0.01,
        exact_constraints=point_problem.exact_constraints,
        dsmc_leaf_proposal=proposal,
        latent_block_coords=1,
        pilot_means=path,
        pilot_vars=jnp.ones_like(path),
        pilot_wide_vars=jnp.ones_like(path) * 4,
    )
    step = jax.jit(kernel.step_fn)
    missing_values = []
    for key in jax.random.split(jax.random.key(7), 8):
        state, _ = step(state, key)
        np.testing.assert_array_equal(state.latent_trajectory[[0, 2], 0], [1.0, 2.0])
        assert jnp.isfinite(state.complete_log_posterior)
        missing_values.append(state.latent_trajectory[1, 0])
    assert float(jnp.var(jnp.stack(missing_values))) > 0
