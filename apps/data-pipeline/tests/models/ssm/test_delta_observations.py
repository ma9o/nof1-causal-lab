"""Exact measurement semantics across authoring, compilation, and observation execution."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient, ParameterCoefficient
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import state
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec, ObservationLaw
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.equations import observation_equations
from nof1_causal_lab.models.likelihoods import observation_law
from nof1_causal_lab.models.model_mechanisms import default_model
from nof1_causal_lab.models.model_parameters import iter_coefficient_uses
from nof1_causal_lab.models.prior_planning import complete_model
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.observation_model import compile_observation_model
from nof1_causal_lab.models.ssm.inference import problem as problem_module
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
    assert likelihood.terms.loadings[owner.id].coefficient == FixedCoefficient(value=1)
    assert likelihood.terms.intercept.coefficient == FixedCoefficient(value=0)
    assert not likelihood.terms.auxiliary
    assert all(
        isinstance(use.coefficient, FixedCoefficient)
        for use in iter_coefficient_uses(model)
        if any(ref.id == indicator.id for ref in use.owners)
    )
    np.testing.assert_array_equal(numeric.loading_block(model).template, np.eye(2))
    assert numeric.observation_mean_block(model).template[0] == 0
    assert not numeric.observation_noise_block(model).diag_support[0]
    np.testing.assert_array_equal(numeric.observation_noise_block(model).template[0], [0, 0])
    assert owner.id in model.state_order
    assert not model.known_inputs
    assert model.execution_readiness.ready
    anchor = next(
        item
        for item in model.execution_readiness.anchor_certificates
        if item.construct_id == owner.id
    )
    assert anchor.location_anchor == "exact_state_observation"
    assert r"\operatorname{Delta}" in observation_equations(model)[indicator.id]


def test_delta_constructor_requires_only_its_exact_value():
    for arguments in ({}, {"loc": state("construct:x")}, {"v": state("construct:x"), "scale": 0}):
        with pytest.raises(ValidationError):
            ObservationLaw(distribution="Delta", arguments=arguments)


def test_authored_affine_delta_keeps_its_calibration_coefficients(exact_model):
    owner = exact_model.constructs[0]
    predictor = observation_law(owner.id, "gaussian", "identity").arguments["loc"]
    likelihood = LikelihoodSpec(
        law=ObservationLaw(distribution="Delta", arguments={"v": predictor}),
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
    assert isinstance(completed.terms.intercept.coefficient, ParameterCoefficient)
    assert not completed.terms.auxiliary
    assert model.execution_readiness.ready


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


def test_particle_backend_rejects_exact_constraints_before_parameter_initialization(
    exact_model, monkeypatch
):
    def unexpected_initialization(*_args):
        pytest.fail("An unsupported exact constraint must fail before numerical initialization")

    monkeypatch.setattr(problem_module, "prepare_model_parameters", unexpected_initialization)
    with pytest.raises(ValueError, match=r"setting_obs.*constraint-preserving particle"):
        problem_module.build_particle_problem(
            SimpleNamespace(spec=exact_model, observation_support=None),
            jnp.array([[1.0, 0.0], [jnp.nan, 0.5]]),
            jnp.array([0.0, 1.0]),
            scheme="euler_maruyama",
            trace_key=jax.random.key(0),
            reparam=None,
        )
