"""Conditional laws retain native density semantics, ownership, and explicit completion."""

import operator

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from pydantic import ValidationError
from scripts.migrate_likelihood_expressions import convert_likelihood, convert_payload

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient, ParameterCoefficient
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    CoefficientExpression,
    coefficient,
    fold_expression,
    state,
)
from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec, ObservationLaw
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.machine.equations import observation_equations
from nof1_causal_lab.models.likelihoods import function, observation_law, revise_law
from nof1_causal_lab.models.model_parameters import iter_coefficient_uses
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.observation_dispatch import get_emission_fn
from tests.helpers import complete_test_model, make_model

CASES = [
    ("gaussian", "identity", 0.4),
    ("student_t", "identity", 0.4),
    ("poisson", "log", 2),
    ("gamma", "log", 1.2),
    ("gamma", "inverse", 1.2),
    ("bernoulli", "logit", 1),
    ("bernoulli", "probit", 1),
    ("negative_binomial", "log", 3),
    ("beta", "logit", 0.4),
    ("beta", "probit", 0.4),
    ("ordered_logistic", "cumulative_logit", 1),
    ("categorical", "softmax", 2),
]


@pytest.mark.parametrize(
    ("family", "link", "observed"),
    [*CASES, ("delta", "identity", 0.35), ("delta", "identity", 0.4)],
)
def test_native_conditional_law_matches_exact_emission_lowering(family, link, observed):
    likelihood = LikelihoodSpec(
        law=observation_law("construct:x", family, link), reasoning="Native density parity"
    )
    likelihood = revise_law(
        likelihood,
        lambda node: (
            node.model_copy(
                update={
                    "coefficient": ParameterCoefficient(
                        parameter_id=scientific_id("parameter", node.role)
                    )
                }
            )
            if isinstance(node, CoefficientExpression)
            else node
        ),
    )
    values = {
        "loading": 0.8,
        "observation_intercept": 0.3,
        "observation_scale": 0.7,
        "degrees_of_freedom": 7.0,
        "shape": 2.0,
        "dispersion": 4.0,
        "concentration": 5.0,
        "cutpoint_base": -0.6,
        "cutpoint_gaps": jnp.array([1.1]),
        "category_intercepts": jnp.array([-0.5, 0.6]),
        "category_slopes": jnp.array([0.8, -0.3]),
    }
    functions = {
        "exp": jnp.exp,
        "sigmoid": jax.nn.sigmoid,
        "normal_cdf": jax.scipy.special.ndtr,
        "ordered_cutpoints": lambda base, gaps: jnp.concatenate(
            (jnp.array([base]), base + jnp.cumsum(gaps))
        ),
        "category_logits": lambda predictor, intercepts, slopes: jnp.concatenate(
            (jnp.zeros(1), intercepts + slopes * predictor)
        ),
    }
    operations = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
        "power": operator.pow,
        "maximum": jnp.maximum,
    }

    def evaluate(expression):
        return fold_expression(
            expression,
            literal=jnp.asarray,
            state_value=lambda _identity: jnp.asarray(0.35),
            coefficient_value=lambda operand: values[operand.role],
            binary=lambda operation, left, right: operations[operation](left, right),
            call=lambda name, arguments: functions[name](*arguments),
        )

    arguments = {name: evaluate(value) for name, value in likelihood.law.arguments.items()}
    expected = getattr(dist, likelihood.law.distribution)(**arguments).log_prob(observed)
    terms = likelihood.terms
    extra = {
        operand.meaning.quantity.value: values[operand.role]
        for operand in terms.auxiliary
        if operand.role != "observation_scale"
    }
    if family == "ordered_logistic":
        extra["obs_ordered_cutpoints"] = arguments["cutpoints"][None, :]
        extra["obs_level_counts"] = jnp.array([3])
    if family == "categorical":
        extra["obs_cat_intercepts"] = values["category_intercepts"][None, :]
        extra["obs_cat_slopes"] = values["category_slopes"][None, :]
        extra["obs_level_counts"] = jnp.array([3])
    density = get_emission_fn(terms.family, extra_params=extra, link=terms.link)
    actual = density(
        jnp.array([observed], dtype=float),
        jnp.atleast_1d(evaluate(terms.predictor)),
        jnp.array([[0.49]]),
        jnp.ones(1),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert (
        float(
            density(
                jnp.array([observed], dtype=float),
                jnp.atleast_1d(evaluate(terms.predictor)),
                jnp.array([[0.49]]),
                jnp.zeros(1),
            )
        )
        == 0
    )
    assert (terms.family, terms.link) == (family, link)


@pytest.mark.parametrize(("family", "link", "_observed"), CASES)
def test_offline_conversion_preserves_authored_coefficient_identities(family, link, _observed):
    retired = {
        "distribution": family,
        "link": link,
        "standardized": False,
        "loading": {"kind": "fixed", "value": -1},
        "intercept": {"kind": "parameter", "parameter_id": scientific_id("parameter", "baseline")},
        "reasoning": "Retained scientific decision",
        "sources": [],
    }
    converted = LikelihoodSpec.model_validate(convert_likelihood(retired, "construct:x"))
    assert converted.terms.loadings["construct:x"].coefficient == FixedCoefficient(value=-1)
    assert converted.terms.intercept.coefficient == ParameterCoefficient(
        parameter_id=scientific_id("parameter", "baseline")
    )
    assert (converted.terms.family, converted.terms.link) == (family, link)
    assert set(converted.model_dump()) == {"law", "standardized", "reasoning", "sources"}
    with pytest.raises(ValidationError, match="Extra inputs"):
        LikelihoodSpec.model_validate(retired)


def test_completion_binding_equations_and_migration_follow_the_same_cross_loading():
    model = complete_test_model(make_model(["X", "Y"], [("X", "Y")]))
    owner, other = model.constructs
    indicator = owner.indicators[0]
    likelihood = indicator.likelihood
    predictor = likelihood.terms.predictor
    extended = predictor + state(other.id) * coefficient(FixedCoefficient(value=0.25), "loading")
    revised = revise_law(likelihood, lambda node: extended if node == predictor else node)
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            (
                owner.model_copy(
                    update={"indicators": (indicator.model_copy(update={"likelihood": revised}),)}
                ),
            ),
        )
    )
    assert model.execution_readiness.ready
    np.testing.assert_allclose(numeric.loading_block(model).template, [[1, 0.25], [0, 1]])
    uses = [use for use in iter_coefficient_uses(model) if use.quantity == SiteKind.LOADING]
    cross = next(use for use in uses if use.coefficient == FixedCoefficient(value=0.25))
    assert {ref.id for ref in cross.owners} == {indicator.id, other.id}
    equation = observation_equations(model)[indicator.id]
    assert r"\operatorname{Normal}" in equation
    assert r"0.25" in equation
    assert r"\eta_{\text{Y}}(t)" in equation
    assert ModelSpec.model_validate(convert_payload(model.model_dump(mode="json"))) == model
    renamed = model.revised(
        edges=replace_constructs(model.edges, (other.model_copy(update={"name": "Renamed"}),))
    )
    assert r"\eta_{\text{Renamed}}(t)" in observation_equations(renamed)[indicator.id]
    assert {p.id for p in renamed.parameters} == {p.id for p in model.parameters}


def test_partial_law_is_explicit_and_unsupported_formulas_fail_before_execution():
    likelihood = LikelihoodSpec(
        law=observation_law("construct:x", "gaussian", "identity"), reasoning="Partial"
    )
    assert all(operand.coefficient is None for operand in likelihood.terms.operands)
    with pytest.raises(ValidationError, match="affine"):
        LikelihoodSpec(
            law=ObservationLaw(
                distribution="Normal",
                arguments={
                    "loc": function("exp", likelihood.terms.predictor),
                    "scale": coefficient(None, "observation_scale"),
                },
            ),
            reasoning="Unsupported nonlinear observation predictor",
        )
    with pytest.raises(ValidationError, match="exactly"):
        ObservationLaw(distribution="Normal", arguments={"rate": likelihood.terms.predictor})
    with pytest.raises(ValidationError, match="non-negative"):
        coefficient(FixedCoefficient(value=-1), "observation_scale")

    model = make_model(["X", "Y"], [("X", "Y")])
    owner = model.constructs[0]
    indicator = owner.indicators[0]
    partial = LikelihoodSpec(
        law=observation_law(owner.id, "gaussian", "identity"), reasoning="Partial"
    )

    def with_law(law):
        return model.revised(
            edges=replace_constructs(
                model.edges,
                (
                    owner.model_copy(
                        update={"indicators": (indicator.model_copy(update={"likelihood": law}),)}
                    ),
                ),
            )
        )

    unfinished = with_law(partial)
    assert not unfinished.execution_readiness.ready
    assert "?" in observation_equations(unfinished)[indicator.id]
    assert complete_test_model(unfinished).execution_readiness.ready
    with pytest.raises(ValidationError, match="unknown constructs"):
        with_law(likelihood)
    wrong_owner = LikelihoodSpec(
        law=observation_law(model.constructs[1].id, "gaussian", "identity"), reasoning="Wrong owner"
    )
    with pytest.raises(ValidationError, match="must include its measured construct"):
        with_law(wrong_owner)
