"""Conditional laws retain native density semantics, ownership, and explicit completion."""

from pathlib import Path

import operator

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    CoefficientExpression,
    coefficient,
    fold_expression,
    state,
)
from nof1_causal_lab.artifacts.identity import ConstructId, scientific_id
from nof1_causal_lab.artifacts.likelihood import (
    DistributionFamily,
    LikelihoodSpec,
    LinkFunction,
    ObservationLawSpec,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.models.likelihoods import function
from nof1_causal_lab.models.model_parameters import iter_coefficient_uses
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.observation_dispatch import get_emission_fn
from nof1_causal_lab.study.equations import observation_equations
from tests.helpers import make_model

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


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(('family', 'link', 'observed', 'observation_law_payload', 'revise_law_payload'), [
    pytest.param('gaussian', 'identity', 0.4, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_gaussian-identity-0_4.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_gaussian-identity-0_4.json', id='gaussian-identity-0.4'),
    pytest.param('student_t', 'identity', 0.4, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_student_t-identity-0_4.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_student_t-identity-0_4.json', id='student_t-identity-0.4'),
    pytest.param('poisson', 'log', 2, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_poisson-log-2.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_poisson-log-2.json', id='poisson-log-2'),
    pytest.param('gamma', 'log', 1.2, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_gamma-log-1_2.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_gamma-log-1_2.json', id='gamma-log-1.2'),
    pytest.param('gamma', 'inverse', 1.2, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_gamma-inverse-1_2.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_gamma-inverse-1_2.json', id='gamma-inverse-1.2'),
    pytest.param('bernoulli', 'logit', 1, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_bernoulli-logit-1.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_bernoulli-logit-1.json', id='bernoulli-logit-1'),
    pytest.param('bernoulli', 'probit', 1, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_bernoulli-probit-1.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_bernoulli-probit-1.json', id='bernoulli-probit-1'),
    pytest.param('negative_binomial', 'log', 3, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_negative_binomial-log-3.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_negative_binomial-log-3.json', id='negative_binomial-log-3'),
    pytest.param('beta', 'logit', 0.4, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_beta-logit-0_4.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_beta-logit-0_4.json', id='beta-logit-0.4'),
    pytest.param('beta', 'probit', 0.4, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_beta-probit-0_4.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_beta-probit-0_4.json', id='beta-probit-0.4'),
    pytest.param('ordered_logistic', 'cumulative_logit', 1, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_ordered_logistic-cumulative_logit-1.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_ordered_logistic-cumulative_logit-1.json', id='ordered_logistic-cumulative_logit-1'),
    pytest.param('categorical', 'softmax', 2, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_categorical-softmax-2.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_categorical-softmax-2.json', id='categorical-softmax-2'),
    pytest.param('delta', 'identity', 0.35, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_delta-identity-0_35.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_delta-identity-0_35.json', id='delta-identity-0.35'),
    pytest.param('delta', 'identity', 0.4, 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_observation_law_delta-identity-0_4.json', 'likelihood_expressions/native_conditional_law_matches_exact_emission_lowering_revise_law_delta-identity-0_4.json', id='delta-identity-0.4'),
])
def test_native_conditional_law_matches_exact_emission_lowering(family, link, observed, observation_law_payload, revise_law_payload):
    likelihood = LikelihoodSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / revise_law_payload).read_text())
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


@pytest.mark.contract
@pytest.mark.parametrize(('family', 'link', '_observed', 'observation_law_payload', 'with_likelihood_coefficients_payload'), [
    pytest.param('gaussian', 'identity', 0.4, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_gaussian-identity-0_4.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_gaussian-identity-0_4.json', id='gaussian-identity-0.4'),
    pytest.param('student_t', 'identity', 0.4, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_student_t-identity-0_4.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_student_t-identity-0_4.json', id='student_t-identity-0.4'),
    pytest.param('poisson', 'log', 2, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_poisson-log-2.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_poisson-log-2.json', id='poisson-log-2'),
    pytest.param('gamma', 'log', 1.2, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_gamma-log-1_2.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_gamma-log-1_2.json', id='gamma-log-1.2'),
    pytest.param('gamma', 'inverse', 1.2, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_gamma-inverse-1_2.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_gamma-inverse-1_2.json', id='gamma-inverse-1.2'),
    pytest.param('bernoulli', 'logit', 1, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_bernoulli-logit-1.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_bernoulli-logit-1.json', id='bernoulli-logit-1'),
    pytest.param('bernoulli', 'probit', 1, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_bernoulli-probit-1.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_bernoulli-probit-1.json', id='bernoulli-probit-1'),
    pytest.param('negative_binomial', 'log', 3, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_negative_binomial-log-3.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_negative_binomial-log-3.json', id='negative_binomial-log-3'),
    pytest.param('beta', 'logit', 0.4, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_beta-logit-0_4.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_beta-logit-0_4.json', id='beta-logit-0.4'),
    pytest.param('beta', 'probit', 0.4, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_beta-probit-0_4.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_beta-probit-0_4.json', id='beta-probit-0.4'),
    pytest.param('ordered_logistic', 'cumulative_logit', 1, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_ordered_logistic-cumulative_logit-1.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_ordered_logistic-cumulative_logit-1.json', id='ordered_logistic-cumulative_logit-1'),
    pytest.param('categorical', 'softmax', 2, 'likelihood_expressions/authored_laws_preserve_coefficient_identities_observation_law_categorical-softmax-2.json', 'likelihood_expressions/authored_laws_preserve_coefficient_identities_with_likelihood_coefficients_categorical-softmax-2.json', id='categorical-softmax-2'),
])
def test_authored_laws_preserve_coefficient_identities(family, link, _observed, observation_law_payload, with_likelihood_coefficients_payload):
    authored = LikelihoodSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / with_likelihood_coefficients_payload).read_text())
    assert authored.terms.loadings[ConstructId("construct:x")].value == -1
    assert authored.terms.intercept.value == scientific_id("parameter", "baseline")
    assert (authored.terms.family, authored.terms.link) == (family, link)
    assert set(authored.model_dump()) == {"law", "standardized", "reasoning", "sources"}
    assert LikelihoodSpec.model_validate_json(authored.model_dump_json()) == authored


@pytest.mark.contract
def test_completion_binding_equations_and_serialization_follow_the_same_cross_loading():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'likelihood_expressions/completion_binding_equations_and_serialization_follow_the_same_cross_loading_complete_test_model.json').read_text())
    owner, other = model.constructs
    indicator = owner.indicators[0]
    revised = LikelihoodSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'likelihood_expressions/completion_binding_equations_and_serialization_follow_the_same_cross_loading_revise_law.json').read_text())
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            (
                type(owner).model_validate(
                    {
                        **owner.model_dump(),
                        "indicators": (
                            type(indicator).model_validate(
                                {**indicator.model_dump(), "likelihood": revised}
                            ),
                        ),
                    }
                ),
            ),
        )
    )
    model.check_execution()
    np.testing.assert_allclose(numeric.loading_block(model).template, [[1, 0.25], [0, 1]])
    uses = [use for use in iter_coefficient_uses(model) if use.quantity == SiteKind.LOADING]
    cross = next(use for use in uses if use.value == 0.25)
    assert {ref.id for ref in cross.owners} == {indicator.id, other.id}
    equation = observation_equations(model)[indicator.id]
    assert r"\operatorname{Normal}" in equation
    assert r"0.25" in equation
    assert r"\eta_{\text{Y}}(t)" in equation
    assert ModelSpec.model_validate_json(model.model_dump_json()) == model
    renamed = model.revised(
        edges=replace_constructs(
            model.edges, (type(other).model_validate({**other.model_dump(), "name": "Renamed"}),)
        )
    )
    assert r"\eta_{\text{Renamed}}(t)" in observation_equations(renamed)[indicator.id]
    assert {p.id for p in renamed.parameters} == {p.id for p in model.parameters}


@pytest.mark.contract
def test_partial_law_is_explicit_and_unsupported_formulas_fail_before_execution():
    likelihood = LikelihoodSpec(
        law=ObservationLawSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'likelihood_expressions/partial_law_is_explicit_and_unsupported_formulas_fail_before_execution_observation_law.json').read_text()),
        reasoning="Partial",
    )
    assert all(operand.value is None for operand in likelihood.terms.operands)
    with pytest.raises(ValidationError, match="affine"):
        LikelihoodSpec(
            law=ObservationLawSpec(
                distribution="Normal",
                arguments={
                    "loc": function("exp", likelihood.terms.predictor),
                    "scale": coefficient(None, "observation_scale"),
                },
            ),
            reasoning="Unsupported nonlinear observation predictor",
        )
    with pytest.raises(ValidationError, match="exactly"):
        ObservationLawSpec(distribution="Normal", arguments={"rate": likelihood.terms.predictor})
    with pytest.raises(ValidationError, match="non-negative"):
        coefficient(-1, "observation_scale")

    model = make_model(["X", "Y"], [("X", "Y")])
    owner = model.constructs[0]
    indicator = owner.indicators[0]
    partial = LikelihoodSpec(
        law=ObservationLawSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'likelihood_expressions/partial_law_is_explicit_and_unsupported_formulas_fail_before_execution_observation_law_2.json').read_text()),
        reasoning="Partial",
    )

    def with_law(law):
        return model.revised(
            edges=replace_constructs(
                model.edges,
                (
                    type(owner).model_validate(
                        {
                            **owner.model_dump(),
                            "indicators": (
                                type(indicator).model_validate(
                                    {**indicator.model_dump(), "likelihood": law}
                                ),
                            ),
                        }
                    ),
                ),
            )
        )

    unfinished = with_law(partial)
    with pytest.raises(IncompleteModelError):
        unfinished.check_execution()
    assert "?" in observation_equations(unfinished)[indicator.id]
    ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'likelihood_expressions/partial_law_is_explicit_and_unsupported_formulas_fail_before_execution_complete_test_model.json').read_text()).check_execution()
    with pytest.raises(ValidationError, match="unknown constructs"):
        with_law(likelihood)
    wrong_owner = LikelihoodSpec(
        law=ObservationLawSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'likelihood_expressions/partial_law_is_explicit_and_unsupported_formulas_fail_before_execution_observation_law_3.json').read_text()),
        reasoning="Wrong owner",
    )
    with pytest.raises(ValidationError, match="must include its measured construct"):
        with_law(wrong_owner)
