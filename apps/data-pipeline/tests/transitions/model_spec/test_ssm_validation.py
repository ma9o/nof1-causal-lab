"""Pure compilation and materialization from a completed scientific ModelSpec."""

import math
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
import pytest

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.prior import PriorValidationResult
from nof1_causal_lab.flows.transitions.model_spec.assembly import (
    AssemblyValidation,
    materialize_model_spec_result,
    validate_assembly,
)
from nof1_causal_lab.models.model_checks import check_execution
from nof1_causal_lab.models.model_distributions import with_parameter_distributions
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.inputs import compile_priors, compile_ssm_inputs_from_model
from nof1_causal_lab.models.ssm.runtime import build_ssm_model
from nof1_causal_lab.prior_distributions import prior_reference_value
from tests.helpers import complete_test_model, graph_constructs, make_model


@pytest.fixture
def mood_model():
    return complete_test_model(make_model(["mood"]))


def _with_laws(model, laws, intervals=None):
    model = with_parameter_distributions(
        model, {p.id: laws[p.name] for p in model.parameters if p.name in laws}
    )
    return model.revised(
        parameters=tuple(
            parameter.model_copy(
                update={
                    **(
                        {"reference_interval_days": intervals[parameter.name]}
                        if intervals is not None and parameter.name in intervals
                        else {}
                    ),
                }
            )
            for parameter in model.parameters
        )
    )


def _compiled_priors(model):
    _spec, lags = (model, numeric.edge_lag_days(model))
    return compile_priors(model, edge_lag_days=lags)


def _binding(bindings, name):
    return next(item for item in bindings.by_parameter.values() if item.parameter_name == name)


def _reference(priors, bindings, name):
    binding = _binding(bindings, name)
    return float(
        jnp.asarray(prior_reference_value(priors[binding.site_name])).reshape(-1)[
            binding.flat_index
        ]
    )


def _coupled_model(clock="1d"):
    base = make_model(["mood", "stress"], [("stress", "mood")]).revised(measurement_clock=clock)
    return _with_laws(
        complete_test_model(base),
        {
            "rho_mood": dist.Beta(2.0, 2.0),
            "rho_stress": dist.Beta(2.0, 2.0),
            "beta_stress_mood": dist.Normal(0.3, 0.15),
        },
    )


def _warning():
    return PriorValidationResult(
        parameter="beta_stress_sleep",
        is_valid=True,
        code="lagged_response_weak",
        origin="compile",
        severity="warning",
        issue="Median one-lag response is much slower than the nominal lag.",
        suggested_adjustment="Confirm that this slow response is intended.",
    )


def test_build_ssm_model_creates_model(mood_model):
    check_execution(mood_model)
    model = build_ssm_model(
        pl.DataFrame({"mood_obs": [1.0, 2.0], "time": [0.0, 1.0]}), model_spec=mood_model
    )
    assert numeric.n_observations(model.spec) == numeric.n_states(model.spec) == 1
    assert numeric.loading_block(model.spec).template.shape == (1, 1)


def test_materialization_retains_warnings_separately_from_science(mood_model):
    payload = mood_model.model_dump(mode="json")
    warning = _warning()
    validation = AssemblyValidation(model=payload, diagnostics=[warning])
    with (
        patch(
            "nof1_causal_lab.flows.transitions.model_spec.assembly.build_exact_prior_predictive_samples",
            return_value={mood_model.indicators[0].id: [0.1, 0.2]},
        ),
    ):
        model, predictive, diagnostics = materialize_model_spec_result(
            model=payload,
            data_for_model=pl.DataFrame(),
            validation=validation,
        )
    assert model.model_dump(mode="json") == payload
    assert diagnostics == [warning]
    assert predictive.samples == {mood_model.indicators[0].id: [0.1, 0.2]}
    assert set(predictive.model_dump()) == {"samples", "diagnostics"}


def test_assembly_keeps_diagnostics_outside_the_scientific_model(mood_model):
    with (
        patch(
            "nof1_causal_lab.models.model_checks.check_execution",
            return_value=(),
        ) as compile_mock,
        patch(
            "nof1_causal_lab.models.ssm.compile.inputs.compile_ssm_inputs_from_model",
            return_value=(None, None, [_warning()], None, None),
        ) as inputs_mock,
    ):
        validation = validate_assembly(mood_model.model_dump(mode="json"))
    assert compile_mock.call_count == 1
    assert inputs_mock.call_count == 1
    assert validation.compile_ok
    assert validation.compile_diagnostics == [_warning()]


def test_assembly_returns_compile_failure(mood_model):
    with patch(
        "nof1_causal_lab.models.model_checks.check_execution",
        side_effect=ValueError("dimension mismatch"),
    ):
        validation = validate_assembly(mood_model.model_dump(mode="json"))
    assert not validation.compile_ok
    assert validation.compile_error == "dimension mismatch"


def test_beta_persistence_compiles_to_its_exact_pushforward(mood_model):
    model = _with_laws(mood_model, {"rho_mood": dist.Beta(2.0, 2.0)})
    priors, bindings, _ = _compiled_priors(model)
    binding = _binding(bindings, "rho_mood")
    law = priors[binding.site_name]
    decay = 0.7
    rho = math.exp(-decay)
    expected = math.log(6.0) + 2.0 * math.log(rho) + math.log1p(-rho)
    assert float(law.log_prob(decay)) == pytest.approx(expected, abs=2e-6)
    key = jax.random.PRNGKey(6)
    np.testing.assert_array_equal(law.sample(key), -jnp.log(dist.Beta(2.0, 2.0).sample(key)))


def test_prior_compilation_derives_its_native_layout(mood_model):
    priors, bindings, _ = compile_priors(mood_model)
    assert priors
    assert bindings.by_parameter


def test_compile_consumes_validated_model_without_reparsing_or_mutating_it(mood_model):
    before = mood_model.model_dump_json()
    with patch.object(ModelSpec, "model_validate", wraps=ModelSpec.model_validate) as validate:
        compile_ssm_inputs_from_model(mood_model)
    assert validate.call_count == 0
    assert mood_model.model_dump_json() == before


@pytest.mark.parametrize("law", [dist.Uniform(-1, 1), dist.Normal(0.5, 0.1)])
def test_compile_rejects_invalid_persistence_support(mood_model, law):
    with pytest.raises(ValueError, match=r"support within \[0, 1\]"):
        _compiled_priors(_with_laws(mood_model, {"rho_mood": law}))


def test_model_rejects_unknown_scientific_owners(mood_model):
    payload = mood_model.model_dump(mode="json")
    import json

    parameter = next(item for item in mood_model.parameters if item.name == "rho_mood")
    dynamics = graph_constructs(payload)[0]["dynamics"]
    graph_constructs(payload)[0]["dynamics"] = json.loads(
        json.dumps(dynamics).replace(parameter.id, "parameter:" + "0" * 64)
    )
    with pytest.raises(ValueError, match=r"undeclared parameter"):
        ModelSpec.model_validate(payload)


@pytest.mark.parametrize(("field", "value"), [("role", "residual_sd"), ("constraint", "positive")])
def test_parameters_do_not_redeclare_native_support_metadata(mood_model, field, value):
    payload = mood_model.model_dump(mode="json")
    payload["parameters"][0][field] = value
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        ModelSpec.model_validate(payload)


def test_each_construct_gets_its_own_decay_law():
    model = _with_laws(
        _coupled_model(), {"rho_mood": dist.Beta(3, 2), "rho_stress": dist.Beta(2, 3)}
    )
    priors, bindings, _ = _compiled_priors(model)
    assert _reference(priors, bindings, "rho_mood") == pytest.approx(-math.log(0.6), abs=0.01)
    assert _reference(priors, bindings, "rho_stress") == pytest.approx(-math.log(0.4), abs=0.01)


@pytest.mark.parametrize(("clock", "days"), [("1d", 1.0), ("1h", 1.0 / 24)])
def test_scientific_interval_scales_persistence_and_effect(clock, days):
    priors, bindings, _ = _compiled_priors(_coupled_model(clock))
    assert _reference(priors, bindings, "rho_mood") == pytest.approx(
        -math.log(0.5) / days, rel=1e-5
    )
    assert _reference(priors, bindings, "beta_stress_mood") == pytest.approx(0.3 / days, rel=1e-5)


def test_reference_interval_takes_precedence_over_measurement_clock():
    model = _with_laws(_coupled_model("1h"), {}, {"rho_mood": 7.0, "beta_stress_mood": 7.0})
    priors, bindings, _ = _compiled_priors(model)
    assert _reference(priors, bindings, "rho_mood") == pytest.approx(-math.log(0.5) / 7, rel=1e-5)
    assert _reference(priors, bindings, "beta_stress_mood") == pytest.approx(0.3 / 7, rel=1e-5)


def test_dt_ct_warning_uses_full_matrix_logm():
    _, _, diagnostics = _compiled_priors(_coupled_model())
    warning = next(item for item in diagnostics if item.code == "dt_ct_approximation_warning")
    assert "matrix-log mismatch; exact CT coupling" in warning.issue
    assert "0.600 1/day" in warning.issue
    assert "beta/dt value 0.300 1/day" in warning.issue
    assert warning.pathology_certificate.primary_score == pytest.approx(0.5, abs=0.001)


def test_compiler_diagnostics_point_to_the_scientific_parameter():
    model = _coupled_model("1h")
    _, _, diagnostics, _, _ = compile_ssm_inputs_from_model(model)
    warning = next(item for item in diagnostics if item.code == "dt_ct_approximation_warning")
    assert warning.parameter == "linear_edge_weight"
    assert warning.related_parameters == [
        next(p.id for p in model.parameters if p.name == "beta_stress_mood")
    ]


def test_incomplete_model_requires_explicit_dynamics_before_compilation():
    model = make_model(["mood"])
    with pytest.raises(ValueError, match=r"likelihood|dynamics|Mechanisms"):
        check_execution(model)


def test_missing_scientific_quantity_is_not_invented_by_compiler(mood_model):
    with pytest.raises(ValueError, match="undeclared parameter"):
        mood_model.revised(
            distributions={
                k: v
                for k, v in mood_model.distributions.items()
                if k
                not in {
                    p.distribution
                    for p in mood_model.parameters
                    if mood_model.parameter_context(p.id).quantity.value == "diffusion_diag"
                }
            },
            parameters=tuple(
                p
                for p in mood_model.parameters
                if mood_model.parameter_context(p.id).quantity.value != "diffusion_diag"
            ),
        )


def test_missing_prior_is_not_invented_by_compiler(mood_model):
    model = mood_model.revised(
        distributions={},
        parameters=tuple(
            p.model_copy(update={"distribution": None}) for p in mood_model.parameters
        ),
    )
    with pytest.raises(ValueError, match="prior"):
        check_execution(model)
