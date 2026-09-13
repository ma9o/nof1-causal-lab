"""Tests for explicit compiler-independent prior defaults."""

import numpyro.distributions as dist
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from nof1_causal_lab.artifacts.distribution import CompiledDistribution
from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.statistical_model_spec import (
    ParameterConstraint,
    ParameterRole,
    ParameterSpec,
)
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.models.prior_planning import default_parameter_prior
from nof1_causal_lab.models.ssm.compile.parameter_identity import parameter_identity
from nof1_causal_lab.prior_distributions import serialize_distribution


def _make_param(
    name: str = "beta_x",
    role: ParameterRole = ParameterRole.FIXED_EFFECT,
    constraint: ParameterConstraint = ParameterConstraint.NONE,
) -> ParameterSpec:
    owner = ConstructRef(id="construct:test")
    quantity = SiteKind.DYNAMICS_WEIGHT
    return ParameterSpec(
        id=parameter_identity(quantity, [owner]),
        owners=[owner],
        quantity=quantity,
        name=name,
        role=role,
        constraint=constraint,
        description="test param",
    )


class TestDefaultParameterPrior:
    @pytest.mark.parametrize(
        (
            "role",
            "constraint",
            "expected_distribution",
            "expected_params",
        ),
        [
            (
                ParameterRole.FIXED_EFFECT,
                ParameterConstraint.NONE,
                PriorDistributionFamily.NORMAL,
                {"mu": 0.0, "sigma": 0.5},
            ),
            (
                ParameterRole.FIXED_EFFECT,
                ParameterConstraint.POSITIVE,
                PriorDistributionFamily.HALF_NORMAL,
                {"sigma": 1.0},
            ),
            (
                ParameterRole.FIXED_EFFECT,
                ParameterConstraint.UNIT_INTERVAL,
                PriorDistributionFamily.BETA,
                {"alpha": 2.0, "beta": 2.0},
            ),
            (
                ParameterRole.FIXED_EFFECT,
                ParameterConstraint.CORRELATION,
                PriorDistributionFamily.UNIFORM,
                {"lower": -1.0, "upper": 1.0},
            ),
            (
                ParameterRole.RESIDUAL_SD,
                ParameterConstraint.NONE,
                PriorDistributionFamily.HALF_NORMAL,
                {"sigma": 1.0},
            ),
            (
                ParameterRole.STATIC_STATE_SD,
                ParameterConstraint.NONE,
                PriorDistributionFamily.HALF_NORMAL,
                {"sigma": 1.0},
            ),
            (
                ParameterRole.AR_COEFFICIENT,
                ParameterConstraint.CORRELATION,
                PriorDistributionFamily.BETA,
                {"alpha": 2.0, "beta": 2.0},
            ),
            (
                ParameterRole.LOADING,
                ParameterConstraint.POSITIVE,
                PriorDistributionFamily.NORMAL,
                {"mu": 0.5, "sigma": 0.5},
            ),
            (
                ParameterRole.LOADING,
                ParameterConstraint.NEGATIVE,
                PriorDistributionFamily.NORMAL,
                {"mu": -0.5, "sigma": 0.5},
            ),
        ],
        ids=[
            "unconstrained-normal",
            "positive-half-normal",
            "unit-interval-beta",
            "correlation-uniform",
            "residual-sd-role",
            "static-state-sd-role",
            "ar-role",
            "positive-loading-pooled-family",
            "negative-loading-pooled-family",
        ],
    )
    def test_distribution_selection(
        self,
        role: ParameterRole,
        constraint: ParameterConstraint,
        expected_distribution: PriorDistributionFamily,
        expected_params: dict[str, float],
    ):
        p = _make_param(role=role, constraint=constraint)
        result = serialize_distribution(default_parameter_prior(p))[0]
        assert result.distribution == expected_distribution
        assert result.params == expected_params

    def test_returns_native_distribution(self):
        assert isinstance(default_parameter_prior(_make_param()), dist.Distribution)

    @pytest.mark.parametrize(
        ("contract", "metadata"),
        [
            (CompiledDistribution, {}),
        ],
    )
    def test_prior_parameters_must_match_the_declared_family(self, contract, metadata):
        payload = {**metadata, "distribution": "Normal", "params": {"sigma": 1.0}}
        with pytest.raises(ValidationError, match="Normal requires exactly"):
            contract.model_validate(payload)
        validator = Draft202012Validator(contract.model_json_schema())
        assert not validator.is_valid(payload)
        payload["params"] = {"mu": 0.0, "sigma": 1.0}
        validated = contract.model_validate(payload)
        validator.validate(validated.model_dump(mode="json"))
        payload["params"]["unknown"] = 2.0
        assert not validator.is_valid(payload)
        with pytest.raises(ValidationError, match="Normal requires exactly"):
            contract.model_validate(payload)
