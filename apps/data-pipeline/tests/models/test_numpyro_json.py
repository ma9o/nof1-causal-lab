"""The persistence boundary preserves native laws, dimensions, and scientific evidence."""

import json

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from notebooks.prior_specification_support import parameter_with_prior
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.numpyro_json import NumPyroDistribution
from nof1_causal_lab.prior_distributions import persistence_to_decay
from tests.slot_fixtures import fixture_parameter_id

_ADAPTER = TypeAdapter(NumPyroDistribution)


@pytest.mark.parametrize(
    ("law", "value"),
    [
        (dist.Normal(0.2, 0.7), 0.3),
        (dist.LogNormal(0.1, 0.4), 0.6),
        (dist.TruncatedNormal(0.0, 1.0, low=-0.2, high=1.0), 0.3),
        (dist.Normal(0.0, 1.0).expand((2, 3)), jnp.zeros((2, 3))),
        (dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1), jnp.array([0.2, 0.4])),
        (
            dist.MultivariateNormal(
                jnp.zeros(2), covariance_matrix=jnp.array([[1.0, 0.2], [0.2, 2.0]])
            ),
            jnp.array([0.2, 0.4]),
        ),
        (
            dist.MixtureGeneral(
                dist.Categorical(probs=jnp.array([0.3, 0.7])),
                [dist.Normal(-1.0, 0.5), dist.StudentT(4.0, 1.0, 0.8)],
            ),
            0.3,
        ),
        (persistence_to_decay(dist.Beta(2.0, 3.0), 7.0), 0.2),
    ],
)
def test_native_json_roundtrip_preserves_law_and_dimensions(law, value):
    encoded = _ADAPTER.dump_json(law)
    restored = _ADAPTER.validate_json(encoded)
    assert type(restored) is type(law)
    assert restored.batch_shape == law.batch_shape
    assert restored.event_shape == law.event_shape
    np.testing.assert_allclose(restored.log_prob(value), law.log_prob(value), atol=2e-6)
    assert json.loads(_ADAPTER.dump_json(restored)) == json.loads(encoded)


@pytest.mark.parametrize(
    "payload",
    [
        {"distribution": "NotANumPyroDistribution", "params": {}},
        {"distribution": "Normal", "params": {"loc": 0.0, "scale": -1.0}},
        {
            "distribution": "Normal",
            "params": {"loc": 0.0, "scale": -1.0, "validate_args": False},
        },
        {"distribution": "Normal", "params": {"loc": 0.0, "scale": 1.0, "unknown": 2.0}},
    ],
)
def test_invalid_native_constructors_are_rejected(payload):
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(payload)


def test_parameter_changes_distribution_without_keeping_authoring_history():
    owner = ConstructRef(id="construct:test")
    parameter = ParameterSpec(
        id=fixture_parameter_id(SiteKind.DYNAMICS_WEIGHT, [owner]),
        name="effect",
        description="Effect",
    )
    specified = parameter_with_prior(
        parameter,
        {
            "distribution": "Normal",
            "params": {"mu": 0.4, "sigma": 0.2},
            "reference_interval_days": 7.0,
            "reasoning": "Weekly effect estimate",
            "sources": [
                {
                    "title": "A study",
                    "snippet": "Weekly estimate",
                    "url": "https://example.com/study",
                }
            ],
        },
    )
    restored = ParameterSpec.model_validate_json(specified.model_dump_json())
    assert restored == specified
    assert restored.id == parameter.id
    assert parameter.distribution is None
    assert isinstance(restored.distribution, dist.Normal)
    assert restored.reference_interval_days == 7.0
    revised = restored.model_copy(update={"distribution": dist.Normal(0.3, 0.1)})
    assert revised.id == restored.id
    assert not hasattr(revised, "prior_sources")
    assert not hasattr(revised, "original_proposal")


def test_parameter_tool_boundary_validates_the_reference_interval():
    owner = ConstructRef(id="construct:test")
    parameter = ParameterSpec(
        id=fixture_parameter_id(SiteKind.DYNAMICS_WEIGHT, [owner]),
        name="effect",
        description="Effect",
    )
    with pytest.raises(ValidationError):
        parameter_with_prior(
            parameter,
            {
                "distribution": "Normal",
                "params": {"mu": 0.4, "sigma": 0.2},
                "reference_interval_days": -7.0,
            },
        )


def test_completed_model_requires_a_prior_on_each_parameter():
    from nof1_causal_lab.compilation_errors import IncompleteModelError
    from nof1_causal_lab.models.prior_planning import complete_parameter_priors
    from tests.helpers import complete_test_model, make_model

    science = complete_test_model(make_model(["X"]))
    draft = science.revised(
        parameters=tuple(
            parameter.model_copy(update={"distribution": None}) for parameter in science.parameters
        )
    )
    with pytest.raises(IncompleteModelError, match="prior"):
        draft.require_priors()
    completed = complete_parameter_priors(draft)
    completed.require_priors()
    assert [p.id for p in completed.parameters] == [p.id for p in draft.parameters]
    assert all(p.distribution is not None for p in completed.parameters)
    assert all(p.distribution is None for p in draft.parameters)
