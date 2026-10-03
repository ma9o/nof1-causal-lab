"""The persistence boundary preserves native laws, dimensions, and scientific evidence."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter_spec import PersistenceTransformSpec
from nof1_causal_lab.models.model_parameters import require_priors
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.numpyro_json import NumPyroDistribution
from nof1_causal_lab.prior_distributions import persistence_to_decay

_ADAPTER = TypeAdapter(NumPyroDistribution)


@pytest.mark.inference(concern="sampling")
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


@pytest.mark.contract
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


@pytest.mark.contract
def test_parameter_changes_distribution_without_keeping_authoring_history():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_model.json"
        ).read_text()
    )
    parameter = model.parameters[0]
    specified = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/models"
            / "numpyro_json/parameter_changes_distribution_without_keeping_authoring_history_model_with_prior_payloads.json"
        ).read_text()
    )
    restored = type(model).model_validate_json(specified.model_dump_json())
    assert restored == specified
    assert restored.parameter(parameter.id).id == parameter.id
    assert isinstance(restored.distribution_for(parameter.id), dist.Normal)
    transform = restored.parameter(parameter.id).transform
    assert isinstance(transform, PersistenceTransformSpec)
    assert transform.interval_days == 7.0
    revised = restored.revised(
        distributions={
            **restored.distributions,
            restored.parameter(parameter.id).distribution: dist.Normal(0.3, 0.1),
        }
    )
    assert (
        revised.parameter(parameter.id).distribution
        == restored.parameter(parameter.id).distribution
    )
    before_law = restored.distribution_for(parameter.id)
    after_law = revised.distribution_for(parameter.id)
    assert isinstance(before_law, dist.Normal)
    assert isinstance(after_law, dist.Normal)
    assert float(before_law.loc) == pytest.approx(0.4)
    assert float(after_law.loc) == pytest.approx(0.3)
    assert not hasattr(revised.parameter(parameter.id), "prior_sources")
    assert not hasattr(revised.parameter(parameter.id), "original_proposal")
    with pytest.raises(ValidationError):
        parameter.revised(distribution=dist.Normal(0, 1))


@pytest.mark.contract
def test_parameter_tool_boundary_validates_the_reference_interval():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_model.json"
        ).read_text()
    )
    with pytest.raises(ValidationError):
        model.parameters[0].revised(
            transform={"kind": "dt_persistence_to_ct_decay", "interval_days": -7.0},
        )


@pytest.mark.contract
def test_completed_model_requires_a_prior_on_each_parameter():

    from nof1_causal_lab.compilation_errors import IncompleteModelError

    science = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_model.json"
        ).read_text()
    )
    draft = science.revised(
        distributions={},
        parameters=tuple(parameter.revised(distribution=None) for parameter in science.parameters),
    )
    with pytest.raises(IncompleteModelError, match="prior"):
        require_priors(StructuralSelection(draft, None))
    completed = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_model.json"
        ).read_text()
    )
    require_priors(StructuralSelection(completed, None))
    assert [p.id for p in completed.parameters] == [p.id for p in draft.parameters]
    assert all(p.distribution is not None for p in completed.parameters)
    assert all(p.distribution is None for p in draft.parameters)


@pytest.mark.contract
def test_law_memberships_reject_dangling_unused_and_accidentally_shared_scalar_laws():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_model.json"
        ).read_text()
    )
    first, second = model.parameters[:2]
    assert first.distribution != second.distribution
    with pytest.raises(ValidationError, match="every reference must exist"):
        model.revised(distributions={})
    with pytest.raises(ValidationError, match="must be referenced"):
        model.revised(
            distributions={**model.distributions, "distribution:unused": dist.Normal(0, 1)}
        )
    with pytest.raises(ValidationError, match="exactly one parameter"):
        model.revised(
            parameters=tuple(
                p.revised(distribution=first.distribution) if p.id == second.id else p
                for p in model.parameters
            ),
            distributions={
                k: v for k, v in model.distributions.items() if k != second.distribution
            },
        )
