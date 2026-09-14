"""Default laws follow native support and explicitly owned scientific choices."""

import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.identity import ConstructRef, IndicatorRef
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.models.likelihoods import observation_law
from nof1_causal_lab.models.prior_planning import default_parameter_prior
from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor
from tests.helpers import make_model
from tests.slot_fixtures import fixture_parameter_id, with_likelihood_coefficients


@pytest.mark.parametrize(
    ("quantity", "support", "transform", "polarity", "expected"),
    [
        (
            SiteKind.DYNAMICS_WEIGHT,
            SupportClass.REAL,
            "identity",
            "positive",
            dist.Normal(0.0, 0.5),
        ),
        (
            SiteKind.DYNAMICS_POTENTIAL_QUARTIC,
            SupportClass.POSITIVE,
            "identity",
            "positive",
            dist.HalfNormal(1.0),
        ),
        (
            SiteKind.DYNAMICS_DECAY,
            SupportClass.POSITIVE,
            "dt_persistence_to_ct_decay",
            "positive",
            dist.Beta(2.0, 2.0),
        ),
        (
            SiteKind.DIFFUSION_LOWER,
            SupportClass.CORRELATION,
            "identity",
            "positive",
            dist.Uniform(-1.0, 1.0),
        ),
        (
            SiteKind.DIFFUSION_DIAG,
            SupportClass.POSITIVE,
            "identity",
            "positive",
            dist.HalfNormal(1.0),
        ),
        (
            SiteKind.STATIC_STATE_SD,
            SupportClass.POSITIVE,
            "identity",
            "positive",
            dist.HalfNormal(1.0),
        ),
        (SiteKind.LOADING, SupportClass.REAL, "identity", "positive", dist.Normal(0.5, 0.5)),
        (SiteKind.LOADING, SupportClass.REAL, "identity", "negative", dist.Normal(-0.5, 0.5)),
    ],
)
def test_default_law(quantity, support, transform, polarity, expected):
    model = make_model(["X"])
    construct = model.constructs[0]
    indicator = construct.indicators[0].model_copy(update={"construct_polarity": polarity})
    model = model.revised(
        edges=replace_constructs(
            model.edges, (construct.model_copy(update={"indicators": (indicator,)}),)
        )
    )
    owners = (ConstructRef(id=construct.id),)
    if quantity == SiteKind.LOADING:
        owners = (*owners, IndicatorRef(id=indicator.id))
    parameter = ParameterSpec.model_validate(
        {
            "id": fixture_parameter_id(quantity, owners),
            "name": "authored label",
            "description": "Test quantity",
            "distribution_transform": transform,
        }
    )
    if quantity == SiteKind.LOADING:
        from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec

        indicator = indicator.model_copy(
            update={
                "likelihood": with_likelihood_coefficients(
                    LikelihoodSpec(
                        law=observation_law(construct.id, "gaussian", "identity"),
                        reasoning="Test",
                    ),
                    {"loading": parameter.id},
                )
            }
        )
        model = model.revised(
            edges=replace_constructs(
                model.edges, (construct.model_copy(update={"indicators": (indicator,)}),)
            ),
            parameters=(parameter,),
        )
    site = SiteDescriptor(
        name="native site", shape=(), support=support, assembly_group="test", site_kind=quantity
    )
    result = default_parameter_prior(parameter, model, site)
    assert isinstance(result, dist.Distribution)
    assert type(result) is type(expected)
    for key, value in expected.get_args().items():
        np.testing.assert_array_equal(result.get_args()[key], value)
