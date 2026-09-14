"""Scientific entities gain detail through stable component references."""

import numpyro.distributions as dist
import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.construct import (
    CausalEdgeSpec,
    endpoint_validation_scope,
    replace_constructs,
    serialize_edge_references,
)
from nof1_causal_lab.artifacts.expressions import hill, linear_effect, state
from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId, scientific_id
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.likelihoods import observation_law
from tests.helpers import graph_constructs


def _model():
    weight = {
        "id": scientific_id("parameter", "weight"),
        "name": "effect",
        "description": "Linear effect",
    }
    emax = {
        "id": scientific_id("parameter", "emax"),
        "name": "emax",
        "description": "Maximum Hill effect",
    }
    return ModelSpec.model_validate(
        {
            "edges": [
                {
                    "id": "edge:xy",
                    "cause": {
                        "id": "construct:x",
                        "name": "X",
                        "description": "X",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                    },
                    "effect": {
                        "id": "construct:y",
                        "name": "Y",
                        "description": "Y",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                        "indicators": [
                            {
                                "id": "indicator:y",
                                "name": "measured_y",
                                "how_to_measure": "Mean value",
                                "construct_polarity": "positive",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            }
                        ],
                    },
                    "description": "X influences Y",
                    "mechanisms": [
                        DynamicsMechanismSpec(
                            id="mechanism:linear",
                            expression=linear_effect(ConstructId("construct:x"), weight["id"]),
                        ).model_dump(mode="json"),
                        DynamicsMechanismSpec(
                            id="mechanism:hill",
                            expression=hill(
                                state(ConstructId("construct:x")),
                                emax=emax["id"],
                                ec50=1,
                                n=2,
                            ),
                        ).model_dump(mode="json"),
                    ],
                }
            ],
            "parameters": [weight, emax],
        }
    )


def test_entities_gain_detail_with_one_owner_and_native_prior():
    before = _model()
    payload = before.model_dump(mode="python")
    graph_constructs(payload)[1]["indicators"][0]["likelihood"] = {
        "law": observation_law(ConstructId("construct:y"), "gaussian", "identity").model_dump(
            mode="json"
        ),
        "reasoning": "Continuous measurement",
    }
    from nof1_causal_lab.models.model_distributions import with_parameter_distributions

    after = with_parameter_distributions(
        ModelSpec.model_validate(payload), {before.parameters[0].id: dist.Normal(0.0, 0.1)}
    )
    assert after.indicator_owner(IndicatorId("indicator:y")) is after.get_construct(
        ConstructId("construct:y")
    )
    assert after.indicator(IndicatorId("indicator:y")).id == before.indicator("indicator:y").id
    assert after.indicator(IndicatorId("indicator:y")).likelihood is not None
    assert before.indicator("indicator:y").likelihood is None
    assert after.parameters[0].id == before.parameters[0].id
    assert isinstance(after.distribution_for(after.parameters[0].id), dist.Normal)
    assert before.parameters[0].distribution is None
    encoded = after.model_dump(mode="json")
    assert "construct_id" not in graph_constructs(encoded)[1]["indicators"][0]
    assert "indicator_id" not in graph_constructs(encoded)[1]["indicators"][0]["likelihood"]
    assert all(
        set(term) == {"id", "kind", "expression"} for term in encoded["edges"][0]["mechanisms"]
    )
    assert all("edge_id" not in term for term in encoded["edges"][0]["mechanisms"])
    assert after.edges[0].id == before.edges[0].id
    assert after.edges[0].mechanisms == before.edges[0].mechanisms
    assert "policies" not in encoded
    assert {"quantity", "owners", "elements"}.isdisjoint(encoded["parameters"][0])


def test_partial_model_is_valid_but_operation_requirements_are_explicit():
    initial = ModelSpec(question="  Does X change Y?  ")
    assert initial.question == "Does X change Y?"
    assert initial.edges == initial.constructs == ()
    assert ModelSpec.model_validate({}).edges == ()
    with pytest.raises(ValueError, match="clock"):
        initial.check_execution()
    partial = _model()
    with pytest.raises(ValueError, match="clock"):
        partial.require_measurements()
    with pytest.raises(ValueError, match="prior"):
        partial.require_priors()
    complete_measurements = partial.revised(measurement_clock="1d")
    complete_measurements.require_measurements()
    assert partial.measurement_clock is None


@pytest.mark.parametrize(
    "change",
    [
        "delete_owner",
        "coefficient_owner",
        "dangling_parameter",
        "duplicate_indicator",
        "incompatible_likelihood",
    ],
)
def test_inconsistent_enrichment_is_rejected(change):
    payload = _model().model_dump(mode="json")
    if change == "delete_owner":
        payload["edges"][0]["cause"] = {"kind": "construct", "id": "construct:x"}
    elif change == "coefficient_owner":
        payload["parameters"][0]["owners"] = [{"kind": "construct", "id": "construct:y"}]
    elif change == "dangling_parameter":
        payload["parameters"].pop(0)
    elif change == "duplicate_indicator":
        graph_constructs(payload)[0]["indicators"] = graph_constructs(payload)[1]["indicators"]
    else:
        graph_constructs(payload)[1]["indicators"][0]["likelihood"] = {
            "law": observation_law(ConstructId("construct:y"), "bernoulli", "logit"),
            "reasoning": "Wrong type",
        }
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(payload)


def test_shared_endpoints_round_trip_once_and_resolve_forward_references():
    from tests.helpers import make_model

    model = make_model(["A", "B", "Y"], [("A", "Y"), ("B", "Y")])
    assert model.edges[0].effect is model.edges[1].effect
    payload = model.model_dump(mode="json")
    assert "constructs" not in payload
    assert "constructs" not in ModelSpec.model_json_schema()["properties"]
    assert payload["edges"][1]["effect"] == {"kind": "construct", "id": model.edges[0].effect.id}
    payload["edges"].reverse()
    restored = ModelSpec.model_validate(payload)
    assert restored.edges[0].effect is restored.edges[1].effect
    assert ModelSpec.model_validate_json(model.model_dump_json()) == model
    changed = model.revised(
        edges=replace_constructs(
            model.edges, [model.edges[0].effect.model_copy(update={"name": "Renamed Y"})]
        )
    )
    assert changed.edges[0].effect is changed.edges[1].effect
    assert changed.edges[1].effect.name == "Renamed Y"
    assert model.edges[1].effect.name == "Y"
    # Incremental submissions carry edge references and one separately authored endpoint.
    references = serialize_edge_references(model.edges)
    with endpoint_validation_scope(references, endpoints=changed.constructs):
        submitted = TypeAdapter(tuple[CausalEdgeSpec, ...]).validate_python(references)
    assert submitted[0].effect is changed.edges[0].effect
    assert submitted[1].effect is submitted[0].effect


def test_endpoint_identity_rejects_conflicting_definitions():
    from tests.helpers import make_model

    model = make_model(["A", "B", "Y"], [("A", "Y"), ("B", "Y")])
    payload = model.model_dump(mode="json")
    payload["edges"][1]["effect"] = {**payload["edges"][0]["effect"], "name": "Conflicting Y"}
    with pytest.raises(ValidationError, match="Conflicting definitions"):
        ModelSpec.model_validate(payload)


def test_graph_membership_follows_edges_and_revisions_preserve_connectivity():
    from tests.helpers import make_model

    model = make_model(["A", "B", "C", "D"], [("A", "B"), ("B", "C"), ("C", "D")])
    with pytest.raises(ValidationError, match="connected causal graph"):
        model.revised(edges=(model.edges[0], model.edges[2]))
    assert model.revised(edges=()).constructs == ()
    shortened = model.revised(edges=model.edges[:-1])
    assert [construct.name for construct in shortened.constructs] == ["A", "B", "C"]
    assert [construct.name for construct in model.constructs] == ["A", "B", "C", "D"]
