"""Retained scientific identity and explicit execution requirements; no numerical runs."""

import json
from pathlib import Path

import pytest

from nof1_causal_lab.artifacts.expressions import (
    hill as expr_hill,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_structure import StructuralCompilationError
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.simulation_checks import _incoming_edge_off_target
from nof1_causal_lab.recipes.construct_authoring import ConstructContribution
from tests.helpers import complete_test_model, make_model


@pytest.fixture(scope="module")
def retained():
    root = Path(__file__).resolve().parents[5]
    return (
        ModelSpec.model_validate_json(
            (root / "data/DEMO/fixture/artifacts/model.json").read_text()
        ),
        json.loads(
            (Path(__file__).resolve().parents[2] / "fixtures/numerical_reference.json").read_text()
        ),
    )


def test_retained_model_exposes_previously_omitted_required_structure(retained):
    model, expected = retained
    before = model.model_dump(mode="json")
    assert set(expected["state_ids"]) < set(model.state_order)
    with pytest.raises(StructuralCompilationError, match="static-target edge"):
        model.check_execution()
    dispositions = model.structural_dispositions
    assert any(item.disposition == "unsupported" for item in dispositions)
    assert model.model_dump(mode="json") == before


def test_retired_execution_arrays_do_not_override_scientific_parameter_identity(retained):
    model, expected = retained
    assert {item["parameter_id"] for item in expected["bindings"]} == {
        p.id for p in model.parameters
    }
    for identity in expected["input_ids"]:
        construct = model.get_construct(identity)
        assert identity in model.state_order
        assert any(
            ind.likelihood is not None and ind.likelihood.law.family == "delta"
            for ind in construct.indicators
        )
        for edge in model.edges:
            if edge.cause.id == identity:
                for parameter in model.parameters_for(edge.id):
                    assert model.parameter_context(parameter.id).quantity.value == "dynamics_weight"


def test_edge_off_targets_every_additive_contribution_without_running_a_simulation():
    model = complete_test_model(make_model(["A", "B"], [("A", "B")]))
    edge = model.edges[0]
    fixed_hill = DynamicsMechanismSpec(
        id="mechanism:fixed-hill-a",
        expression=expr_hill(
            expr_state(edge.cause.id),
            emax=0.4,
            ec50=1,
            n=2,
        ),
    )
    model = model.revised(
        edges=(
            edge.model_copy(
                update={
                    "mechanisms": (
                        *edge.mechanisms,
                        fixed_hill,
                        fixed_hill.model_copy(update={"id": "mechanism:fixed-hill-b"}),
                    )
                }
            ),
        )
    )
    native = model
    target = model.get_construct(edge.effect.id)
    source = model.get_construct(edge.cause.id)
    contribution = ConstructContribution(
        construct=target, edges=model.edges, edge_parents=(source.name,)
    )
    assert numeric.state_names(native) is not None
    off = _incoming_edge_off_target(
        native,
        contribution,
        numeric.state_names(native),
        numeric.state_names(native).index(target.name),
    )
    assert len(off.components) == 3
