"""Graph comparisons isolate topology from other scientific definition changes."""

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from pathlib import Path

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.actions.revisions import (
    compare_model_definitions,
    compare_model_graph,
    compare_parameters,
)
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import coefficient, state
from nof1_causal_lab.models.model_structure import model_graph_entities
from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
from tests.helpers import make_model

pytestmark = pytest.mark.contract


def test_complete_definition_diff_includes_laws_and_question_without_list_order_noise():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'model_comparison/complete_definition_diff_includes_laws_and_question_without_list_order_noise_complete_test_model.json').read_text())
    parameter = model.parameters[0]
    revised = model.revised(
        question="A revised scientific question",
        distributions={**model.distributions, parameter.distribution: dist.Normal(2.0, 1.0)},
    )
    changes = compare_model_definitions(model, revised)
    assert any(item.path == "/question" and item.after == revised.question for item in changes)
    assert any(
        item.path.startswith(f"/distributions/{parameter.distribution}/") for item in changes
    )
    reordered = model.revised(
        edges=tuple(reversed(model.edges)), parameters=tuple(reversed(model.parameters))
    )
    assert compare_model_definitions(model, reordered) == []
    graph = compare_model_graph(model, reordered)
    assert all(item.change == "unchanged" for item in (*graph.constructs, *graph.edges))


def test_parameter_decisions_and_law_changes_leave_topology_unchanged():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'model_comparison/parameter_decisions_and_law_changes_leave_topology_unchanged_complete_test_model.json').read_text())
    edge = model.edges[0]
    parameter = model.parameters_for(edge.id)[0]
    pinned = model.revised(
        edges=tuple(
            type(item).model_validate(
                {
                    **item.model_dump(),
                    "mechanisms": tuple(
                        type(mechanism).model_validate(
                            {
                                **mechanism.model_dump(),
                                "expression": coefficient(0.0, "weight") * state(edge.cause.id),
                            }
                        )
                        for mechanism in item.mechanisms
                    ),
                }
            )
            if item.id == edge.id
            else item
            for item in model.edges
        ),
        parameters=tuple(item for item in model.parameters if item.id != parameter.id),
        distributions={
            key: law for key, law in model.distributions.items() if key != parameter.distribution
        },
    )
    for before, after, decision in ((model, pinned, "removed"), (pinned, model, "added")):
        changes = compare_parameters(before, after)
        assert [(item.parameter_id, item.change) for item in changes] == [(parameter.id, decision)]
        graph = compare_model_graph(before, after)
        assert all(item.change == "unchanged" for item in (*graph.constructs, *graph.edges))

    # The parameter and law ID can stay the same while the native law changes.
    revised = model.revised(
        distributions={**model.distributions, parameter.distribution: dist.Normal(2.0, 1.0)}
    )
    changes = compare_parameters(model, revised)
    assert [(item.parameter_id, item.change) for item in changes] == [(parameter.id, "revised")]
    graph = compare_model_graph(model, revised)
    assert all(item.change == "unchanged" for item in (*graph.constructs, *graph.edges))


def test_fitted_state_laws_and_time_points_leave_topology_unchanged():
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[1] / "fixtures/models" / 'model_comparison/fitted_state_laws_and_time_points_leave_topology_unchanged_complete_test_model.json').read_text())
    layout = JointLawLayout.from_bindings(
        (), parameters=(), constructs=model.state_order, time_points=(0.0, 1.0)
    )
    identity = layout.distribution_id
    fitted = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                type(item).model_validate({**item.model_dump(), "distribution": identity})
                for item in model.constructs
            ),
        ),
        distributions={
            **model.distributions,
            identity: dist.Delta(jnp.zeros(layout.width), event_dim=1),
        },
        time_points=layout.time_points,
    )
    assert compare_model_definitions(model, fitted)
    for before, after in ((model, fitted), (fitted, model)):
        graph = compare_model_graph(before, after)
        assert all(item.change == "unchanged" for item in (*graph.constructs, *graph.edges))


def test_graph_additions_and_removals_ignore_entity_attribute_changes():
    before = make_model(["X", "Y"], [("X", "Y")])
    after = make_model(["X", "Y", "Z"], [("X", "Y"), ("Z", "Y")])
    x = after.constructs[0]
    renamed = type(x).model_validate(
        {
            **x.model_dump(),
            "name": "Renamed X",
            "description": "Updated measurement",
            "indicators": (
                type(x.indicators[0]).model_validate(
                    {**x.indicators[0].model_dump(), "name": "Renamed observation"}
                ),
            ),
        }
    )
    after = after.revised(
        default_outcome=after.constructs[1].id,
        edges=tuple(
            type(edge).model_validate({**edge.model_dump(), "description": "Updated justification"})
            for edge in replace_constructs(after.edges, (renamed,))
        ),
    )
    graph = compare_model_graph(before, after)
    assert {
        item.after.name: (item.before.name if item.before else None, item.change)
        for item in graph.constructs
        if item.after
    } == {"Renamed X": ("X", "unchanged"), "Y": ("Y", "unchanged"), "Z": (None, "added")}
    assert (
        next(item for item in graph.edges if item.edge_id == before.edges[0].id).change
        == "unchanged"
    )
    assert next(item for item in graph.edges if item.edge_id == after.edges[1].id).change == "added"
    reverse = compare_model_graph(after, before)
    assert (
        next(item for item in reverse.edges if item.edge_id == after.edges[1].id).change
        == "removed"
    )
    assert (
        next(item for item in reverse.constructs if item.before and item.before.name == "Z").after
        is None
    )
    unchanged = compare_model_graph(after, after)
    assert all(item.change == "unchanged" for item in (*unchanged.constructs, *unchanged.edges))


def test_endpoint_and_time_slice_changes_revise_graph_topology():
    model = make_model(["X", "Y"], [("X", "Y")]).revised(measurement_clock=None)
    edge = model.edges[0]
    reversed_edge = model.revised(
        edges=(
            type(edge).model_validate(
                {**edge.model_dump(), "cause": edge.effect, "effect": edge.cause}
            ),
        )
    )
    static_cause = model.revised(
        edges=replace_constructs(
            model.edges,
            (
                type(edge.cause).model_validate(
                    {**edge.cause.model_dump(), "temporal_status": "time_invariant"}
                ),
            ),
        )
    )
    for revised in (reversed_edge, static_cause):
        for before, after in ((model, revised), (revised, model)):
            graph = compare_model_graph(before, after)
            assert [(item.edge_id, item.change) for item in graph.edges] == [(edge.id, "revised")]
            changed_constructs = [
                item.construct_id for item in graph.constructs if item.change != "unchanged"
            ]
            assert changed_constructs == ([edge.cause.id] if revised is static_cause else [])


def test_execution_exclusions_use_the_same_graph_comparison_in_both_directions():
    model = make_model(
        ["U", "X", "Y", "V", "A", "B"],
        [("U", "X"), ("X", "Y"), ("V", "Y"), ("A", "B"), ("B", "V")],
    )
    constructs = {item.name: item for item in model.constructs}
    measured = model.revised(
        default_outcome=constructs["Y"].id,
        edges=replace_constructs(
            model.edges,
            (
                type(constructs["U"]).model_validate(
                    {**constructs["U"].model_dump(), "role": "exogenous", "indicators": ()}
                ),
                type(constructs["V"]).model_validate(
                    {**constructs["V"].model_dump(), "indicators": ()}
                ),
            ),
        ),
    )
    structural = measured.revised(measurement_clock=None)
    retained, edges = model_graph_entities(measured)
    assert {item.name for item in retained} == {"X", "Y"}
    assert [(item.cause.name, item.effect.name) for item in edges] == [("X", "Y")]
    assert set(measured.state_order) == {constructs["X"].id, constructs["Y"].id}
    assert len(model_graph_entities(structural)[0]) == 6
    forward = compare_model_graph(structural, measured)
    excluded = {
        item.before.name: item.after_disposition
        for item in forward.constructs
        if item.before and item.change == "removed"
    }
    assert set(excluded) == {"U", "V", "A", "B"}
    marginalized, unsupported = excluded["U"], excluded["V"]
    assert marginalized is not None
    assert marginalized.disposition == "marginalized"
    assert unsupported is not None
    assert unsupported.disposition == "unsupported"
    for name in ("A", "B"):
        disposition = excluded[name]
        assert disposition is not None
        assert disposition.disposition == "identification_only"
        assert "Disconnected from outcome 'Y'" in disposition.reason
    reverse = compare_model_graph(measured, structural)
    assert {
        item.after.name for item in reverse.constructs if item.after and item.change == "added"
    } == {"U", "V", "A", "B"}
    assert len([item for item in forward.edges if item.change == "removed"]) == 4
    assert len([item for item in reverse.edges if item.change == "added"]) == 4
    same = compare_model_graph(measured, measured)
    assert {item.after.name for item in same.constructs if item.after} == {"X", "Y"}
    assert all(item.change == "unchanged" for item in (*same.constructs, *same.edges))
