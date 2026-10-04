"""Graph comparisons isolate topology from other scientific definition changes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import coefficient, state
from nof1_causal_lab.models.model_structure import (
    StructuralSelection,
    compare_model_graph,
    compare_parameters,
    model_graph_entities,
    selected_state_ids,
)
from nof1_causal_lab.models.ssm.compile.bindings import joint_law_layout
from nof1_causal_lab.study.view_models import Added, Removed
from tests.helpers import make_model
from tests.model_fixtures import load_model_fixture, x_y_model


def _y_z_model() -> ModelSpec:
    return load_model_fixture("model_comparison/y_z_model.json")


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


pytestmark = pytest.mark.contract


def _whole(model: ModelSpec) -> StructuralSelection:
    return StructuralSelection(model, None)


def test_parameter_decisions_and_law_changes_leave_topology_unchanged():
    model = _y_z_model()
    edge = model.edges[0]
    parameter = model.parameters_for(edge.id)[0]
    pinned = model.revised(
        edges=tuple(
            item.revised(
                mechanisms=tuple(
                    mechanism.revised(expression=coefficient(0.0, "weight") * state(edge.cause.id))
                    for mechanism in item.mechanisms
                )
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
        assert [
            ((item.before.id if isinstance(item, Removed) else item.after.id), item.kind)
            for item in changes
        ] == [(parameter.id, decision)]
        graph = compare_model_graph(_whole(before), _whole(after))
        assert all(item.kind == "unchanged" for item in (*graph[0], *graph[1]))

    # The parameter and law ID can stay the same while the native law changes.
    revised = model.revised(
        distributions={**model.distributions, parameter.distribution: dist.Normal(2.0, 1.0)}
    )
    changes = compare_parameters(model, revised)
    assert [
        ((item.before.id if isinstance(item, Removed) else item.after.id), item.kind)
        for item in changes
    ] == [(parameter.id, "revised")]
    graph = compare_model_graph(_whole(model), _whole(revised))
    assert all(item.kind == "unchanged" for item in (*graph[0], *graph[1]))


def test_fitted_state_laws_and_time_points_leave_topology_unchanged():
    model = x_y_model()
    layout = joint_law_layout(
        (),
        parameters=(),
        constructs=selected_state_ids(_whole(model)),
        time_points=(0.0, 1.0),
        construct_labels={item.id: item.name for item in model.constructs},
    )
    identity = layout.distribution_id
    fitted = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(item.revised(distribution=identity) for item in model.constructs),
        ),
        distributions={
            **model.distributions,
            identity: dist.Delta(jnp.zeros(layout.width), event_dim=1),
        },
        law_layouts={identity: layout},
    )
    for before, after in ((model, fitted), (fitted, model)):
        graph = compare_model_graph(_whole(before), _whole(after))
        assert all(item.kind == "unchanged" for item in (*graph[0], *graph[1]))


def test_graph_additions_and_removals_ignore_entity_attribute_changes():
    before = make_model(["X", "Y"], [("X", "Y")])
    after = make_model(["X", "Y", "Z"], [("X", "Y"), ("Z", "Y")])
    x = after.constructs[0]
    renamed = x.revised(
        name="Renamed X",
        description="Updated measurement",
        indicators=(
            x.indicators[0].revised(
                observation=x.indicators[0].observation.revised(name="Renamed observation")
            ),
        ),
    )
    after = after.revised(
        edges=tuple(
            edge.revised(description="Updated justification")
            for edge in replace_constructs(after.edges, (renamed,))
        ),
    )
    creation = compare_model_graph(None, _whole(after))
    removal = compare_model_graph(_whole(after), None)
    assert all(item.kind == "added" for item in (*creation[0], *creation[1]))
    assert all(item.kind == "removed" for item in (*removal[0], *removal[1]))
    assert all(item.kind == "added" for item in compare_parameters(None, after))
    assert all(item.kind == "removed" for item in compare_parameters(after, None))
    assert compare_model_graph(None, None) == ((), ())
    assert compare_parameters(None, None) == ()
    scoped = StructuralSelection(after, after.constructs[1].id)
    graph = compare_model_graph(_whole(before), scoped)
    assert {
        after.get_construct(item.after.id).name: (
            None if isinstance(item, Added) else before.get_construct(item.before.id).name,
            item.kind,
        )
        for item in graph[0]
        if not isinstance(item, Removed)
    } == {"Renamed X": ("X", "unchanged"), "Y": ("Y", "unchanged"), "Z": (None, "added")}
    assert (
        next(
            item
            for item in graph[1]
            if (item.before.id if isinstance(item, Removed) else item.after.id)
            == before.edges[0].id
        ).kind
        == "unchanged"
    )
    assert (
        next(
            item
            for item in graph[1]
            if (item.before.id if isinstance(item, Removed) else item.after.id) == after.edges[1].id
        ).kind
        == "added"
    )
    reverse = compare_model_graph(scoped, _whole(before))
    assert (
        next(
            item
            for item in reverse[1]
            if (item.before.id if isinstance(item, Removed) else item.after.id) == after.edges[1].id
        ).kind
        == "removed"
    )
    assert (
        next(
            item
            for item in reverse[0]
            if (item.before.id if isinstance(item, Removed) else item.after.id)
            == after.constructs[2].id
        ).kind
        == "removed"
    )
    unchanged = compare_model_graph(scoped, scoped)
    assert all(item.kind == "unchanged" for item in (*unchanged[0], *unchanged[1]))


def test_endpoint_and_time_slice_changes_revise_graph_topology():
    model = make_model(["X", "Y"], [("X", "Y")]).revised(measurement_clock=None)
    edge = model.edges[0]
    reversed_edge = model.revised(edges=(edge.revised(cause=edge.effect, effect=edge.cause),))
    static_cause = model.revised(
        edges=replace_constructs(
            model.edges,
            (edge.cause.revised(temporal_status="time_invariant"),),
        )
    )
    for revised in (reversed_edge, static_cause):
        for before, after in ((model, revised), (revised, model)):
            graph = compare_model_graph(_whole(before), _whole(after))
            assert [
                ((item.before.id if isinstance(item, Removed) else item.after.id), item.kind)
                for item in graph[1]
            ] == [(edge.id, "revised")]
            changed_constructs = [
                (item.before.id if isinstance(item, Removed) else item.after.id)
                for item in graph[0]
                if item.kind != "unchanged"
            ]
            assert changed_constructs == ([edge.cause.id] if revised is static_cause else [])


def test_execution_exclusions_use_the_same_graph_comparison_in_both_directions():
    model = make_model(
        ["U", "X", "Y", "V", "A", "B"],
        [("U", "X"), ("X", "Y"), ("V", "Y"), ("A", "B"), ("B", "V")],
    )
    constructs = {item.name: item for item in model.constructs}
    measured = model.revised(
        edges=replace_constructs(
            model.edges,
            (
                constructs["U"].revised(role="endogenous", indicators=()),
                constructs["V"].revised(indicators=()),
            ),
        ),
    )
    outcome = constructs["Y"].id
    structural = StructuralSelection(measured.revised(measurement_clock=None), outcome)
    measured = StructuralSelection(measured, outcome)
    retained, edges = model_graph_entities(measured)
    assert {item.name for item in retained} == {"X", "Y"}
    assert [(item.cause.name, item.effect.name) for item in edges] == [("X", "Y")]
    assert set(selected_state_ids(measured)) == {constructs["X"].id, constructs["Y"].id}
    assert len(model_graph_entities(structural)[0]) == 6
    forward = compare_model_graph(structural, measured)
    excluded = {
        model.get_construct(item.before.id).name: next(
            (
                disposition
                for disposition in measured.structural_dispositions
                if disposition.target.id == item.before.id
            ),
            None,
        )
        for item in forward[0]
        if isinstance(item, Removed)
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
        model.get_construct(item.after.id).name for item in reverse[0] if isinstance(item, Added)
    } == {"U", "V", "A", "B"}
    assert len([item for item in forward[1] if isinstance(item, Removed)]) == 4
    assert len([item for item in reverse[1] if isinstance(item, Added)]) == 4
    same = compare_model_graph(measured, measured)
    assert {
        model.get_construct(item.after.id).name for item in same[0] if not isinstance(item, Removed)
    } == {"X", "Y"}
    assert all(item.kind == "unchanged" for item in (*same[0], *same[1]))
