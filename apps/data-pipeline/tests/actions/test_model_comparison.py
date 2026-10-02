"""Graph comparisons isolate topology from other scientific definition changes."""

from pathlib import Path

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import coefficient, state
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_structure import (
    compare_model_graph,
    compare_parameters,
    model_graph_entities,
    selected_state_ids,
)
from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
from tests.helpers import make_model

pytestmark = pytest.mark.contract


def test_parameter_decisions_and_law_changes_leave_topology_unchanged():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/models"
            / "model_comparison/y_z_model.json"
        ).read_text()
    )
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
        assert [(item.parameter_id, item.change.kind) for item in changes] == [
            (parameter.id, decision)
        ]
        graph = compare_model_graph(before, after)
        assert all(item.change.kind == "unchanged" for item in (*graph.constructs, *graph.edges))

    # The parameter and law ID can stay the same while the native law changes.
    revised = model.revised(
        distributions={**model.distributions, parameter.distribution: dist.Normal(2.0, 1.0)}
    )
    changes = compare_parameters(model, revised)
    assert [(item.parameter_id, item.change.kind) for item in changes] == [
        (parameter.id, "revised")
    ]
    graph = compare_model_graph(model, revised)
    assert all(item.change.kind == "unchanged" for item in (*graph.constructs, *graph.edges))


def test_fitted_state_laws_and_time_points_leave_topology_unchanged():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_y_model.json"
        ).read_text()
    )
    layout = JointLawLayout.from_bindings(
        (), parameters=(), constructs=selected_state_ids(model), time_points=(0.0, 1.0)
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
        time_points=layout.time_points,
    )
    for before, after in ((model, fitted), (fitted, model)):
        graph = compare_model_graph(before, after)
        assert all(item.change.kind == "unchanged" for item in (*graph.constructs, *graph.edges))


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
        default_outcome=after.constructs[1].id,
        edges=tuple(
            edge.revised(description="Updated justification")
            for edge in replace_constructs(after.edges, (renamed,))
        ),
    )
    graph = compare_model_graph(before, after)
    assert {
        item.change.after.name: (
            None if item.change.kind == "added" else item.change.before.name,
            item.change.kind,
        )
        for item in graph.constructs
        if item.change.kind != "removed"
    } == {"Renamed X": ("X", "unchanged"), "Y": ("Y", "unchanged"), "Z": (None, "added")}
    assert (
        next(item for item in graph.edges if item.edge_id == before.edges[0].id).change.kind
        == "unchanged"
    )
    assert (
        next(item for item in graph.edges if item.edge_id == after.edges[1].id).change.kind
        == "added"
    )
    reverse = compare_model_graph(after, before)
    assert (
        next(item for item in reverse.edges if item.edge_id == after.edges[1].id).change.kind
        == "removed"
    )
    assert (
        next(
            item for item in reverse.constructs if item.construct_id == after.constructs[2].id
        ).change.kind
        == "removed"
    )
    unchanged = compare_model_graph(after, after)
    assert all(
        item.change.kind == "unchanged" for item in (*unchanged.constructs, *unchanged.edges)
    )


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
            graph = compare_model_graph(before, after)
            assert [(item.edge_id, item.change.kind) for item in graph.edges] == [
                (edge.id, "revised")
            ]
            changed_constructs = [
                item.construct_id for item in graph.constructs if item.change.kind != "unchanged"
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
                constructs["U"].revised(role="endogenous", indicators=()),
                constructs["V"].revised(indicators=()),
            ),
        ),
    )
    structural = measured.revised(measurement_clock=None)
    retained, edges = model_graph_entities(measured)
    assert {item.name for item in retained} == {"X", "Y"}
    assert [(item.cause.name, item.effect.name) for item in edges] == [("X", "Y")]
    assert set(selected_state_ids(measured)) == {constructs["X"].id, constructs["Y"].id}
    assert len(model_graph_entities(structural)[0]) == 6
    forward = compare_model_graph(structural, measured)
    excluded = {
        item.change.before.name: item.after_disposition
        for item in forward.constructs
        if item.change.kind == "removed"
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
        item.change.after.name for item in reverse.constructs if item.change.kind == "added"
    } == {"U", "V", "A", "B"}
    assert len([item for item in forward.edges if item.change.kind == "removed"]) == 4
    assert len([item for item in reverse.edges if item.change.kind == "added"]) == 4
    same = compare_model_graph(measured, measured)
    assert {
        item.change.after.name for item in same.constructs if item.change.kind != "removed"
    } == {"X", "Y"}
    assert all(item.change.kind == "unchanged" for item in (*same.constructs, *same.edges))
