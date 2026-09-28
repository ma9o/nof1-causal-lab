"""Graph comparisons localize scientific changes without frontend inference."""

import numpyro.distributions as dist
import pytest

from nof1_causal_lab.actions.revisions import (
    compare_model_definitions,
    compare_model_graph,
    compare_parameters,
)
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.models.model_parameters import referenced_parameter_ids
from nof1_causal_lab.models.model_structure import model_graph_entities
from tests.helpers import complete_test_model, make_model

pytestmark = pytest.mark.contract


def test_complete_definition_diff_includes_laws_and_question_without_list_order_noise():
    model = complete_test_model(make_model(["X", "Y", "Z"], [("X", "Y"), ("X", "Z")]))
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


def test_parameter_decisions_and_law_changes_highlight_only_owning_mechanisms():
    model = complete_test_model(make_model(["X", "Y", "Z"], [("X", "Y"), ("X", "Z")]))
    edge = model.edges[0]
    parameter = model.parameter(next(iter(referenced_parameter_ids(edge))))
    pinned = model.revised(
        parameters=tuple(
            item.model_copy(
                update={
                    "value": 0,
                    "distribution": None,
                    "distribution_transform": "identity",
                    "reference_interval_days": None,
                }
            )
            if item.id == parameter.id
            else item
            for item in model.parameters
        ),
        distributions={
            key: law for key, law in model.distributions.items() if key != parameter.distribution
        },
    )
    for before, after, decision in ((model, pinned, "pinned"), (pinned, model, "released")):
        changes = compare_parameters(before, after)
        assert [(item.parameter_id, item.change) for item in changes] == [(parameter.id, decision)]
        graph = compare_model_graph(before, after, changes)
        assert [
            (item.edge_id, item.parameter_ids) for item in graph.edges if item.change == "revised"
        ] == [(edge.id, [parameter.id])]
        assert all(item.change == "unchanged" for item in graph.constructs)

    # The parameter and law ID can stay the same while the native law changes.
    revised = model.revised(
        distributions={**model.distributions, parameter.distribution: dist.Normal(2.0, 1.0)}
    )
    changes = compare_parameters(model, revised)
    assert [(item.parameter_id, item.change) for item in changes] == [(parameter.id, "revised")]
    graph = compare_model_graph(model, revised, changes)
    assert [item.edge_id for item in graph.edges if item.change == "revised"] == [edge.id]


def test_graph_additions_removals_and_measurement_changes_preserve_entity_identity():
    before = make_model(["X", "Y"], [("X", "Y")])
    after = make_model(["X", "Y", "Z"], [("X", "Y"), ("Z", "Y")])
    x = after.constructs[0]
    renamed = x.model_copy(update={"name": "Renamed X", "description": "Updated measurement"})
    after = after.revised(edges=replace_constructs(after.edges, (renamed,)))
    graph = compare_model_graph(before, after, [])
    assert {
        item.after.name: (item.before.name if item.before else None, item.change)
        for item in graph.constructs
        if item.after
    } == {"Renamed X": ("X", "revised"), "Y": ("Y", "unchanged"), "Z": (None, "added")}
    assert (
        next(item for item in graph.edges if item.edge_id == before.edges[0].id).change
        == "unchanged"
    )
    assert next(item for item in graph.edges if item.edge_id == after.edges[1].id).change == "added"
    reverse = compare_model_graph(after, before, [])
    assert (
        next(item for item in reverse.edges if item.edge_id == after.edges[1].id).change
        == "removed"
    )
    assert (
        next(item for item in reverse.constructs if item.before and item.before.name == "Z").after
        is None
    )
    unchanged = compare_model_graph(after, after, [])
    assert all(item.change == "unchanged" for item in (*unchanged.constructs, *unchanged.edges))


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
                constructs["U"].model_copy(update={"role": "exogenous", "indicators": ()}),
                constructs["V"].model_copy(update={"indicators": ()}),
            ),
        ),
    )
    structural = measured.revised(measurement_clock=None)
    retained, edges = model_graph_entities(measured)
    assert {item.name for item in retained} == {"X", "Y"}
    assert [(item.cause.name, item.effect.name) for item in edges] == [("X", "Y")]
    assert set(measured.state_order) == {constructs["X"].id, constructs["Y"].id}
    assert len(model_graph_entities(structural)[0]) == 6
    forward = compare_model_graph(structural, measured, [])
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
    reverse = compare_model_graph(measured, structural, [])
    assert {
        item.after.name for item in reverse.constructs if item.after and item.change == "added"
    } == {"U", "V", "A", "B"}
    assert len([item for item in forward.edges if item.change == "removed"]) == 4
    assert len([item for item in reverse.edges if item.change == "added"]) == 4
    same = compare_model_graph(measured, measured, [])
    assert {item.after.name for item in same.constructs if item.after} == {"X", "Y"}
    assert all(item.change == "unchanged" for item in (*same.constructs, *same.edges))
