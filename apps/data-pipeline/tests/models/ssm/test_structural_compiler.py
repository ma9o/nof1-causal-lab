"""Execution planning reads canonical entities and preserves their source identities."""

import pytest

from nof1_causal_lab.artifacts.construct import (
    KnownInput,
    Role,
    ScientificOnlyConstruct,
    TemporalStatus,
    replace_constructs,
)
from nof1_causal_lab.models.model_checks import check_execution
from nof1_causal_lab.models.model_structure import StructuralCompilationError
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from tests.helpers import complete_test_model, make_model


def test_planner_rejects_retained_static_target_edge():
    model = make_model(["X", "Baseline", "Y"], [("X", "Baseline"), ("Baseline", "Y")])
    x, baseline, y = model.constructs
    model = model.revised(
        edges=replace_constructs(
            tuple(edge.model_copy(update={"lagged": False}) for edge in model.edges),
            (
                x.model_copy(
                    update={
                        "role": Role.EXOGENOUS,
                        "temporal_status": TemporalStatus.TIME_INVARIANT,
                    }
                ),
                baseline.model_copy(update={"temporal_status": TemporalStatus.TIME_INVARIANT}),
                y,
            ),
        )
    )
    with pytest.raises(StructuralCompilationError, match="static-target edge"):
        (model).require_execution_structure()


def test_model_rejects_multiple_lag_classes_for_one_edge():
    model = make_model(["X", "Y"], [("X", "Y")])
    duplicate = model.edges[0].model_copy(update={"id": "edge:another", "lagged": False})
    with pytest.raises(ValueError, match="one causal edge per endpoint pair"):
        model.revised(edges=(*model.edges, duplicate))


def _model_with_usage():
    model = make_model(
        ["X", "Y", "Driver", "History", "U"],
        [("X", "Y"), ("Driver", "Y"), ("History", "Y"), ("U", "X"), ("U", "Y")],
    )
    x, y, driver, history, u = model.constructs
    return model.revised(
        edges=replace_constructs(
            model.edges,
            (
                x,
                y,
                driver.model_copy(
                    update={
                        "role": Role.EXOGENOUS,
                        "usage": KnownInput(source_indicator_id=driver.indicators[0].id),
                    }
                ),
                history.model_copy(
                    update={
                        "role": Role.EXOGENOUS,
                        "usage": ScientificOnlyConstruct(reason="Scientific context"),
                    }
                ),
                u.model_copy(update={"role": Role.EXOGENOUS, "indicators": ()}),
            ),
        )
    )


def test_planner_records_known_input_and_scientific_only_dispositions():
    model = _model_with_usage()
    plan = model
    dispositions = {d.source_id: d.disposition for d in plan.structural_dispositions}
    by_name = {c.name: c for c in model.constructs}
    assert [model.get_construct(key).name for key in plan.state_order] == ["X", "Y"]
    assert [model.indicator(key).name for key in plan.manifest_indicator_order] == [
        "X_obs",
        "Y_obs",
    ]
    assert dispositions[by_name["Driver"].id] == "known_input"
    assert dispositions[by_name["History"].id] == "identification_only"
    assert dispositions[by_name["U"].id] == "marginalized"
    assert plan.known_inputs[by_name["Driver"].id] is by_name["Driver"].usage
    assert plan.induced_dependencies
    assert plan.execution_edges[0] is model.edges[0]
    assert "semantics" not in plan.model_dump(mode="json")


def test_source_ids_are_stable_across_authoring_reordering():
    model = _model_with_usage()
    original = model
    reordered = model.revised(
        edges=replace_constructs(tuple(reversed(model.edges)), tuple(reversed(model.constructs)))
    )
    assert {d.source_id: d.disposition for d in original.structural_dispositions} == {
        d.source_id: d.disposition for d in reordered.structural_dispositions
    }
    assert original.induced_dependencies == reordered.induced_dependencies
    assert original.known_inputs == reordered.known_inputs
    assert original.reference_indicator_ids == reordered.reference_indicator_ids


def test_execution_checks_preserve_the_scientific_model():
    model = complete_test_model(make_model(["X", "Y"], [("X", "Y")]))
    plan = model
    before = model.model_dump(mode="json")
    artifact = check_execution(model)
    from nof1_causal_lab.models.ssm import numerics as numeric

    assert numeric.observation_names(model) == ["X_obs", "Y_obs"]
    assert [c.construct_id for c in artifact] == list(plan.state_order)
    assert {b.parameter_id for b in parameter_bindings(model)[0]} == {
        p.id for p in model.parameters
    }
    assert model.model_dump(mode="json") == before
