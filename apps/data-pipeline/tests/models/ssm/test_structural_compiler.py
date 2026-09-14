"""Execution planning reads canonical entities and preserves their source identities."""

import pytest

from nof1_causal_lab.artifacts.construct import (
    Role,
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


def _model_with_exact_measurement():
    from nof1_causal_lab.artifacts.expressions import state
    from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec, ObservationLawSpec

    model = make_model(
        ["X", "Y", "Driver", "History", "U"],
        [("X", "Y"), ("Driver", "Y"), ("History", "Y"), ("U", "X"), ("U", "Y")],
    )
    x, y, driver, history, u = model.constructs
    indicator = driver.indicators[0].model_copy(
        update={
            "likelihood": LikelihoodSpec(
                law=ObservationLawSpec(distribution="Delta", arguments={"v": state(driver.id)}),
                reasoning="Direct exact driver observation",
            )
        }
    )
    return model.revised(
        edges=replace_constructs(
            model.edges,
            (
                x,
                y,
                driver.model_copy(update={"role": Role.EXOGENOUS, "indicators": (indicator,)}),
                history,
                u.model_copy(update={"role": Role.EXOGENOUS, "indicators": ()}),
            ),
        )
    )


def test_exact_measurements_retain_scientific_states_and_project_only_supported_roots():
    model = _model_with_exact_measurement()
    dispositions = {d.target.id: d.disposition for d in model.structural_dispositions}
    by_name = {c.name: c for c in model.constructs}
    assert [model.get_construct(key).name for key in model.state_order] == [
        "X",
        "Y",
        "Driver",
        "History",
    ]
    assert dispositions[by_name["Driver"].id] == "retained_state"
    assert dispositions[by_name["History"].id] == "retained_state"
    assert dispositions[by_name["U"].id] == "marginalized"
    assert by_name["Driver"].indicators[0].id in model.manifest_indicator_order
    assert model.induced_dependencies
    assert model.execution_edges[0] is model.edges[0]


def test_source_ids_are_stable_across_authoring_reordering():
    model = _model_with_exact_measurement()
    original = model
    reordered = model.revised(
        edges=replace_constructs(tuple(reversed(model.edges)), tuple(reversed(model.constructs)))
    )
    assert {d.target.id: d.disposition for d in original.structural_dispositions} == {
        d.target.id: d.disposition for d in reordered.structural_dispositions
    }
    assert original.induced_dependencies == reordered.induced_dependencies
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


def test_required_unmeasured_mediator_cannot_be_silently_excluded():
    model = make_model(["X", "Mediator", "Y"], [("X", "Mediator"), ("Mediator", "Y")])
    x, mediator, y = model.constructs
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            (
                x,
                mediator.model_copy(update={"indicators": ()}),
                y,
            ),
        )
    )
    assert any(d.disposition == "unsupported" for d in model.structural_dispositions)
    with pytest.raises(StructuralCompilationError, match="Required unmeasured constructs"):
        model.check_execution()


def test_admission_scope_changes_execution_without_becoming_an_authored_field():
    from nof1_causal_lab.flows.model_spec_compile_cache import model_spec_fingerprint
    from nof1_causal_lab.models.model_structure import model_for_constructs

    model = make_model(["X", "Y"], [("X", "Y")])
    before = model.model_dump(mode="json")
    x = model_for_constructs(model, {"X"})
    y = model_for_constructs(model, {"Y"})
    assert model_spec_fingerprint(x) != model_spec_fingerprint(y)
    assert x.revised().state_order == (model.constructs[0].id,)
    assert model.model_dump(mode="json") == before
    assert "_selected_states" not in x.model_dump(mode="json")
