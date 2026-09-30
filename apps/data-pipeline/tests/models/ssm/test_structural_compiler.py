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


@pytest.mark.contract
def test_planner_rejects_retained_static_target_edge():
    model = make_model(["X", "Baseline", "Y"], [("X", "Baseline"), ("Baseline", "Y")])
    x, baseline, y = model.constructs
    model = model.revised(
        edges=replace_constructs(
            model.edges,
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


@pytest.mark.contract
def test_model_rejects_duplicate_endpoint_pairs():
    model = make_model(["X", "Y"], [("X", "Y")])
    duplicate = model.edges[0].model_copy(update={"id": "edge:another"})
    with pytest.raises(ValueError, match="one causal edge per endpoint pair"):
        model.revised(edges=(*model.edges, duplicate))


@pytest.mark.contract
def test_projected_fixed_coefficients_compile_identically_as_literals_and_references():
    import numpy as np

    from nof1_causal_lab.artifacts.expressions import coefficient, linear_effect
    from nof1_causal_lab.artifacts.identity import scientific_id
    from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.models.ssm.compile.mechanisms import iter_mechanism_components

    model = make_model(["U", "X", "Y"], [("U", "Y"), ("X", "Y")])
    root = next(item for item in model.constructs if item.name == "U")
    root = root.model_copy(
        update={
            "role": Role.EXOGENOUS,
            "temporal_status": TemporalStatus.TIME_INVARIANT,
            "indicators": (),
            "coefficients": (coefficient(0.0, "initial_mean"), coefficient(1.0, "initial_scale")),
        }
    )
    model = model.revised(edges=replace_constructs(model.edges, (root,)))
    parameter = ParameterSpec(
        id=scientific_id("parameter", "fixed-loading"),
        name="loading",
        description="Known loading",
        value=0.5,
    )

    def _with_loading(weight, parameters):
        edges = tuple(
            edge.model_copy(
                update={
                    "mechanisms": (
                        DynamicsMechanismSpec(
                            id="mechanism:fixed-loading",
                            expression=linear_effect(root.id, weight),
                        ),
                    )
                }
            )
            if edge.cause.id == root.id
            else edge
            for edge in model.edges
        )
        return model.revised(edges=edges, parameters=parameters)

    literal = _with_loading(0.5, ())
    referenced = _with_loading(parameter.id, (parameter,))
    assert np.array_equal(
        numeric.static_factor_loadings(literal), numeric.static_factor_loadings(referenced)
    )
    assert np.any(np.asarray(numeric.static_factor_loadings(referenced)) == 0.5)
    assert tuple(iter_mechanism_components(literal, literal.state_order)) == tuple(
        iter_mechanism_components(referenced, referenced.state_order)
    )
    unresolved = _with_loading(parameter.id, (parameter.model_copy(update={"value": None}),))
    with pytest.raises(ValueError, match="fixed linear"):
        numeric.static_factor_loadings(unresolved)
    with pytest.raises(ValueError, match="fixed linear"):
        tuple(iter_mechanism_components(unresolved, unresolved.state_order))


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


@pytest.mark.contract
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


@pytest.mark.contract
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


@pytest.mark.contract
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


@pytest.mark.contract
def test_required_unmeasured_mediator_cannot_be_silently_excluded():
    model = make_model(["X", "Mediator", "Y"], [("X", "Mediator"), ("Mediator", "Y")])
    x, mediator, y = model.constructs
    model = model.revised(
        default_outcome=y.id,
        edges=replace_constructs(
            model.edges,
            (
                x,
                mediator.model_copy(update={"indicators": ()}),
                y,
            ),
        ),
    )
    assert any(d.disposition == "unsupported" for d in model.structural_dispositions)
    with pytest.raises(StructuralCompilationError, match="Required constructs"):
        model.check_execution()


@pytest.mark.inference(concern="predictive")
def test_severed_components_do_not_require_priors_or_bind_numerical_parameters():
    import jax
    import jax.numpy as jnp

    from nof1_causal_lab.models.model_parameters import referenced_parameter_ids
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_model
    from nof1_causal_lab.models.ssm.inference.persistence import condition_model, model_draws
    from nof1_causal_lab.models.ssm.inference.types import (
        JointPosteriorDraws,
        ParticleMCMCPosterior,
    )
    from nof1_causal_lab.models.ssm.predictive.parameters import sample_model_laws

    model = make_model(
        ["A", "B", "Sink", "X", "Y"],
        [("A", "B"), ("B", "Sink"), ("Y", "Sink"), ("X", "Y")],
    )
    nodes = {item.name: item for item in model.constructs}
    model = complete_test_model(
        model.revised(
            edges=replace_constructs(
                model.edges, (nodes["Sink"].model_copy(update={"indicators": ()}),)
            )
        )
    )
    island_parameter = model.parameter(
        next(iter(referenced_parameter_ids(model.get_construct(nodes["A"].id))))
    )
    selected = model.revised(
        default_outcome=nodes["Y"].id,
        parameters=tuple(
            item.model_copy(update={"distribution": None})
            if item.id == island_parameter.id
            else item
            for item in model.parameters
        ),
        distributions={
            key: value
            for key, value in model.distributions.items()
            if key != island_parameter.distribution
        },
    )
    before = selected.model_dump(mode="json")
    assert set(numeric.state_names(selected)) == {"X", "Y"}
    assert len(selected.manifest_indicator_order) == 2
    assert island_parameter.id not in {item.id for item in selected.execution_parameters}
    assert {item.construct_id for item in selected.check_execution()} == {
        nodes["X"].id,
        nodes["Y"].id,
    }
    _, bindings, _, _ = compile_ssm_inputs_from_model(selected)
    assert {item.parameter_id for item in bindings} == {
        item.id for item in selected.execution_parameters if item.value is None
    }
    assert selected.model_dump(mode="json") == before

    draws = sample_model_laws(selected, draws=2, key=jax.random.PRNGKey(0))
    assert set(draws.state_ids) == {nodes["X"].id, nodes["Y"].id}
    # Exercise persistence with synthetic draws; no fitting or trajectory simulation.
    conditioned = condition_model(
        selected,
        ParticleMCMCPosterior(
            JointPosteriorDraws(draws.parameters, jnp.zeros((2, 2, 2)), draws.state_ids)
        ),
        times=jnp.array([0.0, 1.0]),
    )
    assert conditioned.parameter(island_parameter.id) == selected.parameter(island_parameter.id)
    assert set(conditioned.distributions) & set(selected.distributions)
    assert model_draws(conditioned).state_ids == draws.state_ids
    assert (
        sample_model_laws(conditioned, draws=2, key=jax.random.PRNGKey(1)).state_ids
        == draws.state_ids
    )

    # Without a selected outcome, the operation still covers all measured components.
    assert set(numeric.state_names(selected.revised(default_outcome=None))) == {"A", "B", "X", "Y"}


@pytest.mark.contract
def test_projected_latent_dependencies_keep_their_connected_states():
    model = make_model(["U", "X", "Y", "A"], [("U", "X"), ("U", "A"), ("X", "Y")])
    nodes = {item.name: item for item in model.constructs}
    selected = model.revised(
        default_outcome=nodes["Y"].id,
        edges=replace_constructs(
            model.edges,
            (nodes["U"].model_copy(update={"role": Role.EXOGENOUS, "indicators": ()}),),
        ),
    )
    assert nodes["U"].id in selected.marginalized_construct_ids
    assert set(selected.state_order) == {nodes[name].id for name in ("X", "Y", "A")}
