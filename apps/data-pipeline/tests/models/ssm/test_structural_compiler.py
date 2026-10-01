"""Execution planning reads canonical entities and preserves their source identities."""

from pathlib import Path

import pytest

from nof1_causal_lab.artifacts.construct import (
    Role,
    TemporalStatus,
    replace_constructs,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_checks import check_execution
from nof1_causal_lab.models.model_structure import StructuralCompilationError
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from tests.helpers import make_model
from tests.model_fixtures import compile_fit_fixture


@pytest.mark.contract
def test_planner_rejects_retained_static_target_edge():
    model = make_model(["X", "Baseline", "Y"], [("X", "Baseline"), ("Baseline", "Y")])
    x, baseline, y = model.constructs
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            (
                type(x).model_validate(
                    {
                        **x.model_dump(),
                        "role": Role.EXOGENOUS,
                        "temporal_status": TemporalStatus.TIME_INVARIANT,
                    }
                ),
                type(baseline).model_validate(
                    {**baseline.model_dump(), "temporal_status": TemporalStatus.TIME_INVARIANT}
                ),
                y,
            ),
        )
    )
    with pytest.raises(StructuralCompilationError, match="static-target edge"):
        (model).require_execution_structure()


@pytest.mark.contract
def test_model_rejects_duplicate_endpoint_pairs():
    model = make_model(["X", "Y"], [("X", "Y")])
    duplicate = type(model.edges[0]).model_validate(
        {**model.edges[0].model_dump(), "id": "edge:another"}
    )
    with pytest.raises(ValueError, match="one causal edge per endpoint pair"):
        model.revised(edges=(*model.edges, duplicate))


@pytest.mark.contract
def test_projected_coefficients_require_literals():
    import numpy as np

    from nof1_causal_lab.artifacts.expressions import coefficient, state
    from nof1_causal_lab.artifacts.identity import scientific_id
    from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.models.ssm.compile.mechanisms import iter_mechanism_components

    model = make_model(["U", "X", "Y"], [("U", "Y"), ("X", "Y")])
    root = next(item for item in model.constructs if item.name == "U")
    root = type(root).model_validate(
        {
            **root.model_dump(),
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
    )

    def _with_loading(weight, parameters):
        edges = tuple(
            type(edge).model_validate(
                {
                    **edge.model_dump(),
                    "mechanisms": (
                        DynamicsMechanismSpec(
                            id="mechanism:fixed-loading",
                            expression=coefficient(weight, "weight") * state(root.id),
                        ),
                    ),
                }
            )
            if edge.cause.id == root.id
            else edge
            for edge in model.edges
        )
        return model.revised(edges=edges, parameters=parameters)

    literal = _with_loading(0.5, ())
    assert np.any(np.asarray(numeric.static_factor_loadings(literal)) == 0.5)
    tuple(iter_mechanism_components(literal, literal.state_order))
    unresolved = _with_loading(parameter.id, (parameter,))
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
    indicator = type(driver.indicators[0]).model_validate(
        {
            **driver.indicators[0].model_dump(),
            "likelihood": LikelihoodSpec(
                law=ObservationLawSpec(distribution="Delta", arguments={"v": state(driver.id)}),
                reasoning="Direct exact driver observation",
            ),
        }
    )
    return model.revised(
        edges=replace_constructs(
            model.edges,
            (
                x,
                y,
                type(driver).model_validate(
                    {**driver.model_dump(), "role": Role.EXOGENOUS, "indicators": (indicator,)}
                ),
                history,
                type(u).model_validate(
                    {**u.model_dump(), "role": Role.EXOGENOUS, "indicators": ()}
                ),
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
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'structural_compiler/execution_checks_preserve_the_scientific_model_complete_test_model.json').read_text())
    before = model.model_dump(mode="json")
    check_execution(model)
    from nof1_causal_lab.models.ssm import numerics as numeric

    assert numeric.observation_names(model) == ["X_obs", "Y_obs"]
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
                type(mediator).model_validate({**mediator.model_dump(), "indicators": ()}),
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

    from nof1_causal_lab.models.ssm import numerics as numeric
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
    model = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'structural_compiler/severed_components_do_not_require_priors_or_bind_numerical_parameters_complete_test_model.json').read_text())
    island_parameter = model.parameters_for(nodes["A"].id)[0]
    selected = model.revised(
        default_outcome=nodes["Y"].id,
        parameters=tuple(
            type(item).model_validate({**item.model_dump(), "distribution": None})
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
    selected.check_execution()
    bindings = compile_fit_fixture(selected).bindings
    assert {item.parameter_id for item in bindings} == {
        item.id for item in selected.execution_parameters
    }
    assert selected.model_dump(mode="json") == before

    draws = sample_model_laws(selected, draws=2, key=jax.random.PRNGKey(0))
    assert set(draws.state_ids) == {nodes["X"].id, nodes["Y"].id}
    # Exercise persistence with synthetic draws; no fitting or trajectory simulation.
    conditioned = condition_model(
        compile_fit_fixture(selected),
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
            (
                type(nodes["U"]).model_validate(
                    {**nodes["U"].model_dump(), "role": Role.EXOGENOUS, "indicators": ()}
                ),
            ),
        ),
    )
    assert nodes["U"].id in selected.marginalized_construct_ids
    assert set(selected.state_order) == {nodes[name].id for name in ("X", "Y", "A")}
