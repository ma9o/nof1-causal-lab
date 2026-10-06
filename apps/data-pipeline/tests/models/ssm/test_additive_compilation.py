"""Retained scientific identity and explicit execution requirements; no numerical runs."""

import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import hill as expr_hill
from nof1_causal_lab.artifacts.expressions import state as expr_state
from nof1_causal_lab.artifacts.likelihood import DeltaLawSpec, LikelihoodSpec
from nof1_causal_lab.artifacts.mechanism import DriftMechanismSpec
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.model_structure import (
    StructuralCompilationError,
    StructuralSelection,
    selected_state_ids,
    validate_execution_structure,
)
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.simulation_checks import (
    ConstructSimulationTarget,
    _incoming_edge_off_target,
)
from tests.helpers import make_model
from tests.inference_fixtures import compile_model_fixture
from tests.model_fixtures import additive_a_b_model

pytestmark = pytest.mark.contract


def test_static_target_is_retained_and_reported_as_unsupported():
    draft = make_model(["A", "B"], [("A", "B")])
    static_constructs = tuple(
        node.revised(temporal_status="time_invariant") for node in draft.constructs
    )
    model = draft.revised(edges=replace_constructs(draft.edges, static_constructs))
    before = model.model_dump(mode="json")
    selection = StructuralSelection(model, None)
    assert set(selected_state_ids(selection)) == {node.id for node in model.constructs}
    with pytest.raises(StructuralCompilationError, match="static-target edge"):
        validate_execution_structure(selection)
    assert any(
        item.target.id == model.edges[0].id and item.disposition == "unsupported"
        for item in selection.structural_dispositions
    )
    assert model.model_dump(mode="json") == before


def test_exact_observations_preserve_state_and_parameter_identity():
    draft = additive_a_b_model()
    source = draft.edges[0].cause
    observed = source.revised(
        indicators=(
            source.indicators[0].revised(
                likelihood=LikelihoodSpec(
                    law=DeltaLawSpec(v=expr_state(source.id)),
                    standardized=False,
                    reasoning="The source is observed exactly.",
                )
            ),
        )
    )
    model = draft.revised(edges=replace_constructs(draft.edges, (observed,)))
    selection = StructuralSelection(model, None)
    assert source.id in selected_state_ids(selection)
    assert model.parameters == draft.parameters
    edge_parameters = model.parameters_for(model.edges[0].id)
    assert edge_parameters
    for parameter in edge_parameters:
        assert model.parameter(parameter.id) == draft.parameter(parameter.id)
        assert model.parameter_context(parameter.id).quantity == SiteKind.DYNAMICS_WEIGHT


def test_edge_off_targets_every_additive_contribution_without_running_a_simulation():
    model = additive_a_b_model()
    edge = model.edges[0]
    fixed_hill = DriftMechanismSpec(
        id="mechanism:fixed-hill-a",
        expression=expr_hill(expr_state(edge.cause.id), emax=0.4, ec50=1, n=2),
    )
    model = model.revised(
        edges=(
            edge.revised(
                mechanisms=(
                    *edge.mechanisms,
                    fixed_hill,
                    fixed_hill.revised(id="mechanism:fixed-hill-b"),
                )
            ),
        )
    )
    native = model
    target = model.get_construct(edge.effect.id)
    source = model.get_construct(edge.cause.id)
    contribution = ConstructSimulationTarget(
        construct=compile_model_fixture(model).states[
            selected_state_ids(StructuralSelection(model, None)).index(target.id)
        ],
        edge_parents=(source.name,),
    )
    assert numeric.state_names(compile_model_fixture(native)) is not None
    off = _incoming_edge_off_target(
        compile_model_fixture(native),
        contribution,
        numeric.state_names(compile_model_fixture(native)),
        numeric.state_names(compile_model_fixture(native)).index(target.name),
    )
    assert len(off.components) == 3


def test_predictive_edge_findings_exclude_the_response_state(monkeypatch):
    """State-dependent edge mechanisms do not invent a response-to-itself edge."""
    import jax.numpy as jnp

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.predictive import simulation
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
    from nof1_causal_lab.models.ssm.reachability import CheckResult
    from nof1_causal_lab.models.ssm.simulation_checks import DesignInfo

    model = additive_a_b_model()
    edge = model.edges[0]
    mechanism = DriftMechanismSpec(
        id="mechanism:response-dependent-hill",
        expression=expr_hill(expr_state(edge.cause.id), emax=0.4, ec50=1, n=2)
        * expr_state(edge.effect.id),
    )
    model = model.revised(edges=(edge.revised(mechanisms=(*edge.mechanisms, mechanism)),))
    compiled = compile_model_fixture(model)
    received: list[ConstructSimulationTarget] = []

    def measured(
        _model: CompiledModel,
        _prediction: PredictiveDraws,
        _design: DesignInfo,
        target: ConstructSimulationTarget,
        **_kwargs: object,
    ) -> tuple[list[CheckResult], list[object]]:
        received.append(target)
        return [
            CheckResult.measured(
                "edge attribution",
                f"{parent}->{target.construct.name}",
                "1",
                "0",
                "Retain the declared parent edge.",
                outcome="passed",
                measurements=(),
            )
            for parent in target.edge_parents
        ], []

    monkeypatch.setattr(simulation, "measure_construct_simulation", measured)
    latent = jnp.zeros((1, 2, len(compiled.states)))
    observed = jnp.zeros((1, 2, len(compiled.observations)))
    prediction = PredictiveDraws(
        {},
        PredictiveTrajectory(
            latent, observed, observed, jnp.ones_like(observed, dtype=bool), observed
        ),
    )
    batch = simulation.SimulationBatch(
        (0.0, 1.0), prediction, None, DesignInfo(jnp.array([0.0, 1.0]), (), {}, {})
    )
    findings, _ = simulation.measure_simulation_batch(
        compiled, batch, groups=("dynamics",), clock=lambda: 0.0
    )
    response = next(target for target in received if target.construct.id == edge.effect.id)
    assert response.edge_parents == (edge.cause.name,)
    assert response.hill_parents == (edge.cause.name,)
    assert [
        finding.subject.target.id
        for finding in findings
        if not isinstance(finding.subject.target, str)
    ] == [edge.id]
