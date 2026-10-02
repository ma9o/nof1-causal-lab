"""The workbench retains irregular timing, modes, paired draws and nonlinear mechanisms."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    expression_coefficients,
    hill,
    restoring_potential,
    state,
)
from nof1_causal_lab.artifacts.identity import IndicatorId, MechanismId
from nof1_causal_lab.artifacts.mechanism import (
    DriftMechanismSpec,
    PotentialMechanismSpec,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.model_structure import selected_state_ids
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.snapshots import ModelReader
from nof1_causal_lab.study.visual_models import MechanismViewRequest
from tests.helpers import make_model
from tests.model_fixtures import compile_model_fixture


@pytest.mark.contract
def test_observations_keep_irregular_anchors_support_missingness_and_empirical_mass():
    origin = datetime(2026, 1, 1)
    identity = IndicatorId("indicator:observed")
    table = pl.DataFrame(
        {
            "indicator_id": [identity] * 5,
            "anchor_time": [origin + timedelta(days=t) for t in (0, 0.25, 8, 10, 11)],
            "support_start": [origin + timedelta(days=t) for t in (-1, 0, 7, 9, 10)],
            "support_end": [origin + timedelta(days=t) for t in (0, 0.25, 8, 10, 11)],
            "value": [-4.0, -4.0, None, 5.0, 5.0],
        }
    )
    variable = SimpleNamespace(
        id=identity, name="Observed", ordinal_levels=None, categorical_levels=None
    )
    reader = Mock(
        spec=ModelReader,
        data_metadata=SimpleNamespace(
            value=SimpleNamespace(variables=[variable], time_origin=origin.replace(tzinfo=UTC))
        ),
        state=SimpleNamespace(current={"panel": SimpleNamespace(revision="pinned")}),
        store=SimpleNamespace(read_parquet_file=lambda *_args: table),
    )
    view = ModelReader.observation_history(reader, identity)
    assert view is not None
    assert view.times == (0, 0.25, 8, 10, 11)
    assert view.values == (-4, -4, None, 5, 5)
    assert view.support_start == (-1, 0, 7, 9, 10)
    assert [(p.value, p.probability, p.count) for p in view.empirical] == [(-4, 0.5, 2), (5, 1, 2)]


@pytest.mark.contract
def test_paging_original_paths_preserves_opposite_modes_and_paired_effects():
    from nof1_causal_lab.artifacts.identity import ConstructId

    state_id = ConstructId("construct:state")
    indicator = IndicatorId("indicator:observed")
    action = np.array([[-5.0, -4.0, -5.0], [5.0, 4.0, 5.0], [8.0, 9.0, 8.0]])[:, :, None]
    reference = action - np.array([1.0, 3.0, 7.0])[:, None, None]
    mask = np.ones_like(action, dtype=bool)
    mask[:, 1, 0] = False
    arrays = {"a": action, "r": reference, "mask": mask}
    report = SimpleNamespace(
        draws=3,
        times=(0.0, 0.25, 9.0),
        time_origin=None,
        latent_paths="a",
        observations="a",
        reference_latent_paths="r",
        reference_observations="r",
        state_ids=(state_id,),
        observation_layout=SimpleNamespace(
            mask="mask",
            variables=[
                SimpleNamespace(
                    id=indicator, name="Observed", ordinal_levels=None, categorical_levels=None
                )
            ],
        ),
        predictive=SimpleNamespace(states={state_id: SimpleNamespace(label="State")}),
        causal_result=SimpleNamespace(outcome=state_id, labels={state_id: "State"}),
    )
    reader = Mock(
        spec=ModelReader,
        simulation=lambda: SimpleNamespace(value=report),
        store=SimpleNamespace(read_array=arrays.__getitem__),
    )
    view = ModelReader.simulation_paths(reader, start=0, count=2)
    assert view is not None
    assert view.effect is not None
    assert view.times == report.times
    assert view.total_draws == 3
    assert [p.values for p in view.states[state_id].action] == [(-5, -4, -5), (5, 4, 5)]
    assert [p.values for p in view.effect.action] == [(1, 1, 1), (3, 3, 3)]
    assert view.indicators[indicator].action[0].values == (-5, None, -5)
    last = ModelReader.simulation_paths(reader, start=2, count=128)
    assert last is not None
    assert last.count == 1
    assert last.states[state_id].action[0].draw == 2
    with pytest.raises(StudyLookupError, match="past"):
        ModelReader.simulation_paths(reader, start=3, count=1)


@pytest.mark.inference(concern="simulation")
def test_exact_hill_curves_retain_saturation_and_sign_changing_moderation():
    model = make_model(["X", "Y"], [("X", "Y")])
    edge = model.edges[0]
    expression = hill(state(edge.cause.id), emax=4.0, ec50=2.0, n=2.0) * state(edge.effect.id)
    model = model.revised(
        edges=(
            edge.revised(
                mechanisms=(
                    DriftMechanismSpec(id=MechanismId("mechanism:hill"), expression=expression),
                )
            ),
        )
    )
    request = MechanismViewRequest(
        owner_id=edge.id,
        lower=0,
        upper=10,
        moderator=edge.effect.id,
        levels=(-1.0, 1.0),
        points=101,
    )
    result = ModelReader.mechanism_curves(Mock(spec=ModelReader, model=model), request)
    x = np.asarray(result.x)
    expected = 4 * x**2 / (4 + x**2)
    np.testing.assert_allclose(
        np.asarray(result.curves[0].values, dtype=float), -expected, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(result.curves[1].values, dtype=float), expected, atol=1e-6
    )
    assert result.law == "fixed"
    assert result.total_draws == 1
    assert result.curves[1].values[20] == pytest.approx(2.0)
    assert result.curves[1].values[-1] is not None
    assert result.curves[1].values[-1] < 4
    with pytest.raises(StudyLookupError, match="moderator"):
        ModelReader.mechanism_curves(
            Mock(spec=ModelReader, model=model),
            request.revised(moderator=edge.cause.id),
        )


@pytest.mark.inference(concern="simulation")
def test_potential_response_is_the_negative_gradient_not_the_potential():
    model = make_model(["X", "Y"], [("X", "Y")])
    owner = model.edges[0].effect
    potential = PotentialMechanismSpec(
        id=MechanismId("mechanism:potential"),
        kind="potential",
        expression=restoring_potential(owner.id, center=1.0, stiffness=2.0, quartic=3.0),
    )
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            [owner.revised(dynamics=(potential,))],
        )
    )
    result = ModelReader.mechanism_curves(
        Mock(spec=ModelReader, model=model),
        MechanismViewRequest(owner_id=owner.id, lower=-2, upper=4),
    )
    delta = np.asarray(result.x) - 1
    np.testing.assert_allclose(
        np.asarray(result.curves[0].values, dtype=float),
        -2 * delta - 3 * delta**3,
        rtol=3e-6,
        atol=1e-5,
    )


@pytest.mark.inference(concern="sampling")
def test_every_parameter_coordinate_and_joint_draw_survives_the_read(monkeypatch):
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.joint_layout import JointLawLayout
    from nof1_causal_lab.numpyro_json import empirical_distribution

    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "visuals/x_y_z_model.json"
        ).read_text()
    )
    bindings, _ = parameter_bindings(compile_model_fixture(model))
    layout = JointLawLayout.from_bindings(
        bindings,
        parameters=[b.parameter_id for b in bindings],
        constructs=selected_state_ids(model),
        time_points=(0, 10),
    )
    atoms = np.arange(503 * layout.width, dtype=float).reshape(503, layout.width)
    model = model.revised(
        parameters=tuple(
            p.revised(distribution=layout.distribution_id, transform={"kind": "identity"})
            for p in model.parameters
        ),
        edges=replace_constructs(
            model.edges,
            [c.revised(distribution=layout.distribution_id) for c in model.constructs],
        ),
        distributions={layout.distribution_id: empirical_distribution(atoms)},
        time_points=(0, 10),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.study.lineage.law_provenance",
        lambda *_args: SimpleNamespace(kind="fitted"),
    )
    reader = Mock(
        spec=ModelReader, model=model, store=None, state=SimpleNamespace(current={"model": None})
    )
    view = ModelReader.parameter_draws(reader)
    assert len(view.columns) > 6
    assert {column.subject.element_id for column in view.columns} == set(layout.parameter_columns)
    for column in view.columns:
        np.testing.assert_array_equal(
            column.values, atoms[:, layout.parameter_columns[column.subject.element_id]]
        )
        assert sum(point.count for point in column.empirical) == 503
        assert column.empirical[-1].probability == 1


@pytest.mark.inference(concern="simulation")
def test_declared_scalar_law_can_be_inspected_with_unfinished_unrelated_mechanisms():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "visuals/x_y_z_model.json"
        ).read_text()
    )
    edge, unfinished = model.edges
    removed = {
        operand.value
        for m in unfinished.mechanisms
        for operand in expression_coefficients(m.expression)
    }
    kept = tuple(p for p in model.parameters if p.id not in removed)
    laws = {p.distribution for p in kept} | {c.distribution for c in model.constructs}
    model = model.revised(
        edges=(
            edge,
            unfinished.revised(mechanisms=()),
        ),
        parameters=kept,
        distributions={
            identity: law for identity, law in model.distributions.items() if identity in laws
        },
    )
    view = ModelReader.mechanism_curves(
        Mock(spec=ModelReader, model=model), MechanismViewRequest(owner_id=edge.id, count=2)
    )
    assert view.law == "sampled"
    assert view.count == 2
    assert view.nonfinite == 0
    assert view.curves[0].values != view.curves[1].values


@pytest.mark.contract
def test_predictive_overlay_uses_pinned_schedule_including_support_boundaries(monkeypatch):
    from nof1_causal_lab.artifacts.posterior_diagnostics import PPCOverlay

    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_y_model.json"
        ).read_text()
    )
    origin = datetime(2026, 1, 1)
    panel = pl.DataFrame(
        [
            {
                "indicator_id": indicator.observation.id,
                "value": 1.0,
                "anchor_time": origin + timedelta(days=t),
                "support_start": origin + timedelta(days=t - 1),
                "support_end": origin + timedelta(days=t),
                "support_kind": "interval",
                "summary_operator": "mean",
                "anchor_policy": "support_end",
                "observation_window": "1d",
            }
            for indicator in model.indicators
            for t in (1, 10)
        ]
    )
    identity = model.indicators[0].observation.id
    overlay = PPCOverlay(
        indicator_id=identity,
        observed=[None, 1.0, None, 1.0],
        median=[None, 2.0, None, 2.0],
        spaghetti_draws=[[None, 3.0, None, 3.0]],
    )
    check = SimpleNamespace(
        model_revision="pinned-model",
        panel_revision="pinned-panel",
        law=SimpleNamespace(fitted_model_revision=None),
        predictive_checks=SimpleNamespace(overlays=[overlay]),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.study.store.read_model",
        lambda _store, revision: (
            model if revision == "pinned-model" else pytest.fail("wrong model")
        ),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.study.lineage.read_data_metadata",
        lambda _store, revision: (
            SimpleNamespace(time_origin=origin.replace(tzinfo=UTC))
            if revision == "pinned-panel"
            else pytest.fail("wrong panel")
        ),
    )
    reader = Mock(
        spec=ModelReader,
        state=SimpleNamespace(checks=SimpleNamespace(predictive=check)),
        store=SimpleNamespace(
            read_parquet_file=lambda _artifact, revision, _file: (
                panel if revision == "pinned-panel" else pytest.fail("wrong panel")
            )
        ),
    )
    view = ModelReader.predictive_history(reader, identity)
    assert view is not None
    assert view.times == (0, 1, 9, 10)
    assert view.overlay == overlay
