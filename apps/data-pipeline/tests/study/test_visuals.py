"""The workbench retains irregular timing, modes and paired draws."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.artifacts.availability import Available
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId
from nof1_causal_lab.artifacts.scenarios import CausalEffectResult
from nof1_causal_lab.models.model_structure import StructuralSelection, selected_state_ids
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.snapshots import ModelReader
from tests.inference_fixtures import compile_model_fixture
from tests.model_fixtures import load_model_fixture, x_y_model


def _x_y_z_model() -> ModelSpec:
    return load_model_fixture("visuals/x_y_z_model.json")


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


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
        data_history=SimpleNamespace(
            variables=[variable], time_origin=origin.replace(tzinfo=UTC), frame=table
        ),
        state=SimpleNamespace(current={"panel": SimpleNamespace(revision="pinned")}),
        store=SimpleNamespace(read_parquet_file=lambda *_args: table),
    )
    view = ModelReader.observation_history(reader, identity)
    assert view is not None
    assert view.times == (0, 0.25, 8, 10, 11)
    assert view.values == (-4, -4, None, 5, 5)
    assert view.support_start == (-1, 0, 7, 9, 10)
    assert [(p.value, p.probability) for p in view.empirical] == [(-4, 0.5), (5, 1)]


@pytest.mark.contract
def test_paging_original_paths_preserves_opposite_modes_and_paired_effects(monkeypatch):

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
            indicator_ids=(indicator,),
            variables=[
                SimpleNamespace(
                    id=indicator,
                    name="Observed",
                    measurement_dtype="continuous",
                    ordinal_levels=None,
                    categorical_levels=None,
                )
            ],
        ),
        model=SimpleNamespace(revision="pinned-model"),
        causal=Available(value=CausalEffectResult(outcome=state_id, labels={state_id: "State"})),
    )
    reader = Mock(
        spec=ModelReader,
        simulation=lambda: SimpleNamespace(value=report),
        store=SimpleNamespace(read_array=arrays.__getitem__),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.study.store.read_model",
        lambda *_args: SimpleNamespace(
            get_construct=lambda _identity: SimpleNamespace(name="State")
        ),
    )
    report.evidence = SimpleNamespace(
        **{key: value for key, value in vars(report).items() if key != "causal"}
    )
    view = ModelReader.simulation_paths(reader, start=0, count=2)
    assert view is not None
    assert view.effect is not None
    assert view.times == report.times
    assert view.total_draws == 3
    assert view.effect_summary is not None
    assert view.effect_summary.mean == pytest.approx(11 / 3)
    assert [p.values for p in view.states[state_id].action] == [(-5, -4, -5), (5, 4, 5)]
    assert [p.values for p in view.effect.action] == [(1, 1, 1), (3, 3, 3)]
    assert view.indicators[indicator].action[0].values == (-5, None, -5)
    last = ModelReader.simulation_paths(reader, start=2, count=128)
    assert last is not None
    assert last.count == 1
    assert last.states[state_id].action[0].draw == 2
    with pytest.raises(StudyLookupError, match="past"):
        ModelReader.simulation_paths(reader, start=3, count=1)


@pytest.mark.inference(concern="sampling")
def test_every_parameter_coordinate_and_joint_draw_survives_the_read(monkeypatch):
    from nof1_causal_lab.models.ssm.compile.bindings import joint_law_layout, parameter_bindings
    from nof1_causal_lab.numpyro_json import empirical_distribution

    model = _x_y_z_model()
    bindings, _ = parameter_bindings(compile_model_fixture(model))
    layout = joint_law_layout(
        bindings,
        parameters=[b.parameter_id for b in bindings],
        constructs=selected_state_ids(StructuralSelection(model, None)),
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
        law_layouts={layout.distribution_id: layout},
    )
    monkeypatch.setattr(
        "nof1_causal_lab.study.lineage.law_provenance",
        lambda *_args: SimpleNamespace(kind="fitted"),
    )
    reader = Mock(
        spec=ModelReader,
        model=model,
        selection=StructuralSelection(model, None),
        store=None,
        state=SimpleNamespace(current={"model": None}),
    )
    monkeypatch.setattr(
        "nof1_causal_lab.models.ssm.compile.inputs.compile_executable_model",
        lambda *_args: pytest.fail("Raw draws must remain readable without the compiler"),
    )
    view = ModelReader.parameter_draws(reader)
    assert view.kind == "available"
    assert len(view.value) > 6
    assert {column.subject.element_id for column in view.value} == set(layout.parameter_columns)
    for column in view.value:
        np.testing.assert_array_equal(
            column.values, atoms[:, layout.parameter_columns[column.subject.element_id]]
        )
        assert len(column.values) == 503
        assert column.empirical[-1].probability == 1


@pytest.mark.contract
def test_predictive_overlay_uses_pinned_schedule_including_support_boundaries(monkeypatch):
    from nof1_causal_lab.artifacts.posterior_diagnostics import PPCOverlay

    model = x_y_model()
    identity = model.indicators[0].observation.id
    overlay = PPCOverlay(
        indicator_id=identity,
        times=(0, 1, 9, 10),
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        standardized=True,
        observed=[None, 1.0, None, 1.0],
        median=[None, 2.0, None, 2.0],
        spaghetti_draws=[[None, 3.0, None, 3.0]],
    )
    check = SimpleNamespace(
        evaluation=SimpleNamespace(
            kind="evaluated", predictive_checks=SimpleNamespace(overlays=[overlay])
        )
    )
    monkeypatch.setattr(
        "nof1_causal_lab.study.store.read_model",
        lambda *_args: pytest.fail("Overlay reads must not reconstruct their schedule"),
    )
    reader = Mock(spec=ModelReader, checks=(SimpleNamespace(predictive=check), None, None))
    view = ModelReader.predictive_history(reader, identity)
    assert view is overlay
    assert view.times == (0, 1, 9, 10)
    assert view.standardized is True
