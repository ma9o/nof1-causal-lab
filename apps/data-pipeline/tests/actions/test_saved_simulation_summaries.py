"""Saved scientific summaries retain masks, paired uncertainty and absolute time."""

from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nof1_causal_lab.actions.simulation_summaries import (
    paired_effect_trajectory,
    summarize_simulation,
)
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.artifacts.simulation import SimulationReport, SimulationSpec
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.predictive.simulation import generate_simulation_batch
from nof1_causal_lab.read_facade import create_read_facade_app
from tests.data_fixtures import simulation_layout
from tests.helpers import complete_test_model, make_model

pytestmark = pytest.mark.inference(concern="simulation")


def test_all_summary_types_and_paired_intervals_are_persisted_before_reads(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    model = make_model(
        ["x", "binary", "ordinal", "category"],
        [("x", "binary"), ("binary", "ordinal"), ("ordinal", "category")],
    )
    dtypes = {
        "x": "continuous",
        "binary": "binary",
        "ordinal": "ordinal",
        "category": "categorical",
    }
    constructs = tuple(
        c.model_copy(
            update={
                "indicators": tuple(
                    i.model_copy(
                        update={
                            "aggregation": "last",
                            "measurement_dtype": dtypes[c.name],
                            "ordinal_levels": ("low", "medium", "high")
                            if c.name == "ordinal"
                            else None,
                            "categorical_levels": ("a", "b") if c.name == "category" else None,
                        }
                    )
                    for i in c.indicators
                )
            }
        )
        for c in model.constructs
    )
    model = model.revised(edges=replace_constructs(model.edges, constructs))
    store, history = ArtifactStore("SUMMARY"), StudyRepository("SUMMARY")
    definition = store.write_artifact(
        "model",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    history.append(
        TransitionRecord(
            seq=1,
            ts="2026-01-01T00:00:00Z",
            action="edit_model",
            status="applied",
            produced=[definition],
            trace_ids=[],
            resume=None,
        )
    )
    by_type = {
        "continuous": [1.0, 2.0, 3.0],
        "binary": [0.0, 1.0, 1.0],
        "ordinal": [0.0, 1.0, 2.0],
        "categorical": [0.0, 1.0, 1.0],
    }
    observations = np.stack(
        [
            np.tile(by_type[model.indicator(i).measurement_dtype], (2, 1)).T
            for i in numeric.observation_ids(model)
        ],
        axis=-1,
    )
    mask = np.ones_like(observations, dtype=bool)
    mask[:, 0, 0] = False
    observations[:, 0, 0] = np.nan
    states = np.arange(observations.size, dtype=float).reshape(observations.shape)
    states[0, 0, 0] = np.nan
    layout = simulation_layout(model, (5, 7), mask, store.write_array)
    summary = summarize_simulation(
        model,
        state_ids=tuple(numeric.state_ids(model)),
        variables=layout.variables,
        latent_paths=states,
        observations=observations,
        mask=mask,
        reference_latent_paths=states - 1,
        reference_observations=observations,
        fit_reliability="unconverged",
    )
    assert summary.states[numeric.state_ids(model)[0]].action.n_draws == (2, 3)
    for variable in layout.variables:
        series = summary.indicators[variable.id].action
        if variable.measurement_dtype == "binary":
            assert series.kind == "categorical"
            assert series.probabilities["1"][1] == pytest.approx(2 / 3)
        if variable.measurement_dtype == "ordinal":
            assert series.kind == "categorical"
            assert series.probabilities["medium"][1] == pytest.approx(1 / 3)
        if variable.measurement_dtype == "categorical":
            assert series.kind == "categorical"
            assert series.probabilities["b"][1] == pytest.approx(2 / 3)
    assert summary.indicators[layout.variables[0].id].action.n_draws == (0, 3)
    paired = paired_effect_trajectory((5, 7), np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert [(p.day, p.lower_95, p.upper_95) for p in paired] == [(5, 1.05, 2.95), (7, 2.05, 3.95)]
    report = SimulationReport(
        model=GitRef(workspace_id="SUMMARY", revision=definition.revision, path="model.json"),
        design=SimulationSpec(
            start=5,
            end=7,
            interventions=({"target": numeric.state_ids(model)[0], "time": 5, "value": 1},),
        ),
        times=(5, 7),
        draws=3,
        seed=0,
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        state_ids=tuple(numeric.state_ids(model)),
        parameter_draws={},
        latent_paths=store.write_array(states),
        observations=store.write_array(observations),
        reference_latent_paths=store.write_array(states - 1),
        reference_observations=store.write_array(observations),
        observation_layout=layout,
        predictive=summary,
    )
    history.append(
        TransitionRecord(
            seq=2,
            ts="2026-01-01T01:00:00Z",
            action="simulate",
            operation_id="simulate",
            status="applied",
            diagnostics={"report": report.model_dump(mode="json")},
            trace_ids=[],
            resume=None,
        )
    )

    client = TestClient(create_read_facade_app())
    paths = client.get("/api/episodes/SUMMARY/model/visuals/simulation?start=1&count=128")
    assert paths.status_code == 200, paths.text
    path_data = paths.json()
    assert path_data["total_draws"] == 3
    assert path_data["count"] == 2
    assert path_data["times"] == [5, 7]
    identity = numeric.state_ids(model)[0]
    assert path_data["states"][identity]["action"][0]["draw"] == 1
    assert path_data["states"][identity]["action"][0]["values"] == states[1, :, 0].tolist()
    assert client.get("/api/episodes/SUMMARY/model/visuals/simulation?start=3").status_code == 422
    assert client.get("/api/episodes/SUMMARY/model/visuals/simulation?count=0").status_code == 422

    def no_array_reads(*args, **kwargs):
        pytest.fail("Reading saved summaries must not load draws")

    monkeypatch.setattr(ArtifactStore, "read_array", no_array_reads)
    client = TestClient(create_read_facade_app())
    response = client.get("/api/episodes/SUMMARY/model")
    assert response.status_code == 200, response.text
    assert response.json()["findings"]["simulation"]["value"] == report.model_dump(mode="json")
    assert client.get("/api/episodes/SUMMARY/model/simulation-trajectories").status_code == 404


def test_authored_law_advances_from_zero_before_a_later_requested_start():
    model = complete_test_model(make_model(["X", "Y"], [("X", "Y")]))
    parameters = tuple(
        p.model_copy(
            update={
                "value": 0.5
                if p.name.startswith("rho")
                else 0.0
                if p.name.startswith("beta")
                else 1e-8,
                "distribution": None,
                "distribution_transform": "identity",
                "reference_interval_days": None,
            }
        )
        for p in model.parameters
    )
    constructs = tuple(
        c.model_copy(
            update={
                "coefficients": tuple(
                    coefficient.model_copy(
                        update={"value": 10.0 if coefficient.role == "initial_mean" else 1e-8}
                    )
                    if coefficient.role in {"initial_mean", "initial_scale"}
                    else coefficient
                    for coefficient in c.coefficients
                )
            }
        )
        for c in model.constructs
    )
    model = model.revised(
        edges=replace_constructs(model.edges, constructs), parameters=parameters, distributions={}
    )
    batch = generate_simulation_batch(
        model, SimulationSpec(start=2, end=3), draws=2, time_origin=None
    )
    np.testing.assert_allclose(
        batch.prediction.trajectory.latents[:, 0], 10 * np.exp(-1), rtol=0.002
    )
    with pytest.raises(ValueError, match="before the initial law"):
        generate_simulation_batch(model, SimulationSpec(start=-1, end=1), draws=2, time_origin=None)
