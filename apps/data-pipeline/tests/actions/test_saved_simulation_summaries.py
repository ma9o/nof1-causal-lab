"""Saved scientific summaries retain masks, paired uncertainty and absolute time."""

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nof1_causal_lab.actions.effects import ActionEffects
from nof1_causal_lab.artifacts.availability import Unavailable
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.simulation import (
    SimulationObservationLayout,
    SimulationReport,
    SimulationSpec,
)
from nof1_causal_lab.models.model_structure import StructuralSelection, selected_state_ids
from nof1_causal_lab.models.ssm.predictive.simulation import generate_simulation_batch
from nof1_causal_lab.read_facade import create_read_facade_app
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Applied, ModelSimulationResult
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import applied_record
from tests.data_fixtures import metadata_for_model
from tests.helpers import make_model, write_question
from tests.model_fixtures import compile_model_fixture

pytestmark = pytest.mark.inference(concern="simulation")


def test_full_categories_and_paired_paths_are_derived_on_read_and_cached(tmp_path, monkeypatch):
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
        c.revised(
            indicators=tuple(
                i.revised(
                    observation=i.observation.revised(
                        aggregation="last",
                        measurement_dtype=dtypes[c.name],
                        ordinal_levels=("low", "medium", "high") if c.name == "ordinal" else None,
                        categorical_levels=("a", "b") if c.name == "category" else None,
                    )
                )
                for i in c.indicators
            )
        )
        for c in model.constructs
    )
    model = model.revised(edges=replace_constructs(model.edges, constructs))
    store, history = ArtifactStore("SUMMARY"), StudyRepository("SUMMARY")
    definition = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    history.append(
        applied_record(
            Applied(
                result=None,
                effects=ActionEffects(produced=[write_question(store), definition]),
            ),
            seq=1,
            ts="2026-01-01T00:00:00Z",
            trace_ids=[],
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
            np.tile(by_type[model.indicator(i).observation.measurement_dtype], (2, 1)).T
            for i in (indicator.observation.id for indicator in model.indicators)
        ],
        axis=-1,
    )
    mask = np.ones_like(observations, dtype=bool)
    mask[:, 0, 0] = False
    observations[:, 0, 0] = np.nan
    states = np.arange(observations.size, dtype=float).reshape(observations.shape)
    states[0, 0, 0] = np.nan
    support_times = np.broadcast_to(np.array([[5.0], [7.0]]), (2, len(model.indicators)))
    layout = SimulationObservationLayout(
        variables=metadata_for_model(model).variables,
        support_start_times=store.write_array(support_times),
        support_end_times=store.write_array(support_times),
        mask=store.write_array(mask),
    )
    report = SimulationReport(
        causal=Unavailable(reason="No causal effect was recorded."),
        model=GitRef(workspace_id="SUMMARY", revision=definition.revision, path="model.json"),
        design=SimulationSpec(
            start=date(2026, 1, 6),
            horizon="2d",
            interventions=(
                {"target": selected_state_ids(StructuralSelection(model, None))[0], "value": 1},
            ),
        ),
        assignments=(
            {
                "target": selected_state_ids(StructuralSelection(model, None))[0],
                "time": 5,
                "value": 1,
            },
        ),
        times=(5, 7),
        draws=3,
        seed=0,
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
        state_ids=tuple(selected_state_ids(StructuralSelection(model, None))),
        parameter_draws={},
        latent_paths=store.write_array(states),
        observations=store.write_array(observations),
        reference_latent_paths=store.write_array(states - 1),
        reference_observations=store.write_array(observations),
        observation_layout=layout,
        fit_reliability="not_fitted",
    )
    history.append(
        applied_record(
            Applied(result=ModelSimulationResult(report=report), effects=ActionEffects()),
            seq=2,
            ts="2026-01-01T01:00:00Z",
            trace_ids=[],
        )
    )

    client = TestClient(create_read_facade_app())
    paths = client.get("/api/studies/SUMMARY/model/visuals/simulation?start=1&count=128")
    assert paths.status_code == 200, paths.text
    path_data = paths.json()
    assert path_data["total_draws"] == 3
    assert path_data["count"] == 2
    assert path_data["times"] == [5, 7]
    identity = selected_state_ids(StructuralSelection(model, None))[0]
    assert path_data["states"][identity]["action"][0]["draw"] == 1
    assert path_data["states"][identity]["action"][0]["values"] == states[1, :, 0].tolist()
    assert client.get("/api/studies/SUMMARY/model/visuals/simulation?start=3").status_code == 422
    assert client.get("/api/studies/SUMMARY/model/visuals/simulation?count=0").status_code == 422

    for variable in layout.variables:
        if variable.measurement_dtype == "binary":
            assert path_data["action_category_probabilities"][variable.id]["probabilities"]["1"][
                1
            ] == pytest.approx(2 / 3)
        if variable.measurement_dtype == "ordinal":
            assert path_data["action_category_probabilities"][variable.id]["probabilities"][
                "medium"
            ][1] == pytest.approx(1 / 3)
        if variable.measurement_dtype == "categorical":
            assert path_data["action_category_probabilities"][variable.id]["probabilities"]["b"][
                1
            ] == pytest.approx(2 / 3)

    def no_array_reads(*args, **kwargs):
        pytest.fail("A cached draw read must not reload arrays")

    monkeypatch.setattr(ArtifactStore, "read_array", no_array_reads)
    assert (
        client.get("/api/studies/SUMMARY/model/visuals/simulation?start=1&count=128").json()
        == path_data
    )
    response = client.get("/api/studies/SUMMARY/model")
    assert response.status_code == 200, response.text
    assert response.json()["simulation"]["value"] == report.model_dump(mode="json")
    assert "predictive" not in response.json()["simulation"]["value"]


def test_authored_law_advances_from_zero_before_a_later_requested_start():
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1] / "fixtures/models" / "common/x_y_model.json"
        ).read_text()
    )

    fixed = {
        p.id: 0.5 if p.name.startswith("rho") else 0.0 if p.name.startswith("beta") else 1e-8
        for p in model.parameters
    }

    def literal_coefficients(value):
        if isinstance(value, list):
            return [literal_coefficients(item) for item in value]
        if isinstance(value, dict):
            if value.get("kind") == "coefficient" and value.get("value") in fixed:
                return {**value, "value": fixed[value["value"]]}
            return {key: literal_coefficients(item) for key, item in value.items()}
        return value

    payload = literal_coefficients(model.model_dump(mode="json"))
    payload.update(parameters=[], distributions={})
    model = ModelSpec.model_validate(payload)
    constructs = tuple(
        c.revised(
            coefficients=tuple(
                coefficient.revised(value=10.0 if coefficient.role == "initial_mean" else 1e-8)
                if coefficient.role in {"initial_mean", "initial_scale"}
                else coefficient
                for coefficient in c.coefficients
            )
        )
        for c in model.constructs
    )
    model = model.revised(edges=replace_constructs(model.edges, constructs))
    batch = generate_simulation_batch(compile_model_fixture(model), start=2.0, end=3.0, draws=2)
    np.testing.assert_allclose(
        batch.prediction.trajectory.latents[:, 0], 10 * np.exp(-1), rtol=0.002
    )
    with pytest.raises(ValueError, match="before the initial law"):
        generate_simulation_batch(compile_model_fixture(model), start=-1.0, end=1.0, draws=2)
