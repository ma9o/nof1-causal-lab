"""Synthetic observations compose with the normal panel, provenance and fitting contracts."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from nof1_causal_lab.actions.contracts import PrepareDataRequest
from nof1_causal_lab.actions.prepare_data import prepare_simulation_panel
from nof1_causal_lab.actions.runners import run_action
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.simulation import (
    SimulationObservationLayout,
    SimulationReport,
    SimulationSpec,
)
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.runtime import project_observation_data
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import ModelEditResult, ModelSimulationResult
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.store import ArtifactStore
from tests.action_fixtures import applied_record
from tests.data_fixtures import metadata_for_model, predictive_summary, simulation_layout
from tests.git_fixtures import git_oid
from tests.helpers import make_model, run_async
from tests.model_fixtures import compile_model_fixture

pytestmark = pytest.mark.contract


def _model(**indicator_changes):
    model = make_model(["X", "Y"], [("X", "Y")])
    return model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                construct.revised(
                    indicators=tuple(
                        indicator.revised(**indicator_changes) for indicator in construct.indicators
                    )
                )
                for construct in model.constructs
            ),
        )
    )


def test_recorded_replicate_becomes_a_compatible_panel(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, history = ArtifactStore("TEST"), StudyRepository("TEST")
    model = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[1]
            / "fixtures/models"
            / "prepare_simulation_data/recorded_replicate_becomes_a_compatible_panel_complete_test_model.json"
        ).read_text()
    )
    model_info = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    state = StudyState().with_artifacts([model_info])
    model_commit = history.append(
        applied_record(
            ModelEditResult(produced=[model_info]), seq=1, ts="2026-09-25T12:00:00Z", trace_ids=[]
        )
    ).commit_id
    times = (0.0, 0.5, 2.0)
    draws = np.array([[[91, 92], [93, 94], [95, 96]], [[2, 4], [3, 8], [6, 12]]], dtype=float)
    design = SimulationSpec(start=5, end=7)
    report = SimulationReport(
        model=GitRef(workspace_id="TEST", revision=model_info.revision, path="model.json"),
        design=design,
        time_origin=None,
        predictive=predictive_summary(model, draws),
        times=tuple(time + 5 for time in times),
        draws=2,
        seed=0,
        state_ids=tuple(numeric.state_ids(compile_model_fixture(model))),
        parameter_draws={"known_truth": "not-an-observation-array"},
        latent_paths="not-an-observation-array",
        observations=store.write_array(draws),
        observation_layout=simulation_layout(
            model, tuple(t + 5 for t in times), np.ones_like(draws, dtype=bool), store.write_array
        ),
    )
    simulation_record = applied_record(
        ModelSimulationResult(report=report), seq=2, ts="2026-09-25T12:01:00Z", trace_ids=[]
    )
    source_commit = history.append(simulation_record).commit_id
    # A later simulation must not replace the explicitly selected source.
    history.append(
        applied_record(
            ModelSimulationResult(
                report=report.revised(**{"observations": store.write_array(np.zeros_like(draws))})
            ),
            seq=3,
        )
    )
    source = SimulationReplicateRef(revision=source_commit, replicate=1)
    request = PrepareDataRequest(input=source)
    with pytest.raises(StudyLookupError, match="applied simulation commit"):
        run_async(
            run_action(
                "TEST",
                PrepareDataRequest(
                    input=SimulationReplicateRef(revision=model_commit, replicate=1)
                ),
                state,
            )
        )
    effects = run_async(run_action("TEST", request, state))
    # The runner stages the panel; the enclosing action owns scientific checks.
    assert {info.artifact_id for info in effects.produced} == {"panel"}
    panel_info = next(info for info in effects.produced if info.artifact_id == "panel")
    assert panel_info.derived_from == {}
    assert effects.simulation_source == source
    assert effects.n_observations == 6
    panel = store.read_parquet_file("panel", panel_info.revision, "panel.parquet")
    wide, _ = project_observation_data(
        panel, model_spec=compile_model_fixture(model), time_origin=None
    )
    np.testing.assert_allclose(wide["time"].to_numpy(), times)
    np.testing.assert_allclose(
        wide.select(numeric.observation_names(compile_model_fixture(model))).to_numpy(),
        draws[1],
        equal_nan=True,
    )
    assert panel["anchor_time"].min() == datetime(1970, 1, 1)
    assert set(panel["support_kind"]) == {"point"}

    history.append(applied_record(effects, seq=4, request=request))
    attempt = history.record(history.head()).record.attempt
    assert attempt.action == "prepare_data"
    assert attempt.outcome.status == "applied"
    assert attempt.outcome.result.simulation_source == source


@pytest.mark.parametrize("interval", [False, True])
def test_materialization_preserves_measurement_support_and_numeric_codes(interval):
    model = _model(
        **(
            {"aggregation": "mean", "observation_window": "1d"}
            if interval
            else {
                "aggregation": "last",
                "measurement_dtype": "categorical",
                "categorical_levels": ("a", "b", "c"),
            }
        )
    )
    values = np.array([[[np.nan, np.nan], [2, 2], [2, 2]]])
    if not interval:
        values[0, 0] = 2
    arrays = {}

    def write_array(value):
        key = str(len(arrays))
        arrays[key] = value
        return key

    from nof1_causal_lab.actions.simulation_summaries import summarize_simulation

    report = SimulationReport(
        model=GitRef(workspace_id="TEST", revision=git_oid(1), path="model.json"),
        design=SimulationSpec(end=2.5),
        time_origin=None,
        predictive=summarize_simulation(
            model,
            state_ids=tuple(item.id for item in model.constructs),
            variables=metadata_for_model(model).variables,
            latent_paths=np.zeros_like(values),
            observations=values,
            mask=np.isfinite(values),
            reference_latent_paths=None,
            reference_observations=None,
            fit_reliability="not_fitted",
        ),
        times=(0, 1, 2.5),
        draws=1,
        seed=0,
        state_ids=tuple(item.id for item in model.constructs),
        parameter_draws={},
        latent_paths="truth",
        observations="observations",
        observation_layout=SimulationObservationLayout(
            variables=metadata_for_model(model).variables,
            support_start_times=write_array(
                np.array([[np.nan, np.nan], [0, 0], [1.5, 1.5]])
                if interval
                else np.array([[0, 0], [1, 1], [2.5, 2.5]])
            ),
            support_end_times=write_array(
                np.array([[np.nan, np.nan], [1, 1], [2.5, 2.5]])
                if interval
                else np.array([[0, 0], [1, 1], [2.5, 2.5]])
            ),
            mask=write_array(np.isfinite(values)),
        ),
    )
    panel = prepare_simulation_panel(
        report, 0, read_array=lambda key: values if key == "observations" else arrays[key]
    )
    assert panel["value"].drop_nulls().to_list() == [2.0] * (4 if interval else 6)
    assert panel["anchor_time"].min() == datetime(1970, 1, 1)
    if interval:
        rows = panel.filter(pl.col("value").is_not_null())
        assert (rows["support_end"] - rows["support_start"]).to_list() == [timedelta(days=1)] * 4
        assert panel.filter(pl.col("value").is_null())["support_start"].null_count() == 2
    else:
        assert panel["support_start"].to_list() == panel["anchor_time"].to_list()
    from nof1_causal_lab.study.errors import StudyLookupError

    with pytest.raises(StudyLookupError, match="replicate"):
        prepare_simulation_panel(
            report, 1, read_array=lambda key: values if key == "observations" else arrays[key]
        )
    values[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite emissions"):
        prepare_simulation_panel(
            report, 0, read_array=lambda key: values if key == "observations" else arrays[key]
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"input": {"source": {"files": ["observations.csv"]}}},
        {"input": {"revision": git_oid(1), "replicate": -1}},
        {
            "input": {"revision": git_oid(1), "replicate": 0},
            "model_revision": git_oid(2),
        },
        {"input": {"revision": git_oid(1), "replicate": 0, "max_windows": 1}},
        {"input": "panel"},
    ],
)
def test_data_sources_require_one_unambiguous_origin(payload):
    with pytest.raises(ValidationError):
        PrepareDataRequest.model_validate(payload)
