"""Synthetic observations compose with the normal panel, provenance and fitting contracts."""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from nof1_causal_lab.actions.contracts import FitRequest, PrepareDataRequest
from nof1_causal_lab.actions.execution import plan_execution
from nof1_causal_lab.actions.prepare_data import prepare_simulation_panel
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.data_preparation import SimulationReplicateRef
from nof1_causal_lab.artifacts.identity import GitRef
from nof1_causal_lab.artifacts.simulation import (
    SimulationReport,
    SimulationSpec,
)
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.runners import execute_transition
from nof1_causal_lab.machine.selection import resolve_input_pins
from nof1_causal_lab.machine.store import ArtifactStore, TransitionRecord
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.runtime import project_observation_data
from tests.data_fixtures import predictive_summary, simulation_layout
from tests.git_fixtures import git_oid
from tests.helpers import complete_test_model, make_model, run_async

pytestmark = pytest.mark.contract


def _model(**indicator_changes):
    model = make_model(["X", "Y"], [("X", "Y")])
    return model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                construct.model_copy(
                    update={
                        "indicators": tuple(
                            indicator.model_copy(update=indicator_changes)
                            for indicator in construct.indicators
                        )
                    }
                )
                for construct in model.constructs
            ),
        )
    )


def test_recorded_replicate_becomes_a_compatible_panel(tmp_path, monkeypatch):
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, history = ArtifactStore("TEST"), StudyRepository("TEST")
    model = complete_test_model(_model(aggregation="last"))
    model_info = store.write_artifact(
        "model",
        derived_from={},
        produced_by="write:model",
        json_files={"model.json": model.model_dump(mode="json")},
    )
    state = EpisodeState().with_artifacts([model_info])
    model_commit = history.append(
        TransitionRecord(
            seq=1,
            ts="2026-09-25T12:00:00Z",
            action="edit_model",
            inputs={"expected_revision": None},
            status="applied",
            produced=[model_info],
            trace_ids=[],
            resume=None,
        )
    )
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
        state_ids=tuple(numeric.state_ids(model)),
        parameter_draws={"known_truth": "not-an-observation-array"},
        latent_paths="not-an-observation-array",
        observations=store.write_array(draws),
        observation_layout=simulation_layout(
            model, tuple(t + 5 for t in times), np.ones_like(draws, dtype=bool), store.write_array
        ),
    )
    simulation_record = TransitionRecord(
        seq=2,
        ts="2026-09-25T12:01:00Z",
        action="simulate",
        operation_id="simulate",
        inputs={},
        status="applied",
        diagnostics={"report": report.model_dump(mode="json")},
        trace_ids=[],
        resume=None,
    )
    source_commit = history.append(simulation_record)
    # A later simulation must not replace the explicitly selected source.
    history.append(
        simulation_record.model_copy(
            update={
                "seq": 3,
                "diagnostics": {
                    "report": report.model_copy(
                        update={"observations": store.write_array(np.zeros_like(draws))}
                    ).model_dump(mode="json")
                },
            }
        )
    )
    source = SimulationReplicateRef(revision=source_commit, replicate=1)
    request = PrepareDataRequest(input=source)
    command = plan_execution(request)
    assert command.operation.operation_id == "simulated_measurements"
    with pytest.raises(ValueError, match="applied simulation commit"):
        run_async(
            execute_transition(
                "TEST",
                command.operation.model_copy(
                    update={"source": source.model_copy(update={"revision": model_commit})}
                ),
                state,
            )
        )
    effects = run_async(execute_transition("TEST", command.operation, state))
    # The operation stages the panel; the enclosing action owns scientific checks.
    assert {info.artifact_id for info in effects.produced} == {"panel"}
    panel_info = next(info for info in effects.produced if info.artifact_id == "panel")
    assert panel_info.derived_from == {}
    assert effects.diagnostics["simulation_source"] == source.model_dump(mode="json")
    assert effects.diagnostics["n_observations"] == 6
    panel = store.read_parquet_file("panel", panel_info.revision, "panel.parquet")
    wide, _ = project_observation_data(panel, model_spec=model, time_origin=None)
    np.testing.assert_allclose(wide["time"].to_numpy(), times)
    np.testing.assert_allclose(
        wide.select(numeric.observation_names(model)).to_numpy(), draws[1], equal_nan=True
    )
    assert panel["anchor_time"].min() == datetime(1970, 1, 1)
    assert set(panel["support_kind"]) == {"point"}

    # Fitting uses the usual panel selection; only observation rows enter it.
    fit = plan_execution(
        FitRequest(model_revision=model_info.revision, panel_revision=panel_info.revision)
    )
    assert resolve_input_pins(
        store,
        state.with_artifacts(effects.produced),
        transition_spec("posterior"),
        fit.input_revisions,
    ) == {"model": model_info.revision, "panel": panel_info.revision}
    history.append(
        simulation_record.model_copy(
            update={
                "seq": 4,
                "action": "prepare_data",
                "operation_id": command.operation.operation_id,
                "inputs": request.model_dump(mode="json", exclude={"action"}),
                "produced": effects.produced,
                "diagnostics": effects.diagnostics,
            }
        )
    )
    assert history.record(history.head()).diagnostics["simulation_source"] == source.model_dump(
        mode="json"
    )


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

    report = SimulationReport(
        model=GitRef(workspace_id="TEST", revision=git_oid(1), path="model.json"),
        design=SimulationSpec(end=2.5),
        time_origin=None,
        predictive=predictive_summary(model, values),
        times=(0, 1, 2.5),
        draws=1,
        seed=0,
        state_ids=tuple(numeric.state_ids(model)),
        parameter_draws={},
        latent_paths="truth",
        observations="observations",
        observation_layout=simulation_layout(model, (0, 1, 2.5), np.isfinite(values), write_array),
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
    with pytest.raises(ValueError, match="replicate"):
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
