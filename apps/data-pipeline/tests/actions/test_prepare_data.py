"""One model-free preparation action retains extraction semantics and data findings."""

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from nof1_causal_lab.actions.contracts import PrepareDataRequest
from nof1_causal_lab.actions.data_checks import evaluate_data_checks
from nof1_causal_lab.actions.execution import plan_execution
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.artifacts.data_preparation import (
    DataPreparationSpec,
    DataVariableSpec,
    FilePreparationSpec,
)
from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact
from nof1_causal_lab.machine.artifacts import EpisodeState
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.temporal.measurement_activities import (
    finalize_measurements_activity,
    plan_measurements_activity,
)
from nof1_causal_lab.machine.temporal.messages import (
    ExtractionChunkResult,
    MeasurementsFinalizeInput,
    MeasurementsWorkflowInput,
)
from nof1_causal_lab.utils import storage
from tests.helpers import run_async

pytestmark = pytest.mark.contract


def test_preparation_without_model_combines_computed_and_semantic_workers(monkeypatch, tmp_path):
    from nof1_causal_lab.machine import store as store_module
    from nof1_causal_lab.utils import data

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))

    def no_model(*_args, **_kwargs):
        raise AssertionError("Preparation must not read a model")

    monkeypatch.setattr(store_module, "read_model", no_model)
    store = ArtifactStore("data-only")
    preparation = DataPreparationSpec(
        default_window="1d",
        context="Daily diary scoring",
        variables=(
            DataVariableSpec(
                id="indicator:steps",
                name="steps",
                measurement_dtype="count",
                aggregation="sum",
                how_to_measure="Total recorded steps",
                extraction_mode="computed",
                source_columns=("steps",),
                fill_null=0,
            ),
            DataVariableSpec(
                id="indicator:stress",
                name="stress",
                measurement_dtype="ordinal",
                aggregation="last",
                how_to_measure="Score the diary as low, medium or high stress",
                ordinal_levels=("low", "medium", "high"),
                source_columns=("diary",),
            ),
        ),
    )
    request = PrepareDataRequest(
        input=FilePreparationSpec(source={"files": ["diary.csv"]}, definition=preparation)
    )
    execution = plan_execution(request)
    assert execution.input_revisions == {}
    assert execution.operation.operation_id == "measurements"
    raw = store.write_artifact(
        "raw_data",
        derived_from={},
        produced_by="run:raw_data",
        parquet_files={
            "raw.parquet": pl.DataFrame(
                {
                    "timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 3)],
                    "steps": [4, 8],
                    "diary": ["calm", "overwhelmed"],
                }
            )
        },
    )
    state = EpisodeState().with_artifacts([raw])
    plan = run_async(
        plan_measurements_activity(
            MeasurementsWorkflowInput(
                workspace_id="data-only",
                seq=1,
                state=state,
                preparation=execution.operation.preparation,
            )
        )
    )
    assert plan.pins == {"raw_data": raw.revision}
    assert plan.chunks
    result_path = str(tmp_path / "worker-result.json")
    storage.write_text(
        result_path,
        json.dumps(
            {
                "dataframe": [
                    {
                        "indicator_id": "indicator:stress",
                        "value": "2",
                        "timestamp": "2026-01-03T00:00:00",
                    },
                ]
            }
        ),
    )
    results = [
        ExtractionChunkResult(
            worker_id=chunk.worker_id,
            n_windows=chunk.n_windows,
            status="completed" if index == 0 else "failed",
            n_extractions=1 if index == 0 else 0,
            result_ref=result_path if index == 0 else None,
        )
        for index, chunk in enumerate(plan.chunks)
    ]
    effects = run_async(
        finalize_measurements_activity(
            MeasurementsFinalizeInput(
                workspace_id="data-only",
                state=state,
                run_id=plan.run_id,
                plan_ref=plan.plan_ref,
                pins=plan.pins,
                chunk_results=results,
            )
        )
    )
    effects = evaluate_data_checks("data-only", state, effects)
    assert {item.artifact_id for item in effects.produced} == {"panel", "data_profile"}
    assert effects.checks is None
    panel = next(item for item in effects.produced if item.artifact_id == "panel")
    assert panel.derived_from == {"raw_data": raw.revision}
    observations = store.read_parquet_file("panel", panel.revision, "panel.parquet")
    assert observations.filter(pl.col("indicator_id") == "indicator:steps")["value"].to_list() == [
        4,
        0,
        8,
    ]
    assert observations.filter(pl.col("indicator_id") == "indicator:stress")["value"].to_list() == [
        2
    ]
    metadata = store.read_json_file("panel", panel.revision, "metadata.json")
    assert (
        metadata["preparation"]["variables"][1]["how_to_measure"]
        == preparation.variables[1].how_to_measure
    )
    assert metadata["variables"][1]["ordinal_levels"] == ["low", "medium", "high"]
    profile_ref = next(item for item in effects.produced if item.artifact_id == "data_profile")
    profile = DataProfileArtifact.model_validate(
        store.read_json_file("data_profile", profile_ref.revision, "data_profile.json")
    )
    assert set(profile.indicators) == {"indicator:steps", "indicator:stress"}
    labels = completion_messages(
        "data-only", "prepare_data", effects.produced, effects.diagnostics, datetime.now(UTC)
    )
    assert "DATA_QUALITY_FINDINGS" in {label.label for label in labels}
    assert all(set(label.model_dump()) == {"timestamp", "level", "label"} for label in labels)


def test_declared_categorical_codes_survive_absent_categories():
    from nof1_causal_lab.flows.transitions.extraction.materialization import materialize_panel
    from nof1_causal_lab.utils.data import annotate_observation_rows

    preparation = DataPreparationSpec(
        default_window="1d",
        variables=(
            DataVariableSpec(
                id="indicator:place",
                name="place",
                measurement_dtype="categorical",
                aggregation="last",
                categorical_levels=("home", "work", "outside"),
                how_to_measure="Last stated location",
            ),
        ),
    )
    from pydantic import TypeAdapter

    from nof1_causal_lab.artifacts.measurements import ObservationRecord

    context = preparation.extraction_context()
    rows = annotate_observation_rows(
        pl.DataFrame(
            {"indicator_id": ["indicator:place"], "value": ["outside"], "timestamp": ["2026-01-01"]}
        ),
        context,
    ).to_dicts()
    assert materialize_panel(TypeAdapter(list[ObservationRecord]).validate_python(rows), context)[
        "value"
    ].to_list() == [2]
