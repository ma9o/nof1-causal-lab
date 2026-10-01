"""One model-free preparation action retains extraction semantics and data findings."""

import json
from datetime import UTC, datetime
from uuid import uuid4

import polars as pl
import pytest

from nof1_causal_lab.actions.contracts import PrepareDataRequest
from nof1_causal_lab.actions.data_checks import evaluate_data_checks
from nof1_causal_lab.actions.messages import completion_messages
from nof1_causal_lab.actions.temporal.measurement_activities import (
    finalize_measurements_activity,
    plan_measurements_activity,
)
from nof1_causal_lab.actions.temporal.messages import (
    ExtractionChunkResult,
    MeasurementsFinalizeInput,
    MeasurementsWorkflowInput,
)
from nof1_causal_lab.artifacts.data_preparation import (
    DataPreparationSpec,
    DataVariableSpec,
    FilePreparationSpec,
)
from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact
from nof1_causal_lab.study.state import StudyState
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils import storage
from tests.helpers import run_async

pytestmark = pytest.mark.contract


def test_preparation_without_model_combines_computed_and_semantic_workers(monkeypatch, tmp_path):
    from nof1_causal_lab.study import store as store_module
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
    assert isinstance(request.input, FilePreparationSpec)
    raw = store.write_artifact(
        "raw_data",
        derived_from={},
        produced_by="prepare_data",
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
    state = StudyState().with_artifacts([raw])
    plan = run_async(
        plan_measurements_activity(
            MeasurementsWorkflowInput(
                workspace_id="data-only",
                seq=1,
                attempt_id=uuid4(),
                raw_data_revision=raw.revision,
                preparation=request.input,
            )
        )
    )
    assert plan.pins == {"raw_data": raw.revision}
    scores = {"2026-01-01T00:00:00": "0", "2026-01-03T00:00:00": "2"}
    assert len(plan.chunks) == len(scores)
    results = []
    for chunk in plan.chunks:
        spec = json.loads(storage.read_text(chunk.spec_ref))
        assert chunk.n_windows == 1
        (window_start,) = spec["window_starts"]
        (indicator,) = spec["measurement_structure"]["indicators"]
        assert indicator["id"] == "indicator:stress"
        result_path = str(tmp_path / f"worker-{chunk.worker_id}-result.json")
        storage.write_text(
            result_path,
            json.dumps(
                {
                    "dataframe": [
                        {
                            "indicator_id": indicator["id"],
                            "value": scores[window_start],
                            "timestamp": window_start,
                        },
                    ]
                }
            ),
        )
        results.append(
            ExtractionChunkResult(
                worker_id=chunk.worker_id,
                n_windows=chunk.n_windows,
                status="completed",
                n_extractions=1,
                result_ref=result_path,
            )
        )
    effects = run_async(
        finalize_measurements_activity(
            MeasurementsFinalizeInput(
                workspace_id="data-only",
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
        0,
        None,
        2,
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


@pytest.mark.parametrize(
    ("value", "expected_code"),
    [("outside", 2), ("Outside", 2), (" outside ", 2), ("park", None)],
)
def test_declared_categorical_codebook_validates_and_encodes_normalized_labels(
    value, expected_code
):
    from nof1_causal_lab.actions.extraction.materialization import materialize_panel
    from nof1_causal_lab.utils.data import annotate_observation_rows
    from nof1_causal_lab.workers.schemas import validate_worker_output

    preparation = DataPreparationSpec(
        default_window="1d",
        variables=(
            DataVariableSpec(
                id="indicator:place",
                name="place",
                measurement_dtype="categorical",
                aggregation="last",
                categorical_levels=("home", "work", " Outside "),
                how_to_measure="Last stated location",
            ),
        ),
    )
    from pydantic import TypeAdapter

    from nof1_causal_lab.artifacts.measurements import ObservationRecord

    context = preparation.extraction_context()
    output, errors = validate_worker_output(
        {
            "extractions": [
                {
                    "indicator_id": "indicator:place",
                    "value": value,
                    "window_start": "2026-01-01",
                }
            ]
        },
        context,
        expected_window_starts=["2026-01-01"],
    )
    if expected_code is None:
        assert output is None
        assert len(errors) == 1
        assert "outside the codebook" in errors[0]
        return
    assert errors == []
    assert output is not None
    rows = annotate_observation_rows(output.to_dataframe(), context).to_dicts()
    assert materialize_panel(TypeAdapter(list[ObservationRecord]).validate_python(rows), context)[
        "value"
    ].to_list() == [expected_code]
