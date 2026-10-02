"""Preparation reuses effective LLM requests while spans and computation rerun."""

import io
import json
from datetime import UTC, date, datetime
from types import SimpleNamespace
from uuid import uuid4

import polars as pl
import pyarrow as pa
import pytest
from pydantic import TypeAdapter

from nof1_causal_lab.actions.temporal.ingestion_activities import (
    finalize_ingestion_activity,
    plan_ingestion_activity,
)
from nof1_causal_lab.actions.temporal.llm_subroutine_storage import read_subroutine_json
from nof1_causal_lab.actions.temporal.measurement_activities import (
    finalize_extraction_chunk_activity,
    plan_measurements_activity,
)
from nof1_causal_lab.actions.temporal.messages import (
    ExtractionChunkFinalizeInput,
    IngestionFinalizeInput,
    IngestionWorkflowInput,
    MeasurementChunkContext,
    MeasurementsWorkflowInput,
)
from nof1_causal_lab.artifacts.data_preparation import (
    DataPreparationSpec,
    DataVariableSpec,
    FilePreparationSpec,
    FileSourceRef,
)
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils import data, storage
from nof1_causal_lab.utils.aggregations import compute_indicators
from nof1_causal_lab.workers.schemas import ExtractionRow
from tests.helpers import run_async

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("code", ["PreconditionFailed", "412", "AccessDenied"])
def test_remote_cache_uses_the_conditional_write_winner_only(code, monkeypatch):
    from botocore.exceptions import ClientError

    from nof1_causal_lab.utils.content_cache import publish

    def conflict(path, value, *, mode):
        assert (path, value, mode) == ("s3://cache/key", b"loser", "create")
        raise OSError("conditional create failed") from ClientError(
            {"Error": {"Code": code}}, "PutObject"
        )

    monkeypatch.setattr(storage, "is_remote", lambda: True)
    monkeypatch.setattr(storage, "get_fs", lambda: SimpleNamespace(pipe_file=conflict))
    monkeypatch.setattr(storage, "open_file", lambda _path, _mode: io.BytesIO(b"winner"))
    if code == "AccessDenied":
        with pytest.raises(OSError, match="conditional create failed"):
            publish("s3://cache/key", b"loser")
    else:
        assert publish("s3://cache/key", b"loser") == b"winner"


def _variable(name):
    return DataVariableSpec(
        id=f"indicator:{name}",
        name=name,
        measurement_dtype="continuous",
        aggregation="last",
        source_columns=(name,),
        how_to_measure="Read the daily score",
    )


def test_extraction_cache_is_shared_and_only_changed_effective_requests_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    raw = pl.DataFrame(
        {"timestamp": [datetime(2026, 1, day) for day in (1, 2, 3)], "x": [1, 2, 3], "y": [4, 5, 6]}
    )

    def plan(workspace, variables, end, context="Daily scores"):
        store = ArtifactStore(workspace)
        artifact = store.write_artifact(
            "raw_data",
            derived_from={},
            produced_by="prepare_data",
            parquet_files={"raw.parquet": raw},
        )
        return run_async(
            plan_measurements_activity(
                MeasurementsWorkflowInput(
                    workspace_id=workspace,
                    seq=1,
                    attempt_id=uuid4(),
                    raw_data_revision=artifact.revision,
                    preparation=FilePreparationSpec(
                        source={"files": ["scores.csv"], "start": "2026-01-01", "end": end},
                        definition=DataPreparationSpec(
                            default_window="1d", context=context, variables=variables
                        ),
                    ),
                )
            )
        )

    first = plan("first", (_variable("x"),), "2026-01-03")
    assert len(first.chunks) == 2
    for chunk in first.chunks:
        spec = read_subroutine_json(chunk.spec_ref, MeasurementChunkContext)
        assert chunk.n_windows == 1
        assert len(spec.measurement_structure.indicators) == 1
        result_ref = chunk.spec_ref + ".result"
        storage.write_text(
            result_ref,
            json.dumps(
                {
                    "extractions": [
                        {
                            "indicator_id": "indicator:x",
                            "value": 7,
                            "window_start": spec.window_starts[0],
                        }
                    ]
                }
            ),
        )
        request = ExtractionChunkFinalizeInput(
            workspace_id="first",
            run_id=first.run_id,
            worker_id=chunk.worker_id,
            attempt=1,
            n_windows=1,
            n_llm_calls=1,
            spec_ref=chunk.spec_ref,
            result_ref=result_ref,
            conversation_ref="unused",
        )
        completed = run_async(finalize_extraction_chunk_activity(request))
        # A concurrent later validated answer cannot replace the first one.
        storage.write_text(
            result_ref, storage.read_text(result_ref).replace('"value": 7', '"value": 99')
        )
        again = run_async(finalize_extraction_chunk_activity(request))
        assert (
            storage.read_json(again.result_ref)["dataframe"]
            == storage.read_json(completed.result_ref)["dataframe"]
        )
        assert (
            TypeAdapter[list[ExtractionRow]](list[ExtractionRow]).validate_python(
                storage.read_json(again.result_ref)["dataframe"]
            )[0]["value"]
            == "7"
        )
    same = plan("second", (_variable("x"),), "2026-01-03")
    assert all(chunk.cached_result_ref for chunk in same.chunks)
    expanded = plan("third", (_variable("x"), _variable("y")), "2026-01-04")
    assert len(expanded.chunks) == 6
    assert sum(chunk.cached_result_ref is not None for chunk in expanded.chunks) == 2
    changed = plan(
        "fourth", (_variable("x"),), "2026-01-03", context="Use a different interpretation"
    )
    assert all(chunk.cached_result_ref is None for chunk in changed.chunks)

    from nof1_causal_lab.actions.temporal import preparation_cache

    with monkeypatch.context() as policy:
        policy.setattr(preparation_cache, "EXTRACTION_POLICY_VERSION", "changed-extraction")
        assert all(
            chunk.cached_result_ref is None
            for chunk in plan("policy", (_variable("x"),), "2026-01-03").chunks
        )
    original_open = storage.open_file

    def evict(path, mode="r", **kwargs):
        if ".preparation-cache" in str(path) and mode == "rb":
            from pathlib import Path

            Path(path).unlink()
        return original_open(path, mode, **kwargs)

    with monkeypatch.context() as eviction:
        eviction.setattr(storage, "open_file", evict)
        assert all(
            chunk.cached_result_ref is None
            for chunk in plan("evicted", (_variable("x"),), "2026-01-03").chunks
        )


def test_ingestion_reuse_preserves_arrow_metadata_and_source_order(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    frame = pl.DataFrame({"timestamp": [datetime(2026, 1, 1)], "score": [2.0]}).to_arrow()
    table = frame.cast(
        pa.schema(
            [field.with_metadata({b"description": field.name.encode()}) for field in frame.schema]
        )
    )

    def plan(workspace, files=("a.csv", "b.csv")):
        for filename in files:
            path = tmp_path / workspace / "input" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("timestamp,score\n2026-01-01,2\n")
        return run_async(
            plan_ingestion_activity(
                IngestionWorkflowInput(
                    workspace_id=workspace,
                    seq=1,
                    attempt_id=uuid4(),
                    source={"files": files},
                )
            )
        )

    first = plan("first")
    assert first.cached_result_ref is None
    result_ref = str(tmp_path / "validated.json")
    table_ref = str(tmp_path / "validated.arrow")
    with pa.OSFile(table_ref, "wb") as stream, pa.ipc.new_file(stream, table.schema) as writer:
        writer.write_table(table)
    storage.write_text(result_ref, json.dumps({"table_ref": table_ref}))
    run_async(
        finalize_ingestion_activity(
            IngestionFinalizeInput(
                workspace_id="first",
                context_ref=first.context_ref,
                result_ref=result_ref,
            )
        )
    )
    reused = plan("second")
    assert reused.cached_result_ref is not None
    effects = run_async(
        finalize_ingestion_activity(
            IngestionFinalizeInput(
                workspace_id="second",
                context_ref=reused.context_ref,
                result_ref=reused.cached_result_ref,
            )
        )
    )
    assert effects.ingestion_reused is True
    saved = ArtifactStore("second").read_parquet_table(
        "raw_data", effects.produced[0].revision, "raw.parquet"
    )
    assert saved.equals(table, check_metadata=True)
    assert plan("reordered", ("b.csv", "a.csv")).cached_result_ref is None

    from nof1_causal_lab.actions.temporal import preparation_cache

    with monkeypatch.context() as policy:
        policy.setattr(preparation_cache, "EXTRACTION_POLICY_VERSION", "changed-extraction")
        assert plan("other-policy").cached_result_ref is not None
        policy.setattr(preparation_cache, "INGESTION_POLICY_VERSION", "changed-ingestion")
        assert plan("ingestion-policy").cached_result_ref is None
    original_open = storage.open_file

    def evict(path, mode="r", **kwargs):
        if ".preparation-cache" in str(path) and mode == "rb":
            from pathlib import Path

            Path(path).unlink()
        return original_open(path, mode, **kwargs)

    monkeypatch.setattr(storage, "open_file", evict)
    assert plan("evicted").cached_result_ref is None


def test_span_keeps_complete_windows_and_never_fills_from_excluded_history():
    raw = pl.DataFrame({"timestamp": [datetime(2026, 1, 1), datetime(2026, 1, 3)], "x": [9, 2]})
    variable = _variable("x").revised(extraction_mode="computed", fill_null="forward")
    context = FilePreparationSpec(
        source=FileSourceRef(files=("source.csv",), start=date(2026, 1, 2), end=date(2026, 1, 4)),
        definition=DataPreparationSpec(default_window="1d", variables=(variable,)),
    ).extraction_context()
    bounded = compute_indicators(raw, context, "timestamp")
    assert bounded["timestamp"].to_list() == ["2026-01-02T00:00:00", "2026-01-03T00:00:00"]
    assert bounded["value"].to_list() == [None, "2"]
    from nof1_causal_lab.actions.extraction.planning import prepare_semantic_chunks

    semantic_context = FilePreparationSpec(
        source=context.source,
        definition=DataPreparationSpec(default_window="1d", variables=(_variable("x"),)),
    ).extraction_context()
    texts, windows, _, empty_output = prepare_semantic_chunks(
        raw_df=raw,
        measurement_structure=semantic_context,
        time_col="timestamp",
        max_events_per_window=300,
    )
    assert windows == [["2026-01-03T00:00:00"]]
    assert len(texts) == 1
    assert empty_output.to_dataframe().to_dicts() == [
        {"indicator_id": "indicator:x", "timestamp": "2026-01-02T00:00:00", "value": None}
    ]
    from nof1_causal_lab.utils.observation_rows import prepared_time_origin

    assert prepared_time_origin(pl.DataFrame(), date(2026, 1, 2)) == datetime(
        2026, 1, 2, tzinfo=UTC
    )


@pytest.mark.parametrize(("aggregation", "offset"), [("first", 0), ("last", 2)])
def test_multiday_span_and_empty_semantic_windows_materialize_without_requests(
    tmp_path, monkeypatch, aggregation, offset
):
    from nof1_causal_lab.actions.temporal.measurement_activities import (
        finalize_measurements_activity,
    )
    from nof1_causal_lab.actions.temporal.messages import MeasurementsFinalizeInput
    from nof1_causal_lab.study.lineage import read_data_metadata

    monkeypatch.setattr(data, "_DATA_URI", str(tmp_path))
    raw = pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, day) for day in (1, 3, 5, 9)],
            "x": [9, None, 2, 12],
            "unselected": [99, 99, 99, 99],
        }
    )
    store = ArtifactStore("empty")
    artifact = store.write_artifact(
        "raw_data", derived_from={}, produced_by="prepare_data", parquet_files={"raw.parquet": raw}
    )
    variable = _variable("x").revised(aggregation=aggregation)
    preparation = FilePreparationSpec(
        source={"files": ["scores.csv"], "start": "2026-01-02", "end": "2026-01-10"},
        definition=DataPreparationSpec(default_window="2d", variables=(variable,)),
    )
    plan = run_async(
        plan_measurements_activity(
            MeasurementsWorkflowInput(
                workspace_id="empty",
                seq=1,
                attempt_id=uuid4(),
                raw_data_revision=artifact.revision,
                preparation=preparation,
            )
        )
    )
    assert len(plan.chunks) == 1  # All-null Jan 3 and no-row Jan 7 never become LLM requests.
    chunk = plan.chunks[0]
    spec = read_subroutine_json(chunk.spec_ref, MeasurementChunkContext)
    assert spec.window_starts == ["2026-01-05T00:00:00"]
    result_ref = chunk.spec_ref + ".result"
    storage.write_text(
        result_ref,
        json.dumps(
            {
                "extractions": [
                    {
                        "indicator_id": "indicator:x",
                        "window_start": spec.window_starts[0],
                        "value": 2,
                    }
                ]
            }
        ),
    )
    result = run_async(
        finalize_extraction_chunk_activity(
            ExtractionChunkFinalizeInput(
                workspace_id="empty",
                run_id=plan.run_id,
                worker_id=chunk.worker_id,
                attempt=1,
                n_windows=1,
                n_llm_calls=0,
                spec_ref=chunk.spec_ref,
                result_ref=result_ref,
                conversation_ref="unused",
            )
        )
    )
    effects = run_async(
        finalize_measurements_activity(
            MeasurementsFinalizeInput(
                workspace_id="empty",
                run_id=plan.run_id,
                pins=plan.pins,
                plan_ref=plan.plan_ref,
                chunk_results=[result],
            )
        )
    )
    revision = effects.produced[0].revision
    panel = store.read_parquet_file("panel", revision, "panel.parquet")
    assert panel["value"].to_list() == [None, 2.0, None]
    assert panel["anchor_time"].to_list() == [datetime(2026, 1, day + offset) for day in (3, 5, 7)]
    assert panel["support_start"].to_list() == [datetime(2026, 1, day) for day in (3, 5, 7)]
    assert panel["support_end"].to_list() == [datetime(2026, 1, day) for day in (5, 7, 9)]
    assert read_data_metadata(store, revision).time_origin == datetime(2026, 1, 2, tzinfo=UTC)
    assert len(list((tmp_path / ".preparation-cache/measurement_extraction").iterdir())) == 1

    computed_variable = DataVariableSpec(
        id=variable.id,
        name=variable.name,
        measurement_dtype=variable.measurement_dtype,
        aggregation=variable.aggregation,
        observation_window=variable.observation_window,
        how_to_measure=variable.how_to_measure,
        source_columns=variable.source_columns,
        extraction_mode="computed",
    )
    computed_context = FilePreparationSpec(
        source=preparation.source,
        definition=DataPreparationSpec(default_window="2d", variables=(computed_variable,)),
    ).extraction_context()
    computed = compute_indicators(raw, computed_context, "timestamp")
    from nof1_causal_lab.utils.observation_rows import annotate_observation_rows

    rows = annotate_observation_rows(computed, preparation.definition.observation_schema())
    assert rows["anchor_time"].to_list() == [value.isoformat() for value in panel["anchor_time"]]
    assert computed["value"].to_list() == [None, "2", None]
