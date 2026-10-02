"""Temporal activities that stage uploaded files and publish the ingested raw-data table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter
from temporalio import activity

from nof1_causal_lab.actions.errors import execution_failure_handler
from nof1_causal_lab.actions.temporal.activity_errors import (
    as_non_retryable_application_error,
)
from nof1_causal_lab.actions.temporal.backend_config import llm_backend_config
from nof1_causal_lab.actions.temporal.llm_tool_adapters import RawDataContext
from nof1_causal_lab.actions.temporal.messages import (
    IngestionFinalizeInput,
    IngestionPlan,
    IngestionWorkflowInput,
)
from nof1_causal_lab.study.artifact_files import parquet_filename
from nof1_causal_lab.study.records import DataPreparationResult
from nof1_causal_lab.study.store import ArtifactStore
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject


def _raw_data_root(workspace_id: str, run_id: str) -> str:
    return storage.join(data_module.scratch_run_dir(workspace_id, run_id), "ingestion")


def _write_raw_data_json(path: str, value: object) -> None:
    storage.write_text(path, json.dumps(value))


def _read_raw_data_json(path: str) -> JsonObject:
    return storage.read_json(path)


@activity.defn
async def plan_ingestion_activity(
    activity_input: IngestionWorkflowInput,
) -> IngestionPlan:
    from nof1_causal_lab.actions.ingestion.flow import (
        _prepare_raw_input,
    )
    from nof1_causal_lab.utils.config import get_config

    run_id = f"seq-{activity_input.seq:06d}"
    root = _raw_data_root(activity_input.workspace_id, run_id)
    upload_dir = storage.join(root, "upload")
    extract_dir = storage.join(root, "input")
    storage.rm_tree(upload_dir)
    storage.rm_tree(extract_dir)
    storage.makedirs(upload_dir)
    storage.makedirs(extract_dir)

    for index, raw_name in enumerate(activity_input.source.files):
        raw_storage_path = storage.join(
            data_module.input_dir(activity_input.workspace_id), raw_name
        )
        local_raw = Path(upload_dir) / raw_name
        with storage.open_file(raw_storage_path, "rb") as uploaded:
            local_raw.write_bytes(uploaded.read())
        # Separate archives so identically named members cannot overwrite one another.
        _prepare_raw_input(local_raw, Path(extract_dir) / str(index))

    context_ref = storage.join(root, "context.json")
    _write_raw_data_json(
        context_ref,
        {
            "extract_dir": extract_dir,
            "dataframe_ref": storage.join(root, "latest-dataframe.ipc"),
        },
    )

    config = get_config()
    max_tool_turns = config.ingestion.max_tool_turns
    llm = llm_backend_config(config.ingestion.llm, config.llm, max_tool_turns)
    from nof1_causal_lab.actions.temporal.preparation_cache import preparation_cache_path
    from nof1_causal_lab.utils.content_cache import read

    manifest = [
        {
            "path": str(path.relative_to(extract_dir)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(Path(extract_dir).rglob("*"))
        if path.is_file()
    ]
    cache_ref = preparation_cache_path(
        "raw_data_ingestion",
        context_ref,
        llm,
        max_tool_turns,
        {"files": activity_input.source.files, "staged": manifest},
    )
    context = dict(_read_raw_data_json(context_ref))
    context["cache_ref"] = cache_ref
    cached = read(cache_ref)
    context["reused"] = cached is not None
    cached_result_ref = None
    if cached is not None:
        table_ref = storage.join(root, "reused.arrow")
        with storage.open_file(table_ref, "wb") as output:
            output.write(cached)
        cached_result_ref = storage.join(root, "reused.json")
        _write_raw_data_json(cached_result_ref, {"table_ref": table_ref})
    _write_raw_data_json(context_ref, context)
    return IngestionPlan(
        workspace_id=activity_input.workspace_id,
        run_id=run_id,
        context_ref=context_ref,
        llm=llm,
        max_tool_turns=max_tool_turns,
        cached_result_ref=cached_result_ref,
    )


@activity.defn
@execution_failure_handler
async def finalize_ingestion_activity(
    activity_input: IngestionFinalizeInput,
) -> DataPreparationResult:
    import pyarrow as pa

    from nof1_causal_lab.utils.content_cache import publish

    try:
        result = _read_raw_data_json(activity_input.result_ref)
        context = RawDataContext.model_validate(_read_raw_data_json(activity_input.context_ref))
        with storage.open_file(TypeAdapter(str).validate_python(result["table_ref"]), "rb") as file:
            payload = file.read()
        # Terminal submit_table validated this Arrow payload, including field metadata.
        assert context.cache_ref is not None
        payload = publish(context.cache_ref, payload)
        table = pa.ipc.open_file(pa.BufferReader(payload)).read_all()

        store = ArtifactStore(activity_input.workspace_id)
        produced = [
            store.write_artifact(
                "raw_data",
                derived_from={},
                produced_by="prepare_data",
                parquet_files={parquet_filename("raw_data", "raw"): table},
            )
        ]
        return DataPreparationResult(produced=tuple(produced), ingestion_reused=context.reused)
    except Exception as exc:
        raise as_non_retryable_application_error(exc) from exc


INGESTION_ACTIVITIES = [
    plan_ingestion_activity,
    finalize_ingestion_activity,
]
