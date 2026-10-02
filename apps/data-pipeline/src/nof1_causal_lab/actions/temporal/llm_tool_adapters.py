"""Tool adapters for generic Temporal LLM subroutines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from pydantic import BaseModel, TypeAdapter

from nof1_causal_lab.actions.extraction.contracts import ValidateExtractionsInput
from nof1_causal_lab.actions.ingestion.contracts import (
    ExecutePythonInput,
    ListFilesInput,
    ReadFileSampleInput,
    SubmitTableInput,
)
from nof1_causal_lab.actions.ingestion.tools import _execute_python_locally
from nof1_causal_lab.actions.temporal.llm_subroutine_storage import (
    read_subroutine_json,
    write_subroutine_json,
)
from nof1_causal_lab.actions.temporal.messages import MeasurementChunkContext
from nof1_causal_lab.study.errors import ArtifactWriteRejected, StudyLookupError
from nof1_causal_lab.utils import storage

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.actions.temporal.messages import (
        HarnessToolRequest,
        LLMToolExecutionInput,
        LLMToolSpec,
    )
    from nof1_causal_lab.json_types import JsonObject


def _validate_measurement_payload(
    *,
    context_ref: str,
    data: object,
) -> tuple[JsonObject | None, str]:
    from nof1_causal_lab.workers.schemas import validate_worker_output

    spec = read_subroutine_json(context_ref, MeasurementChunkContext)
    output, errors = validate_worker_output(
        data,
        spec.measurement_structure,
        spec.window_starts,
    )
    if errors:
        return None, "VALIDATION ERRORS:\n" + "\n".join(f"- {error}" for error in errors)
    if output is None:
        return None, "VALIDATION ERRORS:\n- validator returned no output"
    payload: JsonObject = output.model_dump(mode="json")
    return payload, "VALID"


class RawDataContext(BaseModel):
    extract_dir: str
    dataframe_ref: str
    cache_ref: str | None = None
    reused: bool = False


def _raw_data_context(context_ref: str) -> RawDataContext:
    return RawDataContext.model_validate(read_subroutine_json(context_ref))


def _read_raw_dataframe(dataframe_ref: str) -> pl.DataFrame:
    import polars as pl

    with storage.open_file(dataframe_ref, "rb") as file:
        return pl.read_ipc(file)


def _write_raw_dataframe(dataframe_ref: str, dataframe: pl.DataFrame) -> None:
    with storage.open_file(dataframe_ref, "wb") as file:
        dataframe.write_ipc(file)


def _execute_raw_data_list_files(context_ref: str, args: ListFilesInput) -> tuple[str, str | None]:
    from nof1_causal_lab.actions.ingestion.tools import _safe_resolve

    context = _raw_data_context(context_ref)
    extract_dir = Path(context.extract_dir).resolve()
    path = args.path or "."
    try:
        target = _safe_resolve(extract_dir, path)
    except StudyLookupError as exc:
        return str(exc), None

    if not target.exists():
        return f"Path not found: {path}", None
    if not target.is_dir():
        return f"Not a directory: {path}", None

    entries = []
    for item in sorted(target.iterdir()):
        rel = item.relative_to(extract_dir)
        if item.is_dir():
            n_children = sum(1 for _ in item.iterdir())
            entries.append(f"  [dir]  {rel}/  ({n_children} items)")
        else:
            size = item.stat().st_size
            if size < 1024:
                size_str = f"{size} B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f} MB"
            entries.append(f"  [file] {rel}  ({size_str})")

    if not entries:
        return f"Empty directory: {path}", None
    return "\n".join(entries), None


def _execute_raw_data_read_file_sample(
    context_ref: str, args: ReadFileSampleInput
) -> tuple[str, str | None]:
    from nof1_causal_lab.actions.ingestion.tools import _safe_resolve

    context = _raw_data_context(context_ref)
    extract_dir = Path(context.extract_dir).resolve()
    path = args.path
    n_lines = args.n_lines or 50
    try:
        target = _safe_resolve(extract_dir, path)
    except StudyLookupError as exc:
        return str(exc), None

    if not target.exists():
        return f"File not found: {path}", None
    if target.is_dir():
        return f"Is a directory, not a file: {path}", None

    suffix = target.suffix.lower()
    if suffix in (".xlsx", ".xls", ".parquet", ".feather", ".arrow"):
        size = target.stat().st_size
        return (
            f"Binary file: {target.name} ({size} bytes)\n"
            f"Type: {suffix}\n"
            f"Use pl.read_excel(Path(DATA_DIR) / '{path}') for Excel files\n"
            f"Use pl.read_parquet(Path(DATA_DIR) / '{path}') for Parquet files",
            None,
        )

    payload = target.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = payload.decode("latin-1")
    lines = text.splitlines()[:n_lines]
    total_lines = f"(showing first {n_lines} lines)" if len(lines) == n_lines else ""
    return f"File: {path} {total_lines}\n" + "\n".join(lines), None


def _execute_raw_data_python(context_ref: str, args: ExecutePythonInput) -> tuple[str, str | None]:
    context = _raw_data_context(context_ref)
    code = args.code
    output, result_df = _execute_python_locally(Path(context.extract_dir).resolve(), code)
    if result_df is not None:
        _write_raw_dataframe(context.dataframe_ref, result_df)
    return output, None


def _execute_raw_data_submit_table(
    context_ref: str,
    result_ref: str,
    args: SubmitTableInput,
) -> tuple[str, str | None]:
    import polars as pl
    import pyarrow as pa
    from pydantic import ValidationError

    from nof1_causal_lab.artifacts.raw_data import with_column_descriptions

    context = _raw_data_context(context_ref)
    dataframe_ref = context.dataframe_ref
    if not storage.exists(dataframe_ref):
        return (
            "No DataFrame available. Run execute_python first and assign your result to `result_df`.",
            None,
        )

    df = _read_raw_dataframe(dataframe_ref)
    if df.is_empty():
        return "DataFrame is empty (0 rows). Parse more data before submitting.", None

    try:
        col_descs = TypeAdapter(dict[str, str]).validate_json(args.column_descriptions_json)
    except ValidationError as exc:
        return (
            "column_descriptions_json must be a JSON object mapping column names to string "
            f"descriptions: {exc}",
            None,
        )

    if "timestamp" not in df.columns:
        return (
            "DataFrame must contain a Datetime column named 'timestamp' as the primary "
            "temporal axis. Rename or cast the time column in your execute_python code.",
            None,
        )
    if df.schema["timestamp"].base_type() not in (pl.Datetime, pl.Date):
        return (
            f"Column 'timestamp' has type {df.schema['timestamp']}; it must be Datetime "
            "or Date. Cast it in your execute_python code.",
            None,
        )

    try:
        table = with_column_descriptions(df.to_arrow(), col_descs)
    except ArtifactWriteRejected as exc:
        return str(exc), None

    table_ref = result_ref.removesuffix(".json") + ".arrow"
    with storage.open_file(table_ref, "wb") as file, pa.ipc.new_file(file, table.schema) as writer:
        writer.write_table(table)
    write_subroutine_json(result_ref, {"table_ref": table_ref})
    return "VALID", result_ref


async def execute_subroutine_tool(
    *,
    activity_input: LLMToolExecutionInput | HarnessToolRequest,
    tool: LLMToolSpec,
    args: object,
    result_ref: str,
) -> tuple[str, str | None]:
    if tool.executor == "measurement_validation":
        data = json.loads(ValidateExtractionsInput.model_validate(args).output_json)
        context_output, feedback = _validate_measurement_payload(
            context_ref=activity_input.subroutine.context_ref,
            data=data,
        )
        if context_output is not None:
            write_subroutine_json(result_ref, context_output)
            return feedback, result_ref
        return feedback, None

    if tool.executor == "raw_data_list_files":
        return _execute_raw_data_list_files(
            activity_input.subroutine.context_ref, ListFilesInput.model_validate(args)
        )
    if tool.executor == "raw_data_read_file_sample":
        return _execute_raw_data_read_file_sample(
            activity_input.subroutine.context_ref, ReadFileSampleInput.model_validate(args)
        )
    if tool.executor == "raw_data_execute_python":
        return _execute_raw_data_python(
            activity_input.subroutine.context_ref, ExecutePythonInput.model_validate(args)
        )
    if tool.executor == "raw_data_submit_table":
        return _execute_raw_data_submit_table(
            activity_input.subroutine.context_ref, result_ref, SubmitTableInput.model_validate(args)
        )
    assert_never(tool.executor)
