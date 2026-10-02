"""Shared ingestion tool helpers."""

from __future__ import annotations

import contextlib
import csv
import datetime
import io
import json
import math
import re
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from nof1_causal_lab.actions.errors import execution_failure_handler
from nof1_causal_lab.study.errors import StudyLookupError

if TYPE_CHECKING:
    import polars as pl


def _safe_resolve(base: Path, user_path: str) -> Path:
    """Resolve a user-provided path within the staged input directory."""
    base_resolved = base.resolve()
    resolved = (base / user_path).resolve()
    if not resolved.is_relative_to(base_resolved):
        raise StudyLookupError(f"Path traversal blocked: {user_path}")
    return resolved


_RAW_EXEC_NAMESPACE_NAMES = frozenset(
    {
        "pl",
        "polars",
        "csv",
        "json",
        "Path",
        "datetime",
        "re",
        "math",
        "io",
        "DATA_DIR",
    }
)


def _raw_python_output_prefix(stdout: str, stderr: str) -> str:
    parts = [part.strip() for part in (stdout, stderr) if part.strip()]
    if not parts:
        return ""
    return "\n".join(parts) + "\n\n"


@execution_failure_handler
def _execute_python_locally(extract_dir: Path, code: str) -> tuple[str, pl.DataFrame | None]:
    import polars as pl

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    namespace = {
        "__builtins__": __builtins__,
        "pl": pl,
        "polars": pl,
        "csv": csv,
        "json": json,
        "Path": Path,
        "datetime": datetime,
        "re": re,
        "math": math,
        "io": io,
        "DATA_DIR": str(extract_dir),
    }
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(code, namespace)
    except Exception:  # noqa: BLE001 - execute_python reports code tracebacks as tool feedback.
        prefix = _raw_python_output_prefix(stdout_buffer.getvalue(), stderr_buffer.getvalue())
        return f"{prefix}Execution error:\n{traceback.format_exc()}", None

    prefix = _raw_python_output_prefix(stdout_buffer.getvalue(), stderr_buffer.getvalue())
    result_df = namespace.get("result_df")
    if result_df is None:
        defined = [
            name
            for name in namespace
            if not name.startswith("_") and name not in _RAW_EXEC_NAMESPACE_NAMES
        ]
        return (
            f"{prefix}No `result_df` variable found after execution.\n"
            f"Variables defined: {defined}\n"
            "Assign your final DataFrame to `result_df`."
        ), None

    if not isinstance(result_df, pl.DataFrame):
        return (
            f"{prefix}`result_df` is {type(result_df).__name__}, not a Polars DataFrame.\n"
            "Use pl.DataFrame(...) or pl.read_csv(...) to create one."
        ), None

    if result_df.is_empty():
        return f"{prefix}Warning: `result_df` is empty (0 rows). Check your parsing logic.", None

    lines = [
        f"Shape: {result_df.shape[0]} rows x {result_df.shape[1]} columns",
        "",
        "Schema:",
    ]
    for column in result_df.columns:
        lines.append(f"  {column}: {result_df.schema[column]}")
    sample = min(5, len(result_df))
    lines.append(f"\nFirst {sample} rows:")
    lines.append(str(result_df.head(sample)))
    return f"{prefix}Success!\n\n" + "\n".join(lines), result_df
