"""Pure helper functions shared by transition helpers and the pipeline.

No orchestration imports here, just data transformations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nof1_causal_lab.artifacts.raw_data import column_descriptions

if TYPE_CHECKING:
    import pyarrow as pa


def format_schema_for_llm(table: pa.Table) -> str:
    """Format a DataFrame schema and sample for LLM consumption.

    Used by measurement-structure so the LLM can see what columns are available
    when proposing the measurement structure.
    """
    descriptions = column_descriptions(table)
    df = pl.DataFrame(table)
    lines = ["## Dataset Schema\n"]
    lines.append("| Column | Type | Description |")
    lines.append("|--------|------|-------------|")
    for col in df.columns:
        dtype = str(df.schema[col])
        desc = descriptions[col]
        lines.append(f"| {col} | {dtype} | {desc} |")

    lines.append("\n## Sample Data (first 10 rows)\n")
    lines.append(str(df.head(10)))

    lines.append("\n## Summary\n")
    lines.append(f"- Total rows: {len(df)}")
    lines.append(f"- Total columns: {len(df.columns)}")

    # Basic stats for numeric columns
    numeric_cols = [c for c in df.columns if df.schema[c].is_numeric()]
    if numeric_cols:
        lines.append("\n## Numeric Column Statistics\n")
        lines.append(str(df.select(numeric_cols).describe()))

    return "\n".join(lines)
