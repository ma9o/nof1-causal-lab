"""Optional authored descriptions on ready-to-use source table columns."""

from collections.abc import Mapping

import pyarrow as pa


def column_descriptions(table: pa.Table) -> Mapping[str, str | None]:
    """Read source descriptions; CSV columns carry no authored schema metadata."""
    descriptions: dict[str, str | None] = {
        field.name: (
            field.metadata[b"description"].decode("utf-8")
            if field.metadata is not None and b"description" in field.metadata
            else None
        )
        for field in table.schema
    }
    return descriptions
