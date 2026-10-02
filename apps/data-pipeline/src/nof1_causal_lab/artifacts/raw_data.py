"""Raw data is an Arrow table with descriptions on its schema fields."""

from collections.abc import Mapping

import pyarrow as pa

from nof1_causal_lab.study.errors import ArtifactWriteRejected


def with_column_descriptions(table: pa.Table, descriptions: Mapping[str, str]) -> pa.Table:
    """Attach an authored description to every column without changing its values."""
    missing = [name for name in table.column_names if name not in descriptions]
    if missing:
        raise ArtifactWriteRejected(
            f"Missing descriptions for columns: {missing}", artifact_id="raw_data"
        )
    extra = [name for name in descriptions if name not in table.column_names]
    if extra:
        raise ArtifactWriteRejected(
            f"Descriptions for non-existent columns: {extra}", artifact_id="raw_data"
        )
    schema = pa.schema(
        [
            field.with_metadata(
                {**(field.metadata or {}), b"description": descriptions[field.name].encode("utf-8")}
            )
            for field in table.schema
        ],
        metadata=table.schema.metadata,
    )
    return table.cast(schema)


def column_descriptions(table: pa.Table) -> dict[str, str]:
    """Read the required descriptions from the raw table's Arrow schema."""
    descriptions: dict[str, str] = {}
    for field in table.schema:
        metadata = field.metadata
        if metadata is None or b"description" not in metadata:
            raise ArtifactWriteRejected(
                f"Missing description for raw-data column: {field.name}", artifact_id="raw_data"
            )
        descriptions[field.name] = metadata[b"description"].decode("utf-8")
    return descriptions
