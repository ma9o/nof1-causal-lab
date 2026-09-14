# Agentic Data Ingestion

| Modality | Interactive | Produces |
|---|---|---|
| Semantic | No | `raw_data` |

Normalizes the latest uploaded raw export into one Arrow table with column descriptions.

## Inputs

| Input | Source | Description |
|---|---|---|
| File upload | User | Single file or a zip bundle containing the raw data |
| `workspace_id` | Pipeline request | Identifies the workspace. `raw_data` transition scans `data/{workspace_id}/input/` and selects the most recent non-hidden file. |

The ingestion agent can normalize most tabular or semi-structured formats as long as the data has a time dimension. Other columns can feed either [computed or semantic](measurement-structure.md#extraction-modes) indicators downstream.

## Process

A sandboxed agentic ingestion loop with `list_files`, `read_file_sample`, `execute_python`, and `submit_table`.

The agent uses Polars for parsing and transformations. `submit_table` attaches a description to every column in an Arrow schema, and the transition persists the table as `raw.parquet`. Readers retain the Arrow table and convert to Polars when computing summaries or extracting indicators.

### Example

A ZIP containing `tickets.csv` and `deploys.csv` may be normalized into one dataframe with columns such as `timestamp`, `event_type`, `ticket_count`, `service_name`, `deploy_status`, and `incident_note`, where each row is one raw event on the shared timeline.

## Outputs

| Field | Description |
|---|---|
| `raw_data` | A `pyarrow.Table` persisted as `raw.parquet`, with a typed `timestamp` column. May be wide (multiple columns) or long (event log format), depending on the raw data structure. |
| Schema field metadata: `description` | Required UTF-8 description of each column, authored through `submit_table`. Stored using [Arrow field metadata](https://arrow.apache.org/docs/python/generated/pyarrow.Field.html#pyarrow.Field.metadata), which survives Parquet loading through PyArrow. |

The table contains its own descriptions; there is no separate JSON profile. Conversion to a Polars dataframe is for computation and does not carry the descriptions.
