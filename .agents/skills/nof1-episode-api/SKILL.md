---
name: nof1-episode-api
description: "Drive or inspect the nof1-causal-lab episode state machine over HTTP with curl: edit models, prepare data, fit and simulate; inspect revisions, read episode state/timeline/artifacts, and invoke scientific tools and optional recipes against the tool server. Use when navigating the episode machine as an external agent instead of the web viewer."
---

# nof1-causal-lab episode API — curl skill

> Auto-generated from `packages/api-types/schemas/openapi.json` (the FastAPI OpenAPI spec) by `apps/data-pipeline/scripts/export_agent_api.py`. Edit the route docstrings, not this file.

The scientific interface has four actions: `edit_model`, `prepare_data`, `fit`,
and `simulate`. Requests commit through the serialized episode machine; reads
come from its versioned artifacts and append-only transition log.

## Scientific loop

1. Read `GET /api/machine` for action responsibilities, then
   `GET /api/episodes/{workspace_id}/model` for current model/data versions and findings.
2. Submit to `POST /api/episodes/{workspace_id}/actions`:
   - `edit_model`: `{"action":"edit_model","expected_version":0,"model":{"question":"Does workload affect sleep?"}}`.
     Model structure, measurements, mechanisms, constants, and laws can be edited together.
     Valid incomplete models are saved with applicable specification findings.
   - `prepare_data`: `{"action":"prepare_data","source":"files"}` imports uploaded sources.
     `{"action":"prepare_data","source":"raw_data","raw_data_version":1,"model_version":2}`
     extracts observations using the selected measurement definitions.
   - `fit`: `{"action":"fit","model_version":3,"panel_version":1}` conditions the selected
     model on observations. Returns joint uncertainty and fit diagnostics; predictive
     simulation is a separate request. Current fitting supports independent scalar laws.
   - `simulate`: `{"action":"simulate","model_version":3,"design":{"kind":"trajectory","times":[0,1,2,3],"draws":100,"seed":0}}`
     replicates a study from the selected model's current laws. Independent and joint
     parameter laws use the same nonlinear generator. Optional `comparison_panel_version`
     enables predictive comparisons on the matching observation grid; `design.edge_contrasts`
     adds paired edge-off experiments. Replication draws a new initial state from the
     model's initial distribution, even when the model contains fitted trajectories.
3. Read the action outcome and `GET /api/episodes/{workspace_id}/timeline` for
   `applied`, `rejected`, or `raised` records. Numerical arrays have immutable store references.
   `GET /api/episodes/{workspace_id}/model` includes separately sourced specification,
   identification, data-compatibility, fitting, and simulation findings.

The same requests are available as tools through `GET /api/tools/scientific` and
`POST /api/tools/scientific/{action}`. HTTP and tool calls share execution contracts.

## Revisions and optional recipes

Requests name stored input revisions. Fits check the selected model/data pair; model edits reject base revision conflicts. Read `/revisions` to select history and `/revisions/compare` to compare parameter decisions and recorded evidence.
An edit does not require prior simulation or an authoring admission. Causal numerical
claims still require matching identification and production inference evidence.
Simulation reports retain their own model/data versions; a later edit makes that
report historical rather than evidence for the edited model.

The `/moves` endpoint serves internal jobs; `/recipes/observational-study` runs the optional authoring recipe.
Their stage ordering is not a requirement of the scientific actions. A trajectory design selects
`initial_state` (new_study, retained, fixed, equilibrium), `state_time` or `state_values`,
process/observation noise, timed interventions and check criteria. A causal design
uses `kind: "causal"` and `query` with start, clamps, outcome and readout. It uses the same
generator and requires identification plus committed production-fit evidence.
The `analysis` context is read-only model introspection.

## Data in, results out

Upload files at `POST /api/upload` (`multipart/form-data` with `workspaceId` and
`file`) before `prepare_data` with `source=files`. Read artifact payloads at
`GET /api/episodes/{workspace_id}/artifacts/{artifact_id}`; binary files are served
from `.../files/{filename}`. Long jobs may outlive an HTTP client timeout; inspect
the timeline before submitting another request.

## Read-only deployments

`GET /api/capabilities` reports `moves_enabled`. Read-only deployments reject all
scientific action submissions and machine writes with 403.

## Endpoints

### GET `/api/capabilities`

Whether this deployment serves the move plane.

`moves_enabled` is `false` on the hosted read-only viewer backend, where
every `POST` (moves, auto-run, start-episode) returns 403 and only the read
endpoints are live.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/capabilities"
```

### POST `/api/episodes`

Ensure the episode workflow exists; optionally author its model's question.

Idempotent: attaches to an existing episode or starts a fresh one. Passing
`question` creates or revises the Model with `human` provenance.
Upload raw data at `POST /api/upload` before running the `raw_data`
transition. Returns the same shape as
`GET /api/episodes/{id}`.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"workspace_id": "string"}'
```

### GET `/api/episodes/{workspace_id}`

Current episode state: the single read to poll while navigating.

Returns per-artifact freshness (existence, staleness, version, provenance),
the `legal` moves available right now, and `auto_running` — whether the
background driver is active. Replayed from the append-only transition log,
so it works even against a published read-only store.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID"
```

### POST `/api/episodes/{workspace_id}/actions`

Execute edit_model, prepare_data, fit, or simulate with explicit input revisions.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/actions" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "edit_model", "expected_version": 0, "model": {}}'
```

### GET `/api/episodes/{workspace_id}/artifacts/{artifact_id}`

One artifact version: meta + inline JSON payloads.

Defaults to the episode's *current* version from replayed applied transition
effects. Binary payload files (parquet, pickle) are listed by name, never
inlined.

**Parameters**

- `workspace_id` (path, required)
- `artifact_id` (path, required)
- `version` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/artifacts/ARTIFACT_ID"
```

### GET `/api/episodes/{workspace_id}/artifacts/{artifact_id}/files/{filename}`

One declared payload file from an artifact version.

Defaults to the episode's current version. Unlike the JSON artifact
endpoint, this serves binary files as bytes and refuses undeclared
filenames so callers cannot browse arbitrary workspace paths.

**Parameters**

- `workspace_id` (path, required)
- `artifact_id` (path, required)
- `filename` (path, required)
- `version` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/artifacts/ARTIFACT_ID/files/FILENAME"
```

### GET `/api/episodes/{workspace_id}/artifacts/{artifact_id}/traces`

Traces of the applied transition that produced an artifact version.

Defaults to the episode's current version. The join runs over the
transition journal, so it works against a published read-only store.

**Parameters**

- `workspace_id` (path, required)
- `artifact_id` (path, required)
- `version` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/artifacts/ARTIFACT_ID/traces"
```

### GET `/api/episodes/{workspace_id}/events`

Fine-grained telemetry (e.g. extraction worker fan-out, transition progress).

Pass the last-seen event id as `after` to page forward; omit it for the full
stream. This is finer-grained than the timeline, which records only whole
move outcomes.

**Parameters**

- `workspace_id` (path, required)
- `after` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/events"
```

### GET `/api/episodes/{workspace_id}/model`

Batch canonical aggregates in one committed read transaction.

Omit `at_seq` for the latest applied move, or select a committed journal sequence.
Zero selects the empty model. Rejected/raised attempts are not revisions (404).
Use the returned `context.seq` for subsequent aggregate or collection reads at the same revision.

**Parameters**

- `workspace_id` (path, required)
- `at_seq` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model"
```

### PUT `/api/episodes/{workspace_id}/model`

Validate and atomically replace the named base model revision.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model" \
  -X PUT \
  -H 'Content-Type: application/json' \
  -d '{"expected_version": 0, "model": {}}'
```

### GET `/api/episodes/{workspace_id}/model/constructs`

Authored constructs, using their canonical domain type.

**Parameters**

- `workspace_id` (path, required)
- `at_seq` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/constructs"
```

### GET `/api/episodes/{workspace_id}/model/definition`

The canonical scientific value selected by this journal revision.

**Parameters**

- `workspace_id` (path, required)
- `at_seq` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/definition"
```

### GET `/api/episodes/{workspace_id}/model/edges`

Authored edges, using their canonical domain type.

**Parameters**

- `workspace_id` (path, required)
- `at_seq` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/edges"
```

### GET `/api/episodes/{workspace_id}/model/indicators`

Authored indicators whose owners survive at the selected revision.

**Parameters**

- `workspace_id` (path, required)
- `at_seq` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/indicators"
```

### GET `/api/episodes/{workspace_id}/model/inference-report`

Read the inference transition report associated with the selected model revision.

**Parameters**

- `workspace_id` (path, required)
- `at_seq` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/inference-report"
```

### GET `/api/episodes/{workspace_id}/model/parameters`

Scientific parameter definitions from the selected model, without inference execution.

**Parameters**

- `workspace_id` (path, required)
- `at_seq` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/parameters"
```

### GET `/api/episodes/{workspace_id}/model/views/{artifact_id}`

One display projection from the selected committed model revision.

**Parameters**

- `workspace_id` (path, required)
- `artifact_id` (path, required)
- `at_seq` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/views/ARTIFACT_ID"
```

### POST `/api/episodes/{workspace_id}/moves`

Propose one move; blocks until it is applied, rejected, or raises.

Two kinds:

- Run a transition: `{"move": {"kind": "run", "operation_id": "latent_structure"}}`.
- Author a judgment artifact directly (skip the in-service stage):
  `{"move": {"kind": "write", "artifact_id": "model", "expected_model_version": 0, "provenance":
  "llm"}, "payload": {...}}`. The payload is schema-validated against that
  artifact's contract, journaled, and provenance-stamped; the write becomes a
  revision. Consumers retain their original pins; changed scientific inputs invalidate affected results.

The synchronous outcome is the same record the timeline stores. Long transitions
(statistical model specification, posterior — minutes to hours) can outlive a client timeout; for
those prefer `POST /api/episodes/{workspace_id}/recipes/observational-study` plus polling.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/moves" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"move": {"kind": "run", "operation_id": "raw_data"}}'
```

### GET `/api/episodes/{workspace_id}/operations/{operation_id}/traces`

The latest applied operation's traces, independently of later ModelSpec authorship.

**Parameters**

- `workspace_id` (path, required)
- `operation_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/operations/OPERATION_ID/traces"
```

### POST `/api/episodes/{workspace_id}/recipes/observational-study`

Start the default navigation policy in the background.

Runs enabled stages in dependency order while their outputs are missing or
stale, stopping when quiescent or when a move fails. Returns immediately;
follow progress with `GET /api/episodes/{workspace_id}` (`auto_running`) and
the timeline. An LLM navigator replaces this policy by proposing `moves`
itself. 409 if a driver is already active for this workspace.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/recipes/observational-study" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{}'
```

### GET `/api/episodes/{workspace_id}/revisions`

List stored model, observation and source revisions for deliberate selection.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/revisions"
```

### GET `/api/episodes/{workspace_id}/revisions/compare`

Compare fixed/free decisions, laws and scientific dependencies in the backend.

**Parameters**

- `workspace_id` (path, required)
- `before` (query, required)
- `after` (query, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/revisions/compare"
```

### GET `/api/episodes/{workspace_id}/revisions/data-profile/{panel_version}`

Read the empirical profile for an observation revision independently of the model.

**Parameters**

- `workspace_id` (path, required)
- `panel_version` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/revisions/data-profile/PANEL_VERSION"
```

### GET `/api/episodes/{workspace_id}/revisions/model/{version}`

Read a historical definition, including the input to an earlier fit.

**Parameters**

- `workspace_id` (path, required)
- `version` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/revisions/model/VERSION"
```

### GET `/api/episodes/{workspace_id}/timeline`

The transition journal: every move attempt in order.

Each record is `applied` (state advanced), `rejected` (illegal move, state
unchanged), or `raised` (the transition ran but threw — the record carries the
typed error). Re-running after a `raised`/`rejected` is just proposing the
move again.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/timeline"
```

### GET `/api/episodes/{workspace_id}/traces/{seq}/{subroutine_id}`

One promoted LLM trace from ``episode/traces/{seq:06d}/{subroutine_id}.json``.

**Parameters**

- `workspace_id` (path, required)
- `seq` (path, required)
- `subroutine_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/traces/SEQ/SUBROUTINE_ID"
```

### GET `/api/machine`

The static artifact graph and action hierarchy — read once to orient.

Each transition entry declares what it `consumes`, `produces`, and
optionally co-produces (`produces_optional`), plus its **creation
class**: `deterministic` (pure compute, no credentials), `batch_llm` (bulk
LLM compute on the service's ambient key — you trigger it with a `run` move,
you never supply a key), or `judgment` (proposal work you can author yourself
by writing the produced artifact directly — these are flagged `writable`).

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/machine"
```

### GET `/api/tools/{context_id}`

List a context's validation/query tools — the same tools the in-service LLM loops use.

Each entry is `{name, description, parameters, result}` where `parameters`
and `result` are JSON Schemas. Fetch this first to learn a tool's argument
shape, then call `POST /api/tools/{context_id}/{tool_name}`. Examples:
analysis `simulate` / `get_model_info`, statistical-model-spec `search_literature`.

**Parameters**

- `context_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/tools/CONTEXT_ID"
```

### POST `/api/tools/{context_id}/{tool_name}`

Execute a context tool against the workspace's current artifact-store versions.

Body is `{"workspace_id": "...", "input": {...}}` where `input` matches the
tool's `parameters` schema from `GET /api/tools/{context_id}`; 422 on a schema
violation. Analysis tools reject stale supporting inputs with 409 before
loading or reusing a fitted context.

**Parameters**

- `context_id` (path, required)
- `tool_name` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/tools/CONTEXT_ID/TOOL_NAME" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"workspace_id": "string", "input": {}}'
```

### POST `/api/upload`

Stage one raw input file for the raw_data transition.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/upload" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{}'
```

### GET `/api/workspaces`

Published/local workspaces visible through this facade.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/workspaces"
```
