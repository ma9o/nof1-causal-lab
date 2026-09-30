---
name: nof1-episode-api
description: "Drive or inspect the nof1-causal-lab episode state machine over HTTP with curl: edit models, prepare data, fit and simulate; inspect revisions, read episode state/timeline/artifacts, and invoke scientific tools with dispatch and polling against the tool server. Use when navigating the episode machine as an external agent instead of the web viewer."
---

# nof1-causal-lab episode API — curl skill

> Auto-generated from `packages/api-types/schemas/openapi.json` (the FastAPI OpenAPI spec) by `apps/data-pipeline/scripts/codegen/export_api.py`. Edit the route docstrings, not this file.

The scientific interface has four actions: `edit_model`, `prepare_data`, `fit`,
and `simulate`. Requests commit through the serialized episode machine; reads
come from its versioned artifacts and append-only transition log.

## Scientific loop

1. Read `GET /api/machine` for action responsibilities, then
   `GET /api/episodes/{workspace_id}/model` for current model/data versions and findings.
2. Submit to `POST /api/episodes/{workspace_id}/actions`:
   - `edit_model`: `{"action":"edit_model","expected_revision":null,"model":{"question":"Does workload affect sleep?"}}`.
     Model structure, measurements, mechanisms, constants, and laws can be edited together.
     Valid incomplete models are saved with applicable specification findings.
   - `prepare_data`: supply `input={"source":{"files":["diary.csv"]},"definition":{...}}`
     with `default_window`, `variables`, and optional interpretation `context` in the definition.
     Each variable has a stable ID, dtype, summary, scoring rubric, extraction mode,
     source columns, window and codebook as appropriate. The action ingests and extracts
     in one call, retaining the semantic worker fan-out and deterministic scoring paths.
     Alternatively, `input={"revision":"<simulation commit OID>","replicate":0}`
     selects one recorded simulation draw. Its observations, schema and support layout
     are already defined, so extraction, re-encoding and filling are skipped.
     Files may declare optional source coverage `start` and `end` dates; only complete
     support windows inside that span are prepared. Computed variables may use Polars
     `fill_null` strategies or a numeric constant, with `fill_null_limit` for forward/backward.
     Ingestion and one-variable, one-window extraction requests reuse retained validated
     results by content across studies; reuse is reported in action messages. The committed
     panel and recipe remain the scientific record. Both sources run numerical data checks without loading a model.
     Latent paths and parameter truths stay in simulation sources.
     See the [prepare_data chart](../../../docs/assets/action-flows/prepare-data.svg) for branches
     and [time semantics](../../../docs/assumptions.md#time) for the panel origin.
   - `fit`: `{"action":"fit","model_revision":"<model tree OID>","panel_revision":"<panel tree OID>"}` conditions the selected
     model on observations. Returns joint uncertainty and fit diagnostics; predictive
     simulation is a separate request. Current fitting supports independent scalar laws.
   - `simulate`: `{"action":"simulate","model_revision":"<model tree OID>","end":30,"interventions":[]}`
     generates forward from the model's current laws. `start` optionally selects an earlier
     model time; otherwise generation starts at its latest retained state, or zero when it
     has only an initial-state law. Times use absolute model days. Interventions are optional:
     `{"target":"<construct ID>","time":5,"value":1}` assigns a state at that time,
     then its natural dynamics resume. The framework derives the grid and always includes
     process and observation uncertainty. The saved report includes all state and indicator
     summaries, law provenance and fit reliability, with causal intervals only when certified.
     See [time semantics](../../../docs/assumptions.md#time) for initial laws and calendar binding.
     Compare the saved observations separately with `data_diff`; simulation does not
     accept comparison data or change its generation rules for predictive checks.
3. Dispatch returns HTTP 202 with only `{"attempt_id":"<UUID>"}` after durable acceptance.
   Poll `GET /api/episodes/{workspace_id}/actions/{attempt_id}` for `{done, body, messages}`.
   While running, `done` is false and `body` is null. At completion, `body` contains the
   scientific result, or is null on failure. Messages accumulate as
   `{timestamp, level, label}` with UTC timestamps, `debug|info|warn|error` levels,
   and stable `SCREAMING_SNAKE_CASE` labels. Warnings can accompany a saved result;
   failed actions leave the scientific branch unchanged. Do not redispatch while polling.
   `GET /api/episodes/{workspace_id}/timeline` retains `applied`, `rejected`, or `raised`
   attempts and their messages. Numerical arrays have immutable store references.
   `GET /api/episodes/{workspace_id}/model` includes separately sourced specification,
   identification, data-compatibility, fitting, and simulation findings.

The same requests are available as tools through `GET /api/tools/scientific` and
`POST /api/tools/scientific/{action}`; these tools return the receipt in `result`.
Use `poll_action` with `{attempt_id}` to read the same poll response in `result`.
HTTP and tool calls share execution contracts. V2 is a read-only inspector.

## Revisions and execution

Requests name stored input revisions. Fits check the selected model/data pair; model edits reject base revision conflicts. Read `/revisions` to select history. `GET /model-diff?before=...&after=...` compares model trees or Git checkpoints; `POST /data-diff` compares saved data selections in `left` and `right`. Both are read-only inspection operations and create no action attempt.
An edit does not require prior simulation or an authoring admission. Causal numerical
claims still require matching identification and production inference evidence.
Model edits automatically run affected checks, including one exact whole-model
predictive batch when a compatible panel is available. Data preparation runs only
data checks. Unchanged checks reuse
their recorded results. Scientific failures save as findings; missing prerequisites carry
not_evaluated reasons. Read predictive details and law provenance in the action body.
Simulation reports retain their own generating model revision; a later edit makes that
report historical rather than evidence for the edited model.

Only the four scientific actions submit scientific work. Execution jobs and
LLM subroutines are private implementation details; callers do not select them.
Simulation always uses the same nonlinear generator. Paired intervention histories share
joint parameter/state draws and random streams. Causal effects on the model's default
outcome are reported only when identification and committed production-fit evidence support
that interpretation; otherwise the report keeps its histories with an explicit reason.
The `analysis` context is read-only model introspection.

## Data in, results out

Upload files at `POST /api/upload` (`multipart/form-data` with `workspaceId` and
`file`) before `prepare_data` with its file preparation input. Read artifact payloads at
`GET /api/episodes/{workspace_id}/artifacts/{artifact_id}`; binary files are served
from `.../files/{filename}`. Long jobs may outlive an HTTP client timeout; inspect
the timeline before submitting another request.

## Read-only deployments

`GET /api/capabilities` reports `actions_enabled`. Read-only deployments reject scientific action submissions with 403.

## Endpoints

### GET `/api/capabilities`

Whether this deployment serves scientific actions.

`actions_enabled` is `false` on the hosted read-only viewer backend, where
every `POST` (scientific actions and study management) returns 403 and only the read
endpoints are live.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/capabilities"
```

### GET `/api/episodes/{workspace_id}`

Current episode state: the single read to poll while navigating.

Returns the four scientific action names and per-artifact existence,
freshness and revision from the selected Git branch snapshot, and the
attempt the episode's Temporal workflow is executing on any branch, if any.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID"
```

### POST `/api/episodes/{workspace_id}/actions`

Accept durable work and return its receipt; retrieve results by polling the attempt.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)
- `expected_head` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/actions" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "edit_model", "expected_revision": "string", "model": {}}'
```

### GET `/api/episodes/{workspace_id}/actions/{attempt_id}`

Read accumulated labels and the final scientific body without dispatching work.

**Parameters**

- `workspace_id` (path, required)
- `attempt_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/actions/ATTEMPT_ID"
```

### GET `/api/episodes/{workspace_id}/artifacts/{artifact_id}`

One artifact revision: meta + inline JSON payloads.

Defaults to the selected branch's current revision. Binary payload files (parquet, pickle) are listed by name, never
inlined.

**Parameters**

- `workspace_id` (path, required)
- `artifact_id` (path, required)
- `revision` (query, optional)
- `branch` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/artifacts/ARTIFACT_ID"
```

### GET `/api/episodes/{workspace_id}/artifacts/{artifact_id}/files/{filename}`

One declared payload file from an artifact revision.

Defaults to the episode's current revision. Unlike the JSON artifact
endpoint, this serves binary files as bytes and refuses undeclared
filenames so callers cannot browse arbitrary workspace paths.

**Parameters**

- `workspace_id` (path, required)
- `artifact_id` (path, required)
- `filename` (path, required)
- `revision` (query, optional)
- `branch` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/artifacts/ARTIFACT_ID/files/FILENAME"
```

### GET `/api/episodes/{workspace_id}/artifacts/{artifact_id}/traces`

Traces of the applied transition that produced an artifact revision.

Defaults to the episode's current revision. The join runs over the
transition journal, so it works against a published read-only store.

**Parameters**

- `workspace_id` (path, required)
- `artifact_id` (path, required)
- `revision` (query, optional)
- `branch` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/artifacts/ARTIFACT_ID/traces"
```

### GET `/api/episodes/{workspace_id}/branches`

Get Branches

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/branches"
```

### POST `/api/episodes/{workspace_id}/branches`

Fork the complete study at a checkpoint; the new branch shares its ancestry.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/branches" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"name": "string", "at": "string"}'
```

### POST `/api/episodes/{workspace_id}/data-diff`

Compare existing datasets without creating an action, fitting or simulating.

Each side accepts a data reference or a nonempty array of references. Panel
references select artifact revisions; simulation references select applied
simulation commits and optionally one replicate (otherwise every draw).
Simulation calendar coordinates come from the saved report's origin.
Exact anchors and measurement windows determine which predictive comparisons
are available. Results preserve each history and report incompatible inputs.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/data-diff" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"left": {"kind": "panel", "revision": "string"}, "right": {"kind": "panel", "revision": "string"}}'
```

### GET `/api/episodes/{workspace_id}/events`

Fine-grained telemetry (e.g. extraction worker fan-out, transition progress).

Pass the last-seen event id as `after` to page forward; omit it for the full
stream. This is finer-grained than the timeline, which records only whole
action outcomes.

**Parameters**

- `workspace_id` (path, required)
- `after` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/events"
```

### GET `/api/episodes/{workspace_id}/logs/{commit_id}`

Get Attempt Log

**Parameters**

- `workspace_id` (path, required)
- `commit_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/logs/COMMIT_ID"
```

### GET `/api/episodes/{workspace_id}/model`

Batch canonical aggregates in one committed read transaction.

Omit `at` for the selected branch head, or pass an exact Git commit ID.
Use `context.commit_id` to pin subsequent reads. Failed attempts retain logs without advancing scientific state.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model"
```

### GET `/api/episodes/{workspace_id}/model-diff`

Compare two model artifact revisions or Git checkpoints containing a model.

Returns identity-aligned definition changes, parameter decisions and graph
topology differences. Graph highlights exclude laws and other entity attributes.
Checkpoint selections also include their recorded fit/simulation
evidence; selecting a model tree alone does not infer an associated run.

**Parameters**

- `workspace_id` (path, required)
- `before` (query, required)
- `after` (query, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model-diff"
```

### GET `/api/episodes/{workspace_id}/model/constructs`

Authored constructs, using their canonical domain type.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/constructs"
```

### GET `/api/episodes/{workspace_id}/model/definition`

The canonical scientific value selected by this journal revision.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/definition"
```

### GET `/api/episodes/{workspace_id}/model/edges`

Authored edges, using their canonical domain type.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/edges"
```

### GET `/api/episodes/{workspace_id}/model/indicators`

Authored indicators whose owners survive at the selected revision.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/indicators"
```

### GET `/api/episodes/{workspace_id}/model/inference-report`

Read the inference transition report associated with the selected model revision.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/inference-report"
```

### GET `/api/episodes/{workspace_id}/model/parameters`

Scientific parameter definitions from the selected model, without inference execution.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/parameters"
```

### GET `/api/episodes/{workspace_id}/model/views/{artifact_id}`

One display projection from the selected committed model revision.

**Parameters**

- `workspace_id` (path, required)
- `artifact_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/views/ARTIFACT_ID"
```

### POST `/api/episodes/{workspace_id}/model/visuals/mechanism`

Read conditional drift curves using the exact model equations; creates no scientific action.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/visuals/mechanism" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"owner_id": "string"}'
```

### GET `/api/episodes/{workspace_id}/model/visuals/observations/{indicator_id}`

All prepared observations on their recorded temporal support.

**Parameters**

- `indicator_id` (path, required)
- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/visuals/observations/INDICATOR_ID"
```

### GET `/api/episodes/{workspace_id}/model/visuals/parameters`

All coordinates and all draws of the retained joint posterior.

**Parameters**

- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/visuals/parameters"
```

### GET `/api/episodes/{workspace_id}/model/visuals/predictive/{indicator_id}`

Saved predictive paths on the exact schedule of their pinned inputs.

**Parameters**

- `indicator_id` (path, required)
- `workspace_id` (path, required)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/visuals/predictive/INDICATOR_ID"
```

### GET `/api/episodes/{workspace_id}/model/visuals/simulation`

A contiguous page of original simulation draws, without time thinning.

**Parameters**

- `workspace_id` (path, required)
- `start` (query, optional)
- `count` (query, optional)
- `branch` (query, optional)
- `at` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/model/visuals/simulation"
```

### GET `/api/episodes/{workspace_id}/operations/{operation_id}/traces`

The latest applied operation's traces, independently of later ModelSpec authorship.

**Parameters**

- `workspace_id` (path, required)
- `operation_id` (path, required)
- `branch` (query, optional)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/operations/OPERATION_ID/traces"
```

### GET `/api/episodes/{workspace_id}/revisions`

List stored model, observation and source revisions for deliberate selection.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/revisions"
```

### GET `/api/episodes/{workspace_id}/revisions/data-profile/{panel_revision}`

Read the empirical profile for an observation revision independently of the model.

**Parameters**

- `workspace_id` (path, required)
- `panel_revision` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/revisions/data-profile/PANEL_REVISION"
```

### GET `/api/episodes/{workspace_id}/revisions/model/{revision}`

Read a historical definition, including the input to an earlier fit.

**Parameters**

- `workspace_id` (path, required)
- `revision` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/revisions/model/REVISION"
```

### GET `/api/episodes/{workspace_id}/timeline`

The transition journal: every action attempt in order.

Each record is `applied` (state advanced), `rejected` (rejected action, state
unchanged), or `raised` (the transition ran but threw — the record carries the
typed error). Re-running after a `raised`/`rejected` is just proposing the
action again.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/timeline"
```

### GET `/api/episodes/{workspace_id}/traces/{commit_id}/{subroutine_id}`

One trace from the owning attempt's Git commit.

**Parameters**

- `workspace_id` (path, required)
- `commit_id` (path, required)
- `subroutine_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/episodes/WORKSPACE_ID/traces/COMMIT_ID/SUBROUTINE_ID"
```

### GET `/api/machine`

The static artifact graph and action hierarchy — read once to orient.

Each transition entry declares what it `consumes`, `produces`, and
optionally co-produces (`produces_optional`), plus its **creation
class**: `deterministic` (pure compute, no credentials), `batch_llm` (bulk
LLM compute on the service's ambient key), or `judgment` (model-authoring
implementation). These jobs are private; callers submit the four scientific actions.

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
