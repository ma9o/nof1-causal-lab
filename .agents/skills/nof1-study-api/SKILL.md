---
name: nof1-study-api
description: "Drive or inspect a nof1-causal-lab study over HTTP with curl: call the seven public actions, read saved complete results, compare models and data, and inspect the slim timeline. Use when working on a study as an external agent instead of the web viewer."
---

# nof1-causal-lab study API — curl skill

> Auto-generated from `packages/api-types/schemas/openapi.json` (the FastAPI OpenAPI spec) by `apps/data-pipeline/scripts/codegen/export_api.py`. Edit the route docstrings, not this file.

The public scientific interface has seven calls: `set_question`, `edit_model`,
`prepare_data`, `fit`, `simulate`, `data_diff`, and `model_diff`.
Call each at `POST /api/studies/{workspace_id}/{action}` with its typed JSON arguments.

A call is identified by its action and parsed scientific arguments. Every call accepts
an optional `reasoning` string explaining why the caller is taking the action and
what goal it serves. It is retained with the original request and shown first in the
action log, including running and failed calls. Reasoning is excluded from call
identity: changing or omitting it while polling preserves the original explanation
and never starts another applied or running call. Name model and panel inputs
by immutable revision OIDs. `prepare_data` names uploaded files and captures their
call-time SHA-256 hashes in `input.source.hashes`; repeat those retained hashes to
read a saved call without the upload files. `edit_model` names its base
`expected_revision` and optional check `panel_revision`. `simulate` names an optional
`panel_revision` for authored-law calendar binding. There are no branches or public
head-conflict controls. Every study starts with its immutable `set_question` call.

All seven calls run through the serialized Temporal study workflow.
The response is `kind: running` with the call's arguments, attempt_id, messages and
step/extraction events, or `kind: completed` with the correlated attempt and outcome.
For running calls, repeat the response's `request`; it includes the canonical
arguments and captured file hashes even when the first request omitted them.
Repeat exactly the same arguments to read progress or the completed result. An
identical running call starts no second attempt. Applied calls reuse saved results
without action execution or another timeline entry. Rejected and raised calls remain in
the journal with their messages and error; repeating them retries execution.

Completed applied responses include `snapshot`, `checks`, `inference_report`,
`observation_histories`, `predictive_overlays`, `simulation_paths`, `parameter_draws`,
`data_comparison`, `model_comparison`, `artifacts`, `arrays`, and `traces`, alongside the typed `attempt.outcome`.
All retained simulation paths and large arrays are returned without paging. Tables
are JSON rows, and `arrays` maps immutable array identities to their complete values.
Missing numerical values are null. No extra result or artifact read is needed.
Reports and checks are retained with the action that computed them. Comparisons are
projections of immutable inputs, cached by the package code digest in the shared read cache.

`model_diff {before, after}` records a comparison leaf like `data_diff` and returns
the comparison and each present model definition in `model_comparison`. Comparison
leaves retain their requests and outcomes without advancing scientific state.
Edit details and hover/pin previews read the same immutable projection through
`GET /api/studies/{workspace_id}/model-comparison?before=OID&after=OID`.
Pre-model checkpoints have an empty comparison side; failed attempts compare their
unchanged execution parent.

Read `GET /api/studies/{workspace_id}/timeline` for replayable arguments, status,
messages, errors, trace ids, input dependency links, and the running call. Scientific
results and branches are excluded. The viewer repeats applied calls only; failed or
unknown calls are displayed from the journal without executing them. Input panes
resolve the earlier calls named by the selected call's dependency links.

`GET /api/workspaces` lists available studies and questions. Its X-Actions-Enabled
header supplies the landing page capability. `POST /api/upload` stages a named input
file (`multipart/form-data` with `workspaceId` and `file`).

With READ_ONLY_FACADE=1, saved calls are answered and unsaved recorded calls return
403. Viewer comparison projections remain available. Scientific checks,
production fitting and simulations continue using the exact nonlinear engines.

## Endpoints

### POST `/api/studies/{workspace_id}/data_diff`

Compare immutable left/right data selections and retain a comparison leaf. Identical applied calls reuse the complete comparison without another attempt; identical running calls return that attempt's progress. Failures stay in the timeline and can be retried.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/data_diff" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "data_diff", "left": {"kind": "panel", "revision": "string"}, "right": {"kind": "panel", "revision": "string"}}'
```

### POST `/api/studies/{workspace_id}/edit_model`

Save a model naming its base expected_revision and optional panel_revision. No head conflict check. The complete result includes findings, draws, histories, artifacts and traces; use model_diff separately for changes.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/edit_model" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "edit_model", "expected_revision": "string", "model": {}}'
```

### POST `/api/studies/{workspace_id}/fit`

Condition the named model_revision on panel_revision. Returns the saved complete inference result, including joint posterior arrays, diagnostics and every observation history. Repeat the same arguments to read progress or the saved completion.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/fit" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "fit", "model_revision": "string", "panel_revision": "string"}'
```

### GET `/api/studies/{workspace_id}/model-comparison`

Read the immutable comparison projection for viewer previews without submitting an action.

**Parameters**

- `workspace_id` (path, required)
- `before` (query, required)
- `after` (query, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/model-comparison"
```

### POST `/api/studies/{workspace_id}/model_diff`

Compare named before/after model trees or checkpoints and retain a comparison leaf, including definitions and evidence in model_comparison. Identical applied calls reuse the comparison; running calls return progress. Failures stay in the timeline and can be retried.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/model_diff" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "model_diff", "before": "string", "after": "string"}'
```

### POST `/api/studies/{workspace_id}/prepare_data`

Prepare named uploaded files or a saved simulation replicate. Capture each named file's SHA-256 at call time. Repeat the retained source.hashes to read saved results without uploaded bytes. Running results include step/extraction events; completed results include all observations, profiles and traces.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/prepare_data" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "prepare_data", "input": {"source": {"files": ["string"]}, "definition": {"default_window": "string", "variables": [{"observation": {"id": "string", "name": "string", "measurement_dtype": "continuous", "aggregation": "first"}, "extraction": {"kind": "computed", "how_to_measure": "string", "source_columns": [{}]}}]}}}'
```

### POST `/api/studies/{workspace_id}/set_question`

Set the immutable study question first. Repeat parsed arguments to read saved results or running progress; failed calls may be retried.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/set_question" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "set_question", "question": {"text": "string"}}'
```

### POST `/api/studies/{workspace_id}/simulate`

Simulate the named model_revision and optional panel_revision. Fitted laws retain their fit origin. Returns all paths and arrays without paging, their summaries, causal evidence and traces. Repeat the same arguments to read progress or saved completion.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/simulate" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"start": "string", "horizon": "string", "action": "simulate", "model_revision": "string"}'
```

### GET `/api/studies/{workspace_id}/timeline`

Slim call log: replayable arguments, status, messages, errors, trace ids and dependencies. No inline results and no branches. Selecting an applied node repeats its action; never repeat failed or unknown nodes from the viewer.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/timeline"
```

### POST `/api/upload`

Stage one named raw input file for prepare_data; that call captures its SHA-256.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/upload" \
  -X POST \
  -F "file=@/path/to/file" \
  -F "workspaceId=WORKSPACEID"
```

### GET `/api/workspaces`

Available workspaces and their immutable study questions.

X-Actions-Enabled preserves the landing page's deployment capability without a separate endpoint.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/workspaces"
```
