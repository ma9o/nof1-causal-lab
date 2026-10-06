---
name: nof1-study-api
description: "Drive or inspect a nof1-causal-lab study over HTTP with curl: call the seven public actions, read saved complete results, compare models and data, and inspect the slim timeline. Use when working on a study as an external agent instead of the web viewer."
---

# nof1-causal-lab study API — curl skill

> Auto-generated from `packages/api-types/schemas/openapi.json` (the FastAPI OpenAPI spec) by `apps/data-pipeline/scripts/codegen/export_api.py`. Edit the route docstrings, not this file.

The public scientific interface has seven actions: `edit_question`, `edit_model`,
`prepare_data`, `fit`, `simulate`, `data_diff`, and `model_diff`.
Start a call with `POST /api/studies/{workspace_id}/{action}` using
`{action, input, reasoning}`. Each action owns its typed `input`; optional top-level
`reasoning` explains the caller's intent and is excluded from call identity.

POST and polling GET return `{call_id, action, status, commit_id, body, messages}`.
`status` is `running`, `failed`, or `success`. `commit_id` is null before publication;
a journaled failure also has a commit. `body` is null while running or failed and
contains only that action's typed scientific output on success.
`messages` is the sole execution-log accumulator: lifecycle entries, structured
progress, full LLM/tool traces, and failure details all remain there.

Poll `GET /api/studies/{workspace_id}/{action}/{call_id}`. GET only reads an existing
call; it never starts a workflow, executes an action, or retries a failure. Unknown
call IDs return 404. Identical resolved inputs reuse running calls and both successful
and failed completed calls, preserving the first request's reasoning.

Revision selectors accept exact Git hashes or `"latest"`. POST resolves selectors
once before computing `call_id`; GET keeps that selection when newer revisions appear.
Missing valid inputs return HTTP 422. `prepare_data` takes `input.source` as one
folder name under `data/{workspace_id}/`, such as `"input"`. Every file beneath
that folder, including subfolders, is captured and hashed before call identity.
Polling saved calls does not require the original source folder.
`edit_model` selects a question or model parent through `input.parent_ref`;
a model parent supplies its pinned question, without a head-conflict
check. Every study starts with `edit_question`.

Each action owns its successful `body`; input state is read from its producing calls.
For `model_diff` and `data_diff`, `body` is the complete saved comparison.
Fit and simulation outputs include their own retained arrays and paths without paging.
Missing numerical values are null.
Reports and checks are retained with their action in Git and never recomputed by GET,
including after cache deletion or code changes. Comparisons and failures remain
journal leaves without advancing scientific state.

`GET /api/studies/{workspace_id}/timeline` returns call IDs, retained requests,
outcome summaries, lifecycle messages, trace IDs, input dependencies and the running
call. The read-only web viewer follows recorded input dependencies and uses their call
IDs to compose saved results and logs. For example, a simulation reads its model
from the call named by its model dependency.
Agents create studies and submit actions directly to the facade.
`GET /api/workspaces` lists studies and questions; X-Actions-Enabled reports whether
new actions are enabled. `POST /api/upload` stages a named input file using
`multipart/form-data` with `workspaceId` and `file`.

With READ_ONLY_FACADE=1, existing calls remain readable. POST for an unsaved call
returns 403; GET for an unknown call returns 404.

## Endpoints

### POST `/api/studies/{workspace_id}/data_diff`

Compare input.left_ref and input.right_ref, each a gitref with an optional replicate_index or a list of those references. Omit the index to compare all recorded histories. Data latest selects prepared user data; simulations require explicit gitrefs. Retain the complete comparison in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/data_diff" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "data_diff", "input": {"left_ref": {"revision": "string"}, "right_ref": {"revision": "string"}}}'
```

### POST `/api/studies/{workspace_id}/edit_model`

Save input.model from input.parent_ref, a question or model revision, and evaluate data-independent model checks. A model parent supplies its pinned question. The body contains the produced model and its findings; messages retain all execution logging. GET polls the returned call_id.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/edit_model" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "edit_model", "input": {"parent_ref": "string", "model": {}}}'
```

### POST `/api/studies/{workspace_id}/edit_question`

Set the study question from input.question. POST returns a call_id; GET polls it. Identical resolved calls reuse both successes and failures.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/edit_question" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "edit_question", "input": {"question": {"text": "string", "outcome": "string"}}}'
```

### POST `/api/studies/{workspace_id}/fit`

Condition input.model_ref on the history selected by input.data_ref and required input.replicate_index. Zero selects prepared user data; a simulation index selects one recorded draw. GET polls call_id. The body retains the complete inference result and arrays.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/fit" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "fit", "input": {"model_ref": "string", "data_ref": "string", "replicate_index": 0}}'
```

### POST `/api/studies/{workspace_id}/model_diff`

Compare input.before_ref and input.after_ref and retain definitions and evidence in Git. GET polls call_id. Identical resolved calls reuse successes and failures; the comparison is the body.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/model_diff" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "model_diff", "input": {"before_ref": "string", "after_ref": "string"}}'
```

### POST `/api/studies/{workspace_id}/prepare_data`

Prepare observations from uploaded tables using the selected model's definitions.

The source is a folder under ``data/{workspace_id}/``, including subfolders.
CSV and Parquet tables must supply a date or datetime timestamp column and
are concatenated in captured file order. Source bytes and revision selectors
are pinned before computing call identity.

Args:
    workspace_id: Study workspace containing the source folder and revision
        history.
    body: Preparation request selecting a model, source folder, and computed
        rules or semantic extraction instructions for its observation IDs.
    clients: Provider of the shared Temporal connection used to dispatch
        or retrieve the preparation workflow.

Returns:
    Current call status or a cached HTTP response for the same call. Polling
    by call ID exposes progress messages and, on success, the prepared
    observations and available profiles.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/prepare_data" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "prepare_data", "input": {"model_ref": "string", "source": "string", "extraction": {}}}'
```

### POST `/api/studies/{workspace_id}/simulate`

Simulate input.model_ref using input.simulation and optional input.panel_ref. Fitted laws retain their fit origin. body.data contains an array of observation histories of the same type returned by prepare_data. The successful body also includes all paths, arrays and causal evidence; messages retain traces. GET polls call_id.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/simulate" \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"action": "simulate", "input": {"model_ref": "string", "simulation": {"start": "string", "horizon": "string"}}}'
```

### GET `/api/studies/{workspace_id}/timeline`

Slim call log: call IDs, retained arguments, outcome summaries and dependencies. The viewer reads complete results and messages with GET using each call_id.

**Parameters**

- `workspace_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/timeline"
```

### GET `/api/studies/{workspace_id}/{action}/{call_id}`

Read an existing call by ID. Never resolve inputs, start a workflow, execute, or retry. The envelope and accumulated messages match POST, including cached failures.

**Parameters**

- `workspace_id` (path, required)
- `action` (path, required)
- `call_id` (path, required)

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/studies/WORKSPACE_ID/ACTION/CALL_ID"
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

X-Actions-Enabled reports whether the facade accepts new calls.

```bash
curl -s "${TOOL_SERVER_URL:-http://localhost:8100}/api/workspaces"
```
