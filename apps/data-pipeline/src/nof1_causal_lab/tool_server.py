"""Study HTTP API: content-named Temporal calls and immutable viewer projections.

Run with ``uv run uvicorn nof1_causal_lab.tool_server:app --port 8100``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nof1_causal_lab.study_api import (
    TemporalClientProvider,
    uploads_router,
    workspaces_router,
)
from nof1_causal_lab.study_api import router as study_router

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


_API_DESCRIPTION = """\
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
"""


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    from nof1_causal_lab.utils.config import configure_jax_persistent_cache

    configure_jax_persistent_cache()
    yield


app = FastAPI(
    lifespan=_lifespan,
    title="nof1-causal-lab study API",
    description=_API_DESCRIPTION,
    docs_url="/api/docs",
)
app.state.study_clients = TemporalClientProvider()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Actions-Enabled"],
)

app.include_router(study_router)
app.include_router(workspaces_router)
app.include_router(uploads_router)
