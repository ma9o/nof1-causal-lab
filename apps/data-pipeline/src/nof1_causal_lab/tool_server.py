"""Study HTTP API: content-named Temporal calls and cached model comparisons.

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

A call is identified by its action and parsed arguments. Name model and panel inputs
by immutable revision OIDs. `prepare_data` names uploaded files and captures their
call-time SHA-256 hashes in `input.source.hashes`; repeat those retained hashes to
read a saved call without the upload files. `edit_model` names its base
`expected_revision` and optional check `panel_revision`. `simulate` names an optional
`panel_revision` for authored-law calendar binding. There are no branches or public
head-conflict controls. Every study starts with its immutable `set_question` call.

The six recorded calls run through the existing serialized Temporal study workflow.
The response is `kind: running` with the call's arguments, attempt_id, messages and
step/extraction events, or `kind: completed` with the correlated attempt and outcome.
For running calls, repeat the response's `request`; it includes the canonical
arguments and captured file hashes even when the first request omitted them.
Repeat exactly the same arguments to read progress or the completed result. An
identical running call starts no second attempt. Applied calls reuse saved results
without execution or another timeline entry. Rejected and raised calls remain in
the journal with their messages and error; repeating them retries execution.

Completed applied responses include `snapshot`, `inference_report`,
`observation_histories`, `predictive_overlays`, `simulation_paths`, `parameter_draws`,
`artifacts`, `arrays`, and `traces`, alongside the typed `attempt.outcome`.
All retained simulation paths and large arrays are returned without paging. Tables
are JSON rows, and `arrays` maps immutable array identities to their complete values.
Missing numerical values are null. No extra result or artifact read is needed.

`model_diff {before, after}` is a cached comparison read. It returns the comparison
and each present model definition, creates no attempt, leaf or timeline entry, and works on
the read-only facade. Edit details and hover/pin previews use this same call.
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
403. Model comparisons remain available. Scientific checks,
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
