"""Read-only study facade: the hosted viewer's entire backend.

Serves the same journal-backed read endpoints as the full tool server —
same code, same projections — against whatever store the environment
selects (R2 in production), without importing the tool-execution/SSM
stack. Deployments set ``READ_ONLY_FACADE=1`` so the move plane
403s and ``/api/capabilities`` advertises ``actions_enabled: false``; no
Temporal, no tool execution, no LLM anywhere. A published workspace is
viewable (including live, while a local service is still writing to it)
without any hosted stateful service.

Deployed as a Modal ASGI app (see :mod:`nof1_causal_lab.actions.modal_runners`),
or run locally::

    READ_ONLY_FACADE=1 uv run uvicorn --factory \
        nof1_causal_lab.read_facade:create_read_facade_app --port 8100
"""

from __future__ import annotations

from fastapi import FastAPI

from nof1_causal_lab.study_api import (
    TemporalClientProvider,
    capabilities_router,
    uploads_router,
    workspaces_router,
)
from nof1_causal_lab.study_api import router as study_router


def create_read_facade_app() -> FastAPI:
    app = FastAPI(title="Study Read Facade")
    app.state.study_clients = TemporalClientProvider()
    app.include_router(study_router)
    app.include_router(capabilities_router)
    app.include_router(workspaces_router)
    app.include_router(uploads_router)
    return app
