"""Modal-backed fits and the hosted read-only facade.

When ``DEPLOYMENT_ENV=production``, ``actions.runners.run_action`` routes fits
here. The remote function runs the same ``run_action_locally`` against the same
R2-backed artifact store — Modal is compute placement, not a different execution
path. Version stamps come back as plain dicts (Modal pickles across an image boundary).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import modal
from pydantic import TypeAdapter

from nof1_causal_lab.study.records import Applied

if TYPE_CHECKING:
    from fastapi import FastAPI

    from nof1_causal_lab.actions.contracts import FitRequest
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
    from nof1_causal_lab.json_types import JsonObject
    from nof1_causal_lab.study.records import ModelFitResult

# ═══════════════════════════════════════════════════════════════════════════════
# Modal images
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[3]  # -> apps/data-pipeline/
GPU_A100_80GB = "A100-80GB"

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(uv_project_dir=str(ROOT), groups=["dev", "cloud"], frozen=True)
    .env({"PYTHONPATH": "/root/src", "DEPLOYMENT_ENV": "production"})
    .add_local_file(ROOT / "config.yaml", remote_path="/root/config.yaml")
    .add_local_file(ROOT / "pyproject.toml", remote_path="/root/pyproject.toml")
    .add_local_dir(ROOT / "src" / "nof1_causal_lab", remote_path="/root/src/nof1_causal_lab")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(uv_project_dir=str(ROOT), groups=["dev", "cloud", "gpu"], frozen=True)
    .env({"PYTHONPATH": "/root/src", "DEPLOYMENT_ENV": "production"})
    .add_local_file(ROOT / "config.yaml", remote_path="/root/config.yaml")
    .add_local_file(ROOT / "pyproject.toml", remote_path="/root/pyproject.toml")
    .add_local_dir(ROOT / "src" / "nof1_causal_lab", remote_path="/root/src/nof1_causal_lab")
)

app = modal.App("nof1-causal-lab-pipeline", image=cpu_image)
secrets = modal.Secret.from_name("nof1-causal-lab-pipeline-secrets")


# ═══════════════════════════════════════════════════════════════════════════════
# Remote functions
# ═══════════════════════════════════════════════════════════════════════════════


@app.function(
    timeout=10800,
    cpu=8,
    memory=32768,
    image=gpu_image,
    gpu=GPU_A100_80GB,
    secrets=[secrets],
)
async def _run_fit_gpu(
    workspace_id: str,
    request: JsonObject,
    pins: dict[ArtifactId, GitOid],
) -> JsonObject:
    """Run a fit on Modal GPU compute against the R2 artifact store."""
    from nof1_causal_lab.actions.contracts import FitRequest
    from nof1_causal_lab.actions.runners import run_action_locally
    from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid

    result = await run_action_locally(
        workspace_id,
        FitRequest.model_validate(request),
        TypeAdapter(dict[ArtifactId, GitOid]).validate_python(pins),
    )
    payload: JsonObject = result.model_dump(mode="json")
    return payload


@app.function(
    image=cpu_image,
    env={"READ_ONLY_FACADE": "1"},
    secrets=[secrets],
)
@modal.asgi_app()
def read_facade() -> FastAPI:
    """The hosted viewer's backend: journal reads over the R2 store, no moves."""
    from nof1_causal_lab.read_facade import create_read_facade_app

    return create_read_facade_app()


# ═══════════════════════════════════════════════════════════════════════════════
# Runner callables (bound by actions.runners)
# ═══════════════════════════════════════════════════════════════════════════════


async def run_fit_on_modal(
    workspace_id: str,
    request: FitRequest,
    pins: dict[ArtifactId, GitOid],
) -> Applied[ModelFitResult]:
    """Run a fit remotely; credentials come from the Modal secret block."""
    from nof1_causal_lab.study.records import ModelFitResult

    raw = await _run_fit_gpu.remote.aio(workspace_id, request.model_dump(mode="json"), dict(pins))
    return Applied[ModelFitResult].model_validate(raw)
