"""Shared Modal images and the hosted read-only facade."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from fastapi import FastAPI


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
    image=cpu_image,
    env={"READ_ONLY_FACADE": "1"},
    secrets=[secrets],
)
@modal.asgi_app()
def read_facade() -> FastAPI:
    """The hosted viewer's backend: journal reads over the R2 store, no moves."""
    from nof1_causal_lab.read_facade import create_read_facade_app

    return create_read_facade_app()
