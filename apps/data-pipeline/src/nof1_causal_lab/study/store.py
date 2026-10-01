"""Immutable artifact payloads and access to commit-local execution records.

Git owns study state and history in ``study/history.git``. Artifact trees
contain JSON payloads and input references; ``store/`` contains content-addressed
numerical arrays and external tables. Git OIDs identify exact scientific inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.parquet as pq

from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.study.git_objects import object_tree, open_repository, read_file, write_tree
from nof1_causal_lab.study.state import ArtifactRecord, StudyState
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

if TYPE_CHECKING:
    import polars as pl

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.json_types import JsonObject


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def read_model(store: ArtifactStore, revision: GitOid) -> ModelSpec:
    """Decode a pinned model, keeping its numerical arrays lazy."""
    from functools import cache

    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    return ModelSpec.model_validate(
        store.read_json_file("model", revision, "model.json"),
        context={"distribution_array_loader": cache(store.read_array)},
    )


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Immutable Git trees for payloads and input references; large values live in a blob store."""

    def __init__(self, workspace_id: str, *, repository_path: Path | None = None) -> None:
        self.workspace_id = workspace_id
        self._root = data_module.store_dir(workspace_id)
        self.repo = open_repository(workspace_id, repository_path)

    def write_array(self, values) -> str:
        from nof1_causal_lab.utils.arrays import write_array

        return write_array(storage.join(self._root, "arrays"), values)

    def read_array(self, identity: str):
        from nof1_causal_lab.utils.arrays import read_array

        return read_array(storage.join(self._root, "arrays"), identity)

    def write_artifact(
        self,
        artifact_id: ArtifactId,
        *,
        derived_from: dict[ArtifactId, GitOid],
        produced_by: str | None,
        json_files: JsonObject | None = None,
        parquet_files: dict[str, pl.DataFrame | pa.Table] | None = None,
        created_at: str | None = None,
    ) -> ArtifactRecord:
        """Write one content-addressed tree; Git assigns its immutable identity."""
        model_inputs: dict[str, str] = {}
        consumed_model_inputs: dict[str, str] = {}
        if artifact_id == "model":
            from nof1_causal_lab.artifacts.model_spec import ModelSpec
            from nof1_causal_lab.models.model_inputs import input_fingerprints
            from nof1_causal_lab.study.artifact_files import json_filename

            assert json_files is not None
            value = ModelSpec.model_validate(json_files[json_filename("model", "model")])
            model_inputs = input_fingerprints(value)
        elif "model" in derived_from:
            from nof1_causal_lab.study.model_dependencies import MODEL_INPUTS

            purpose = MODEL_INPUTS[artifact_id]
            source = self.read_meta("model", derived_from["model"])
            consumed_model_inputs = {purpose: source.model_inputs[purpose]}

        metadata = {
            "artifact_id": artifact_id,
            "derived_from": derived_from,
            "produced_by": produced_by,
            "created_at": created_at if created_at is not None else utc_now_iso(),
            "model_inputs": model_inputs,
            "consumed_model_inputs": consumed_model_inputs,
        }

        files = {
            name: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            for name, value in (json_files or {}).items()
        }
        external = {}
        for name, frame in (parquet_files or {}).items():
            buffer = io.BytesIO()
            if isinstance(frame, pa.Table):
                pq.write_table(frame, buffer, compression="zstd")
            else:
                frame.write_parquet(buffer)
            payload = buffer.getvalue()
            identity = hashlib.sha256(payload).hexdigest()
            path = Path(self._root) / "blobs" / identity
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(payload)
            external[name] = identity
        files["meta.json"] = json.dumps(metadata, sort_keys=True).encode()
        if external:
            files["external.json"] = json.dumps(external, sort_keys=True).encode()
        revision = write_tree(self.repo, files)
        self.repo.references.create(
            f"refs/artifacts/{artifact_id}/{revision}", revision, force=True
        )
        return ArtifactRecord.model_validate({**metadata, "revision": str(revision)})

    def list_revisions(self, artifact_id: ArtifactId) -> list[GitOid]:
        prefix = f"refs/artifacts/{artifact_id}/"
        revisions = [
            GitOid(ref.removeprefix(prefix))
            for ref in self.repo.references
            if ref.startswith(prefix)
        ]
        return sorted(revisions, key=lambda oid: (self.read_meta(artifact_id, oid).created_at, oid))

    def read_meta(self, artifact_id: ArtifactId, revision: str) -> ArtifactRecord:
        metadata = json.loads(read_file(self.repo, revision, "meta.json"))
        info = ArtifactRecord(**metadata, revision=GitOid(revision))
        if info.artifact_id != artifact_id:
            raise ValueError(f"Git object {revision} is {info.artifact_id}, not {artifact_id}")
        return info

    def read_json_file(self, artifact_id: ArtifactId, revision: str, name: str) -> Any:
        self.read_meta(artifact_id, revision)
        return json.loads(read_file(self.repo, revision, name))

    def filenames(self, artifact_id: ArtifactId, revision: str) -> list[str]:
        self.read_meta(artifact_id, revision)
        tree = object_tree(self.repo, revision)
        names = [
            entry.name
            for entry in tree
            if entry.name is not None and entry.name not in {"meta.json", "external.json"}
        ]
        if "external.json" in tree:
            names.extend(json.loads(read_file(self.repo, revision, "external.json")))
        return sorted(names)

    def file_path(self, artifact_id: ArtifactId, revision: str, name: str) -> str:
        self.read_meta(artifact_id, revision)
        tree = object_tree(self.repo, revision)
        external = (
            json.loads(read_file(self.repo, revision, "external.json"))
            if "external.json" in tree
            else {}
        )
        if name in external:
            path = Path(self._root) / "blobs" / external[name]
            if hashlib.sha256(path.read_bytes()).hexdigest() != external[name]:
                raise ValueError("Stored table failed its content identity check")
            return str(path)
        payload = read_file(self.repo, revision, name)
        path = Path(data_module.cache_dir(self.workspace_id)) / "git" / revision / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        return str(path)

    def read_parquet_file(self, artifact_id: ArtifactId, revision: str, name: str) -> pl.DataFrame:
        import polars as pl

        return pl.read_parquet(self.file_path(artifact_id, revision, name))

    def read_parquet_table(self, artifact_id: ArtifactId, revision: str, name: str) -> pa.Table:
        return pq.read_table(self.file_path(artifact_id, revision, name))


# ---------------------------------------------------------------------------
# Attempt logs and derived state
# ---------------------------------------------------------------------------


_TRACE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def trace_log_path(subroutine_id: str) -> str:
    """A trace's path within the owning commit tree."""
    if _TRACE_ID.fullmatch(subroutine_id) is None:
        raise ValueError(f"Invalid attempt trace subroutine id: {subroutine_id!r}")
    return f"traces/{subroutine_id}.json"


def collect_run_traces(workspace_id: str, seq: int) -> dict[str, bytes]:
    """Collect finalized traces before scratch is swept; publication happens in Git."""
    llm_root = storage.join(data_module.scratch_run_dir(workspace_id, f"seq-{seq:06d}"), "llm")
    logs: dict[str, bytes] = {}
    for subroutine_root in sorted(storage.listdir(llm_root)):
        subroutine_id = subroutine_root.rstrip("/").rsplit("/", 1)[-1]
        source = storage.join(subroutine_root, "trace.json")
        if storage.exists(source):
            logs[trace_log_path(subroutine_id)] = storage.read_text(source).encode()
    return logs


def read_attempt_trace(workspace_id: str, commit_id: str, subroutine_id: str) -> Any:
    from nof1_causal_lab.study.history import StudyRepository

    repository = StudyRepository(workspace_id)
    try:
        return json.loads(repository.read_file(commit_id, f"logs/{trace_log_path(subroutine_id)}"))
    except KeyError as exc:
        raise FileNotFoundError(f"No trace {subroutine_id} at {commit_id}") from exc


def read_current_state(workspace_id: str, *, branch: str = "main") -> StudyState:
    from nof1_causal_lab.study.history import StudyRepository

    repository = StudyRepository(workspace_id)
    return repository.state(repository.head(branch))
