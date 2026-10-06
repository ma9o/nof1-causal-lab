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
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pygit2

from nof1_causal_lab.artifacts.identity import ArtifactId, GitOid
from nof1_causal_lab.study.errors import StudyLookupError
from nof1_causal_lab.study.git_objects import object_tree, open_repository, read_file, write_tree
from nof1_causal_lab.study.state import ArtifactRecord
from nof1_causal_lab.study.view_models import DataPoint, DataRef, DataSeries, Dataset
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import ArrayLike
    from pydantic import BaseModel, TypeAdapter

    from nof1_causal_lab.artifacts.measurements import ObservationRecord
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.observation_data import ObservationDataset
    from nof1_causal_lab.artifacts.question import QuestionSpec
    from nof1_causal_lab.json_types import JsonObject


_CODE_DIGEST = hashlib.sha256(
    b"".join(path.read_bytes() for path in sorted(Path(__file__).parents[1].rglob("*.py")))
).hexdigest()


def cached_read[T](
    workspace_id: str, key: tuple[str, ...], adapter: TypeAdapter[T], render: Callable[[], T]
) -> tuple[bytes, bool]:
    """One atomic cache for projections of immutable inputs and current package code."""
    digest = hashlib.sha256("\0".join((_CODE_DIGEST, *key)).encode()).hexdigest()
    path = Path(data_module.cache_dir(workspace_id)) / "reads" / f"{digest}.json"
    reused = path.exists()
    if not reused:
        body = adapter.dump_json(render(), by_alias=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{digest}.{uuid4().hex}.partial")
        partial.write_bytes(body)
        partial.replace(path)
    return path.read_bytes(), reused


def cached_value[T](
    workspace_id: str, key: tuple[str, ...], adapter: TypeAdapter[T], render: Callable[[], T]
) -> tuple[T, bool]:
    """Read or render a typed cached value and return it with a cache-reuse flag."""
    body, reused = cached_read(workspace_id, key, adapter, render)
    return adapter.validate_json(body), reused


def utc_now_iso() -> str:
    """Read the current UTC clock as a timezone-qualified ISO timestamp."""
    return datetime.now(tz=UTC).isoformat()


def read_model(store: ArtifactStore, revision: GitOid) -> ModelSpec:
    """Decode a pinned model, keeping its numerical arrays lazy."""
    from functools import cache

    from nof1_causal_lab.artifacts.model_spec import ModelSpec

    return ModelSpec.model_validate(
        store.read_json_file("model", revision, "model.json"),
        context={"distribution_array_loader": cache(store.read_array)},
    )


def read_question(store: ArtifactStore, revision: GitOid) -> QuestionSpec:
    """Decode the study question pinned at one revision."""
    from nof1_causal_lab.artifacts.question import QuestionSpec

    return store.read_value("question", revision, "question.json", QuestionSpec)


# ---------------------------------------------------------------------------
# Artifact store
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Immutable Git trees for payloads and input references; large values live in a blob store."""

    def __init__(self, workspace_id: str, *, repository_path: Path | None = None) -> None:
        """Bind the workspace's blob store and open its local artifact repository."""
        self.workspace_id = workspace_id
        self._root = data_module.store_dir(workspace_id)
        self.repo = open_repository(workspace_id, repository_path)

    def write_array(self, values: ArrayLike) -> str:
        """Persist numerical values in the workspace array store and return their content identity."""
        from nof1_causal_lab.utils.arrays import write_array

        return write_array(storage.join(self._root, "arrays"), np.asarray(values))

    def read_array(self, identity: str) -> np.ndarray:
        """Load and verify a numerical array using its retained content identity."""
        from nof1_causal_lab.utils.arrays import read_array

        return read_array(storage.join(self._root, "arrays"), identity)

    def write_report(self, report: BaseModel) -> GitOid:
        """Retain the computed report as a Git blob for the action's publication."""
        return GitOid(str(self.repo.create_blob(report.model_dump_json(round_trip=True).encode())))

    def read_report[ReportT: BaseModel](self, revision: GitOid, target: type[ReportT]) -> ReportT:
        """Parse a retained Git report blob with the caller's expected report schema."""
        payload = self.repo[pygit2.Oid(hex=revision)].peel(pygit2.Blob).data
        return target.model_validate_json(payload)

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
        metadata = {
            "artifact_id": artifact_id,
            "derived_from": derived_from,
            "produced_by": produced_by,
            "created_at": created_at if created_at is not None else utc_now_iso(),
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

    def read_record(self, revision: str) -> ArtifactRecord:
        """Read an artifact's identity and provenance without presupposing its kind."""
        metadata = json.loads(read_file(self.repo, revision, "meta.json"))
        return ArtifactRecord(**metadata, revision=GitOid(revision))

    def read_meta(self, artifact_id: ArtifactId, revision: str) -> ArtifactRecord:
        """Read artifact metadata and reject revisions belonging to a different artifact kind."""
        info = self.read_record(revision)
        if info.artifact_id != artifact_id:
            raise StudyLookupError(
                f"Git object {revision} is {info.artifact_id}, not {artifact_id}"
            )
        return info

    def read_value[ValueT: BaseModel](
        self, artifact_id: ArtifactId, revision: str, name: str, target: type[ValueT]
    ) -> ValueT:
        """Check artifact ownership and parse a named JSON payload with its production schema."""
        self.read_meta(artifact_id, revision)
        return target.model_validate_json(read_file(self.repo, revision, name))

    def read_json_file(self, artifact_id: ArtifactId, revision: str, name: str) -> JsonObject:
        """Check artifact ownership and read a named JSON object from the selected revision."""
        self.read_meta(artifact_id, revision)
        value: JsonObject = json.loads(read_file(self.repo, revision, name))
        return value

    def file_path(self, artifact_id: ArtifactId, revision: str, name: str) -> str:
        """Materialize an artifact file locally, verifying external blob content identities."""
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
        """Read an artifact's retained Parquet file into a Polars dataframe."""
        import polars as pl

        return pl.read_parquet(self.file_path(artifact_id, revision, name))

    def read_parquet_table(self, artifact_id: ArtifactId, revision: str, name: str) -> pa.Table:
        """Read an artifact's retained Parquet file as an Arrow table with its schema metadata."""
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
    """Collect the accumulated trace snapshots before scratch is swept."""
    llm_root = storage.join(data_module.scratch_run_dir(workspace_id, f"seq-{seq:06d}"), "llm")
    logs: dict[str, bytes] = {}
    for subroutine_root in sorted(storage.listdir(llm_root)):
        subroutine_id = subroutine_root.rstrip("/").rsplit("/", 1)[-1]
        source = storage.join(subroutine_root, "trace.json")
        if storage.exists(source):
            logs[trace_log_path(subroutine_id)] = storage.read_text(source).encode()
    return logs


def read_payload(store: ArtifactStore, artifact_id: ArtifactId, revision: str) -> BaseModel:
    """Parse an immutable artifact's primary JSON payload while keeping distribution arrays lazy."""
    from functools import cache

    from pydantic import BaseModel, TypeAdapter

    from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS
    from nof1_causal_lab.study.artifact_files import artifact_file_spec

    filename = next(iter(artifact_file_spec(artifact_id).json_files.values()))
    return TypeAdapter[BaseModel](ARTIFACT_CONTRACTS[artifact_id]).validate_python(
        store.read_json_file(artifact_id, revision, filename),
        context={"distribution_array_loader": cache(store.read_array)},
    )


def observation_sample(panel: pl.DataFrame) -> tuple[ObservationRecord, ...]:
    """Parse stored rows before their compact projection."""
    from pydantic import TypeAdapter

    from nof1_causal_lab.artifacts.measurements import ObservationRecord

    sample = []
    for row in panel.head(20).to_dicts():
        record = {
            key: value for key, value in row.items() if key in ObservationRecord.__annotations__
        }
        for key in ("anchor_time", "support_start", "support_end"):
            if record.get(key) is not None:
                record[key] = str(record[key])
        sample.append(TypeAdapter(ObservationRecord).validate_python(record))
    return tuple(sample)


def read_dataset(
    source: DataRef[GitOid, int],
    observations: ObservationDataset,
) -> Dataset:
    """Project selected observations into per-indicator dated series with their exact source reference."""
    frame = observations.recorded.frame.with_columns(
        pl.col("anchor_time", "support_start", "support_end").dt.replace_time_zone("UTC")
    )
    series = {
        variable.id: DataSeries(
            variable=variable,
            time_origin=observations.time_origin,
            points=tuple(
                DataPoint.model_validate(row)
                for row in frame.filter(pl.col("indicator_id") == variable.id)
                .select("anchor_time", "support_start", "support_end", "value")
                .iter_rows(named=True)
            ),
        )
        for variable in observations.variables
    }

    return Dataset(source=source, series=series)
