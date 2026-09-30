"""Shared, evictable content cache. Atomic publication keeps the first result."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile

from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.data import data_root


def content_key(request: object) -> str:
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def cache_path(kind: str, key: str) -> str:
    return storage.join(data_root(), ".preparation-cache", kind, key)


def read(path: str) -> bytes | None:
    """Read once; eviction before or during the read is a cache miss."""
    try:
        with storage.open_file(path, "rb") as stored:
            return stored.read()
    except FileNotFoundError:
        return None


def publish(path: str, value: bytes) -> bytes:
    """Publish a fully validated value; return the winning value after a race."""
    if storage.is_remote():
        from botocore.exceptions import ClientError

        try:
            storage.get_fs().pipe_file(path, value, mode="create")
        except FileExistsError:
            pass
        except OSError as exc:
            # S3's conditional-create conflict is translated to OSError by s3fs
            # when the provider omits the optional Error.Condition field.
            cause = exc.__cause__
            if not isinstance(cause, ClientError) or cause.response["Error"]["Code"] not in {
                "PreconditionFailed",
                "412",
            }:
                raise
    else:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent) as temporary:
            temporary.write(value)
            temporary.flush()
            os.fsync(temporary.fileno())
            with suppress(FileExistsError):
                os.link(temporary.name, target)
    with storage.open_file(path, "rb") as stored:
        return stored.read()
