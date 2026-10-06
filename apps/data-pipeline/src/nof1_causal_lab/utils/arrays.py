"""Content-addressed numerical values for native distribution constructor trees."""

import hashlib
import io
import re

import numpy as np

from nof1_causal_lab.utils import storage


def encode_array(values: np.ndarray) -> tuple[str, bytes]:
    """Encode the portable bytes used by both storage and compute transfers."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values), allow_pickle=False)
    payload = buffer.getvalue()
    identity = hashlib.sha256(payload).hexdigest()
    return identity, payload


def decode_array(identity: str, payload: bytes) -> np.ndarray:
    """Verify content identity before loading a numerical value."""
    if hashlib.sha256(payload).hexdigest() != identity:
        raise ValueError("Stored numerical array failed its content identity check")
    return np.load(io.BytesIO(payload), allow_pickle=False)


def write_array(directory: str, values: np.ndarray) -> str:
    """Persist a content-addressed NumPy array only when its encoded content is not already stored."""
    identity, payload = encode_array(values)
    storage.makedirs(directory)
    path = storage.join(directory, f"{identity}.npy")
    if not storage.exists(path):
        with storage.open_file(path, "wb") as stream:
            stream.write(payload)
    return identity


def read_array(directory: str, identity: str) -> np.ndarray:
    """Load a numerical array by a validated SHA-256 identity and verify its encoded content."""
    if re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise ValueError("Invalid numerical array identity")
    with storage.open_file(storage.join(directory, f"{identity}.npy"), "rb") as stream:
        payload = stream.read()
    return decode_array(identity, payload)
