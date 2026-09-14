"""Content-addressed numerical values for native distribution constructor trees."""

import hashlib
import io
import re

import numpy as np

from nof1_causal_lab.utils import storage


def write_array(directory: str, values: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values), allow_pickle=False)
    payload = buffer.getvalue()
    identity = hashlib.sha256(payload).hexdigest()
    storage.makedirs(directory)
    path = storage.join(directory, f"{identity}.npy")
    if not storage.exists(path):
        with storage.open_file(path, "wb") as stream:
            stream.write(payload)
    return identity


def read_array(directory: str, identity: str) -> np.ndarray:
    if re.fullmatch(r"[0-9a-f]{64}", identity) is None:
        raise ValueError("Invalid numerical array identity")
    with storage.open_file(storage.join(directory, f"{identity}.npy"), "rb") as stream:
        payload = stream.read()
    if hashlib.sha256(payload).hexdigest() != identity:
        raise ValueError("Stored numerical array failed its content identity check")
    return np.load(io.BytesIO(payload), allow_pickle=False)
