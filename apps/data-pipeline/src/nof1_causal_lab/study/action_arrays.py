"""Resolve owned numerical views and index embedded law values for storage."""

from collections.abc import Mapping

import numpy as np

from nof1_causal_lab.artifacts.arrays import ArrayVector, NumericalArray, ScalarValues


def decode_array(value: NumericalArray) -> np.ndarray:
    """Decode one owned value for a scientific consumer."""
    return value.values


def resolve_vector(value: ScalarValues) -> tuple[float | None, ...]:
    """Select stored scalar values, applying the recorded observation mask."""
    if not isinstance(value, ArrayVector):
        return value
    indices = tuple(slice(None) if index is None else index for index in value.indices)
    values = value.array.values[indices][value.start : value.stop]
    if value.mask is not None:
        values = np.where(np.asarray(resolve_vector(value.mask), dtype=bool), values, np.nan)
    return tuple(float(item) if np.isfinite(item) else None for item in values)


def owned_arrays(value: object) -> Mapping[str, NumericalArray]:
    """Index embedded numerical owners only at the persistence boundary."""
    if isinstance(value, Mapping):
        if set(value) == {"npy"}:
            array = NumericalArray.model_validate(value)
            return {array.identity: array}
        return {ref: array for child in value.values() for ref, array in owned_arrays(child).items()}
    if isinstance(value, (list, tuple)):
        return {ref: array for child in value for ref, array in owned_arrays(child).items()}
    return {}
