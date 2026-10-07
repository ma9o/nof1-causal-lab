"""Encode execution buffers once and resolve their stored vector selections."""

from collections.abc import Iterable, Mapping
from typing import Literal, cast

import numpy as np

from nof1_causal_lab.artifacts.arrays import ArrayVector, NumericalArray, ScalarValues
from nof1_causal_lab.json_types import JsonValue
from nof1_causal_lab.study.store import ArtifactStore


def array_value(values: np.ndarray) -> NumericalArray:
    """Retain shape, dtype and every scalar, distinguishing NaN from either infinity."""

    def scalar(value: np.generic) -> bool | int | float | Literal["nan", "+inf", "-inf"]:
        item = cast("bool | int | float", value.item())
        if isinstance(item, float) and not np.isfinite(item):
            return "nan" if np.isnan(item) else "+inf" if item > 0 else "-inf"
        return item

    if values.dtype.kind not in "bifu":
        raise TypeError("Action results support boolean, integer and floating numerical buffers")
    return NumericalArray(
        dtype=str(values.dtype),
        shape=tuple(values.shape),
        values=tuple(scalar(value) for value in values.reshape(-1)),
    )


def decode_array(value: NumericalArray) -> np.ndarray:
    """Decode an exact stored buffer for a subsequent scientific action."""
    return np.asarray(value.values, dtype=value.dtype).reshape(value.shape)


def result_arrays(store: ArtifactStore, references: Iterable[str]) -> Mapping[str, NumericalArray]:
    """Read execution buffers once while constructing the complete result."""
    return {ref: array_value(store.read_array(ref)) for ref in sorted(set(references))}


def resolve_vector(
    value: ScalarValues, arrays: Mapping[str, NumericalArray]
) -> tuple[float | None, ...]:
    """Select stored scalar values, applying the recorded observation mask."""
    if not isinstance(value, ArrayVector):
        return value
    indices = tuple(slice(None) if index is None else index for index in value.indices)
    values = decode_array(arrays[value.array_ref])[indices][value.start : value.stop]
    if value.mask is not None:
        mask = np.asarray(resolve_vector(value.mask, arrays), dtype=bool)
        values = np.where(mask, values, np.nan)
    return tuple(float(item) if np.isfinite(item) else None for item in values)


def array_references(value: JsonValue) -> frozenset[str]:
    """Find the numerical buffers referenced by a serialized scientific model."""
    if isinstance(value, dict):
        own = (str(value["array_ref"]),) if "array_ref" in value else ()
        return frozenset(
            (*own, *(ref for child in value.values() for ref in array_references(child)))
        )
    if isinstance(value, list):
        return frozenset(ref for child in value for ref in array_references(child))
    return frozenset[str]()


def check_result_vectors(value: JsonValue, arrays: Mapping[str, NumericalArray]) -> None:
    """Close every view over its owning buffers at the result publication boundary."""

    def length(vector: ArrayVector) -> int:
        array = arrays[vector.array_ref]
        if len(vector.indices) != len(array.shape) or any(
            index is not None and index >= size
            for index, size in zip(vector.indices, array.shape, strict=True)
        ):
            raise ValueError("Vector coordinates exceed the owned buffer")
        extent = array.shape[vector.indices.index(None)]
        stop = vector.stop if vector.stop is not None else extent
        if stop > extent or vector.start > extent:
            raise ValueError("Vector range exceeds the owned buffer")
        size = stop - vector.start
        if vector.mask is not None and length(vector.mask) != size:
            raise ValueError("Vector mask must align with its values")
        return size

    if isinstance(value, dict):
        if "array_ref" in value and "indices" in value:
            length(ArrayVector.model_validate(value))
        for child in value.values():
            check_result_vectors(child, arrays)
    elif isinstance(value, list):
        for child in value:
            check_result_vectors(child, arrays)
