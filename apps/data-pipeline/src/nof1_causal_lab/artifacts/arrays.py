"""Exact numerical buffers and vector selections owned by one action result."""

from __future__ import annotations

from math import prod
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, model_validator

from nof1_causal_lab.artifacts.base import Value


class NumericalArray(Value):
    """A lossless row-major buffer, including its original dtype and dimensions."""

    dtype: str
    shape: tuple[Annotated[int, Field(ge=0)], ...]
    values: tuple[bool | int | FiniteFloat | Literal["nan", "+inf", "-inf"], ...]

    @model_validator(mode="after")
    def complete_buffer(self) -> Self:
        """Own the relationship between dimensions and retained scalars."""
        if len(self.values) != prod(self.shape):
            raise ValueError("Numerical buffer length must match its shape")
        return self


class ArrayVector(Value):
    """One vector selected from the result's numerical buffers without copying its values."""

    array_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    indices: tuple[Annotated[int, Field(ge=0)] | None, ...]
    mask: ArrayVector | None = None
    start: int = Field(default=0, ge=0)
    stop: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def one_vector_axis(self) -> Self:
        """Exactly one axis remains free in a stored vector selection."""
        if self.indices.count(None) != 1:
            raise ValueError("A vector selection must have exactly one free axis")
        if self.stop is not None and self.stop < self.start:
            raise ValueError("A vector range must be ordered")
        return self


type ScalarValues = tuple[FiniteFloat | None, ...] | ArrayVector
