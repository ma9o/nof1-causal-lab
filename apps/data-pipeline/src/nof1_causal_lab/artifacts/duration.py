"""Positive fixed durations, retaining the authored wire spelling exactly."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema

type DurationUnit = Literal["s", "m", "h", "d", "w"]
_UNIT_SECONDS: dict[DurationUnit, int] = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
_DURATION_RE = re.compile(r"([0-9]+)(s|m|h|d|w)")


@dataclass(frozen=True, init=False)
class Duration:
    """A fixed interval; days and weeks mean exactly 86400 and 604800 seconds."""

    source: str
    count: int
    unit: DurationUnit

    def __init__(self, source: str) -> None:
        """Parse a positive integer duration with a seconds, minutes, hours, days, or weeks suffix.

        Args:
            source: Compact duration such as ``30m`` or ``7d``.

        Raises:
            ValueError: The spelling does not match a supported integer duration or
                the duration is zero.
        """
        match = _DURATION_RE.fullmatch(source)
        if match is None:
            raise ValueError(
                f"Invalid duration: {source!r}. Expected <int><unit> with s, m, h, d, w"
            )
        count = int(match[1])
        if count == 0:
            raise ValueError("Duration must be positive (got 0)")
        unit = next(unit for unit in _UNIT_SECONDS if unit == match[2])
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "unit", unit)

    @property
    def seconds(self) -> int:
        """Duration expressed as an integer number of seconds."""
        return self.count * _UNIT_SECONDS[self.unit]

    @property
    def days(self) -> float:
        """Lower the exact interval to the numerical engine's day axis."""
        return self.seconds / 86400

    @override
    def __str__(self) -> str:
        return self.source

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: object, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Accept parsed durations or duration strings and serialize them with their original spelling."""
        parsed = core_schema.no_info_after_validator_function(cls, core_schema.str_schema())
        return core_schema.json_or_python_schema(
            json_schema=parsed,
            python_schema=core_schema.union_schema([core_schema.is_instance_schema(cls), parsed]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                str, return_schema=core_schema.str_schema()
            ),
        )


__all__ = ["Duration"]
