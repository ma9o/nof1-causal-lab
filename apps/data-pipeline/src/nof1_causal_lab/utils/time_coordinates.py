"""Calendar bindings for model days; the epoch only serializes calendar-free histories."""

from datetime import UTC, datetime

SYNTHETIC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def serialization_origin(time_origin: datetime | None) -> datetime:
    """Return a UTC-naive origin for the canonical panel's datetime columns."""
    origin = SYNTHETIC_EPOCH if time_origin is None else time_origin
    return origin.astimezone(UTC).replace(tzinfo=None)
