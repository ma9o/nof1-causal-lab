"""Shared helpers and constants for the pure SSM compilation pipeline."""

from __future__ import annotations


def axis_names_with_fallback(
    names: list[str] | None,
    *,
    expected: int,
    prefix: str,
) -> list[str]:
    """Return axis names with deterministic fallbacks when metadata is incomplete."""
    resolved = [str(name) for name in (names or []) if name]
    if len(resolved) >= expected:
        return resolved[:expected]
    return resolved + [f"{prefix}_{idx}" for idx in range(len(resolved), expected)]
