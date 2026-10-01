"""Typed action exceptions.

Discriminator: an exception means the action FAILED TO EXECUTE ITS
CONTRACT — state must not change, and the failure is a property of the
attempt (recorded in the attempt log), never of the world. Negative
findings (no estimable treatments, zero observations extracted) are NOT
exceptions: those actions succeed and simply withhold their enabling
artifact while producing a report.

Temporal maps these to non-retryable ApplicationErrors; transient infra
failures (network, OOM, Modal preemption) stay as ordinary exceptions and
are retried by policy without the navigator ever seeing them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.json_types import JsonObject


class ActionExecutionError(Exception):
    """An action failed to produce its output artifacts.

    ``diagnostics`` carries whatever partial information the attempt yielded
    (e.g. sampler diagnostics from a diverged fit) — informative for the
    navigator, but never persisted as a poisoned pseudo-artifact.
    """

    def __init__(self, message: str, *, diagnostics: JsonObject | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ModelFitError(ActionExecutionError):
    """Fitting failed to produce a usable posterior."""
