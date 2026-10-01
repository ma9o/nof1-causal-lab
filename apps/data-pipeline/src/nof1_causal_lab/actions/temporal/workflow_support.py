"""Replay-safe support shared by the study's workflows."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Any

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

from nof1_causal_lab.actions.temporal.messages import ProgressEventInput

if TYPE_CHECKING:
    from nof1_causal_lab.actions.progress import ProgressEvent

EVENT_TIMEOUT = timedelta(seconds=30)
EVENT_RETRY = RetryPolicy(initial_interval=timedelta(seconds=1), maximum_attempts=5)
type TemporalFailureDiagnostics = dict[str, Any]


async def emit_progress(workspace_id: str, event: ProgressEvent) -> None:
    """Publish one live progress event using the common activity policy."""
    await workflow.execute_activity(
        "emit_progress_event_activity",
        ProgressEventInput(workspace_id=workspace_id, event=event),
        start_to_close_timeout=EVENT_TIMEOUT,
        retry_policy=EVENT_RETRY,
    )


def temporal_failure_details(
    exc: BaseException,
) -> tuple[str, str, TemporalFailureDiagnostics]:
    """Unwrap a Temporal cause chain into a stable runtime error payload."""
    cause = exc
    while not isinstance(cause, ApplicationError):
        next_cause = cause.__cause__
        if next_cause is None:
            break
        cause = next_cause
    if isinstance(cause, ApplicationError):
        diagnostics = (
            cause.details[0] if cause.details and isinstance(cause.details[0], dict) else {}
        )
        return cause.type or "ApplicationError", cause.message, dict(diagnostics)
    return type(cause).__name__, str(cause), {}
