"""Tests for replay-safe study workflow helpers."""

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from temporalio.exceptions import ApplicationError, FailureError

from nof1_causal_lab.actions.progress import StepError, StepEvent
from nof1_causal_lab.actions.temporal.workflow_support import (
    EVENT_RETRY,
    EVENT_TIMEOUT,
    emit_progress,
    temporal_failure_details,
)

pytestmark = pytest.mark.contract


class _WrappedError(FailureError):
    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.__cause__ = cause


def test_temporal_failure_details_unwraps_application_error_and_copies_diagnostics() -> None:
    diagnostics = {"reason": "invalid"}
    error = _WrappedError(
        _WrappedError(
            ApplicationError(
                "model failed",
                diagnostics,
                type="ModelCompileError",
                non_retryable=True,
            )
        )
    )

    error_type, message, extracted = temporal_failure_details(error)
    extracted["local"] = True

    assert error_type == "ModelCompileError"
    assert message == "model failed"
    assert diagnostics == {"reason": "invalid"}


def test_temporal_failure_details_uses_deepest_untyped_cause() -> None:
    assert temporal_failure_details(_WrappedError(ValueError("bad input"))) == (
        "ValueError",
        "bad input",
        {},
    )


def test_emit_progress_uses_shared_activity_policy(monkeypatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def execute_activity(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))

    from nof1_causal_lab.actions.temporal import workflow_support

    monkeypatch.setattr(workflow_support.workflow, "execute_activity", execute_activity)
    step = StepEvent(
        attempt_id=uuid4(),
        step="extraction",
        status="failed",
        error=StepError(type="ValueError", message="bad input"),
    )

    asyncio.run(emit_progress("workspace-1", step))

    (activity_name, payload), kwargs = calls[0]
    assert activity_name == "emit_progress_event_activity"
    assert payload.workspace_id == "workspace-1"
    assert payload.event == step
    assert kwargs == {
        "start_to_close_timeout": EVENT_TIMEOUT,
        "retry_policy": EVENT_RETRY,
    }
