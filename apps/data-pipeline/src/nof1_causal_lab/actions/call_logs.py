"""Collect execution messages and retain the same typed accumulator in Git."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_lab.actions.logs import (
    ActionLog,
    ExecutionMessage,
    FailureMessage,
    ProgressMessage,
    TraceLogMessage,
)
from nof1_causal_lab.actions.progress import read_events
from nof1_causal_lab.actions.progress_contracts import ProgressEvent
from nof1_causal_lab.study.store import collect_run_traces
from nof1_causal_lab.utils.llm import LLMTrace

if TYPE_CHECKING:
    from collections.abc import Mapping
    from uuid import UUID

    from nof1_causal_lab.study.history import StudyRepository
    from nof1_causal_lab.study.records import ActionMessage, FailedOutcome

_PROGRESS = TypeAdapter(tuple[ProgressEvent, ...])


def assemble_call_log(
    messages: tuple[ActionMessage, ...],
    logs: Mapping[str, bytes],
    *,
    at: datetime,
    failure: FailedOutcome | None,
) -> ActionLog:
    """All publication paths use this one owner for the complete execution log."""
    entries: list[ExecutionMessage] = list(messages)
    for path, raw in logs.items():
        if path == "progress.json":
            entries.extend(
                ProgressMessage(
                    timestamp=datetime.fromtimestamp(
                        int(event.cursor.split("-", 1)[0]) / 1_000_000_000, UTC
                    ),
                    progress=event,
                )
                for event in _PROGRESS.validate_json(raw)
            )
        elif path.startswith("traces/"):
            entries.append(
                TraceLogMessage(
                    timestamp=at,
                    trace_id=path.removeprefix("traces/").removesuffix(".json"),
                    trace=LLMTrace.model_validate_json(raw),
                )
            )
    if failure is not None:
        entries.append(FailureMessage(timestamp=at, failure=failure))
    ordered = sorted(entries, key=lambda entry: entry.timestamp)
    return ActionLog(messages=tuple(ordered))


def collect_call_log(
    workspace_id: str,
    *,
    seq: int,
    attempt_id: UUID,
    messages: tuple[ActionMessage, ...],
    at: datetime,
    failure: FailedOutcome | None = None,
) -> ActionLog:
    """Gather progress events and LLM traces into the log retained for one attempt.

    Args:
        workspace_id: Workspace containing the attempt's scratch events and traces.
        seq: Study sequence number identifying the run's trace directory.
        attempt_id: Attempt whose progress events are included.
        messages: Action messages already emitted by the workflow.
        at: Timestamp to use when assembling the final log entries.
        failure: Rejection or exception to include in the log, if the attempt failed.

    Returns:
        Combined action log ready to be published with the attempt.
    """
    return assemble_call_log(
        messages,
        {
            **collect_run_traces(workspace_id, seq),
            "progress.json": _PROGRESS.dump_json(tuple(read_events(workspace_id, attempt_id))),
        },
        at=at,
        failure=failure,
    )


def read_call_log(repository: StudyRepository, commit_id: str) -> ActionLog:
    """Read the action log retained at an exact study commit."""
    return ActionLog.model_validate_json(repository.read_file(commit_id, "logs/messages.json"))
