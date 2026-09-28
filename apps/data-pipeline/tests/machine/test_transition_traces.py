"""Journal-owned durable LLM traces."""

import asyncio
import json

import pytest

from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.store import collect_run_traces, read_episode_trace, trace_log_path
from nof1_causal_lab.machine.temporal.activities import journal_activity, read_branch_activity
from nof1_causal_lab.machine.temporal.messages import JournalInput, ReadBranchInput
from nof1_causal_lab.utils import data as data_module
from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.llm import LLMTrace, TraceMessage, TraceUsage

pytestmark = pytest.mark.contract


@pytest.fixture
def data_root(monkeypatch, tmp_path):
    root = tmp_path / "data"
    monkeypatch.setattr(data_module, "_DATA_URI", str(root))
    return root


def _trace(model: str = "openrouter/test-model") -> LLMTrace:
    return LLMTrace(
        messages=[TraceMessage(role="assistant", content="Done.")],
        model=model,
        usage=TraceUsage(input_tokens=3, output_tokens=5),
        total_time_seconds=0.25,
    )


def _scratch_trace(workspace_id: str, seq: int, subroutine_id: str) -> str:
    path = storage.join(
        data_module.scratch_run_dir(workspace_id, f"seq-{seq:06d}"),
        "llm",
        subroutine_id,
        "trace.json",
    )
    storage.write_text(path, _trace().model_dump_json())
    return path


def test_collects_finalized_traces_and_validates_commit_paths(data_root):
    del data_root
    _scratch_trace("ws-trace", 1, "zeta")
    _scratch_trace("ws-trace", 1, "alpha")
    logs = collect_run_traces("ws-trace", 1)
    assert list(logs) == ["traces/alpha.json", "traces/zeta.json"]
    with pytest.raises(ValueError, match="Invalid transition trace subroutine id"):
        trace_log_path("../scratch")


def test_raised_transition_discovers_trace_and_retry_no_longer_needs_scratch(data_root):
    del data_root
    from nof1_causal_lab.flows.runtime_events import emit_transition_event

    emit_transition_event("ws-trace", "raw_data", "completed")
    base = asyncio.run(read_branch_activity(ReadBranchInput(workspace_id="ws-trace")))
    emit_transition_event("ws-trace", "latent_structure", "failed")
    _scratch_trace("ws-trace", 1, "latent-structure")
    input = JournalInput(
        workspace_id="ws-trace",
        expected_head=base.commit_id,
        event_cursor=base.event_cursor,
        seq=1,
        action="edit_model",
        operation_id="latent_structure",
        inputs={},
        status="raised",
        error_type="LLMSubroutineError",
        error_message="validation failed",
        resume=None,
    )

    commit_id = asyncio.run(journal_activity(input))
    repository = StudyRepository("ws-trace")
    captured = json.loads(repository.read_file(commit_id, "logs/events.json"))
    assert [event["transition_id"] for event in captured] == ["latent_structure"]
    assert repository.head() == base.commit_id
    storage.rm_tree(data_module.scratch_events_dir("ws-trace"))
    storage.rm_tree(data_module.scratch_run_dir("ws-trace", "seq-000001"))
    asyncio.run(journal_activity(input))

    record = StudyRepository("ws-trace").read_attempt(1)
    assert record is not None
    assert record.status == "raised"
    assert record.trace_ids == ["latent-structure"]
    assert read_episode_trace("ws-trace", commit_id, "latent-structure")["model"] == (
        "openrouter/test-model"
    )
