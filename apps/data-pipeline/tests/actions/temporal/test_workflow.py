"""Content-named calls, atomic outcomes, and recovery from Git.

Only ingestion's LLM is stubbed. Model edits supply their definition directly;
fit and simulation readiness failures exercise the real execution boundary.
"""

import asyncio
import dataclasses
import json
import uuid
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import Response
from pydantic import TypeAdapter

from nof1_causal_lab.actions.contracts import (
    EditModelRequest,
    FitRequest,
    PrepareDataRequest,
    SetQuestionRequest,
    SimulateRequest,
)
from nof1_causal_lab.actions.results import ActionPoll, CompletedPoll, RunningPoll
from nof1_causal_lab.actions.temporal.workflow import StudyWorkflow
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.question import QuestionSpec
from nof1_causal_lab.study.history import StudyRepository
from nof1_causal_lab.study.records import Rejected
from nof1_causal_lab.study.store import ArtifactStore, read_model
from nof1_causal_lab.utils.openrouter_client import create_openrouter_client
from tests.helpers import graph_constructs

pytestmark = [pytest.mark.workflow, pytest.mark.timeout(60, method="thread")]
_QUESTION = "does exercise improve sleep?"
_PREPARATION: dict[str, Any] = {
    "source": {"files": ["observations.csv"]},
    "definition": {
        "default_window": "1d",
        "variables": [
            {
                "observation": {
                    "id": "indicator:sleep",
                    "name": "sleep_steps_proxy",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                "extraction": {
                    "kind": "computed",
                    "how_to_measure": "Read the steps column",
                    "source_columns": ["steps"],
                },
            }
        ],
    },
}


def _proposed_model() -> dict[str, Any]:
    return {
        "edges": [
            {
                "id": "edge:test-outcome-0",
                "cause": {
                    "id": "construct:sleep",
                    "name": "sleep",
                    "description": "sleep quality",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
                "effect": {
                    "id": "construct:unmeasured_outcome",
                    "name": "unmeasured_outcome",
                    "description": "Downstream response outside the measured test states.",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
                "description": "Test state affects an unmeasured downstream response",
            }
        ],
    }


def _measured_model() -> dict[str, Any]:
    model = _proposed_model()
    model["measurement_clock"] = "1d"
    graph_constructs(model)[0]["indicators"] = [
        {
            "observation": {
                "id": "indicator:sleep",
                "name": "sleep_steps_proxy",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
            },
            "construct_polarity": "positive",
        }
    ]
    return model


@pytest.fixture
def machine_env(monkeypatch, tmp_path):
    import nof1_causal_lab.utils.openrouter_client as openrouter_client
    from nof1_causal_lab.llm_specs import EmbeddedLLMSpec
    from nof1_causal_lab.utils import config as config_module
    from nof1_causal_lab.utils import data as data_module

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path / "data"))
    workspace_id = f"ws-{uuid.uuid4().hex[:8]}"
    input_root = Path(data_module.input_dir(workspace_id))
    input_root.mkdir(parents=True, exist_ok=True)
    (input_root / "observations.csv").write_text(
        "timestamp,steps\n2026-01-01T08:00:00,1000\n2026-01-02T08:00:00,2000\n"
    )

    config = config_module.get_config()
    monkeypatch.setattr(
        config_module,
        "get_config",
        lambda: dataclasses.replace(
            config,
            ingestion=dataclasses.replace(
                config.ingestion,
                llm=EmbeddedLLMSpec(harness="none", model="openrouter/mock-raw"),
            ),
        ),
    )

    async def fake_call_model(
        model_name, messages, *, client, tools=None, config=None, log_label=None
    ):
        del config, log_label
        tool_names = {tool.name for tool in tools or []}
        if "execute_python" in tool_names:
            if not any(
                message.get("role") == "tool" and message.get("name") == "execute_python"
                for message in messages
            ):
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-python",
                                "type": "function",
                                "function": {
                                    "name": "execute_python",
                                    "arguments": json.dumps(
                                        {
                                            "code": (
                                                "result_df = pl.read_csv(Path(DATA_DIR) / "
                                                "'0/observations.csv')\n"
                                                "result_df = result_df.with_columns("
                                                "pl.col('timestamp').str.strptime(pl.Datetime))"
                                            )
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    "completion": "",
                    "usage": {"input_tokens": 3, "output_tokens": 5, "reasoning_tokens": None},
                    "model": model_name,
                    "time": 0.25,
                    "stop_reason": "tool_calls",
                }
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-submit-table",
                            "type": "function",
                            "function": {
                                "name": "submit_table",
                                "arguments": json.dumps(
                                    {
                                        "column_descriptions_json": json.dumps(
                                            {
                                                "timestamp": "observation time",
                                                "steps": "step count",
                                            }
                                        )
                                    }
                                ),
                            },
                        }
                    ],
                },
                "completion": "",
                "usage": {"input_tokens": 3, "output_tokens": 5, "reasoning_tokens": None},
                "model": model_name,
                "time": 0.25,
                "stop_reason": "tool_calls",
            }
        raise AssertionError(f"Unexpected authoring call: {tool_names}")

    monkeypatch.setattr(openrouter_client, "call_model", fake_call_model)
    return workspace_id


def test_study_workflow_journey(machine_env, monkeypatch):
    workspace_id = machine_env

    async def scenario():
        from temporalio.testing import WorkflowEnvironment

        from nof1_causal_lab import study_api
        from nof1_causal_lab.actions.temporal.client import pydantic_data_converter
        from nof1_causal_lab.actions.temporal.worker import (
            build_model_checks_worker,
            build_openrouter_worker,
            build_worker,
        )

        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                create_openrouter_client() as transport,
                build_worker(env.client),
                build_model_checks_worker(env.client),
                build_openrouter_worker(env.client, transport),
                asyncio.timeout(30),
            ):
                monkeypatch.setenv("READ_ONLY_FACADE", "0")
                clients = study_api.TemporalClientProvider()
                monkeypatch.setattr(clients, "get", AsyncMock(return_value=env.client))
                handle = await study_api._study_handle(workspace_id, clients)

                async def execute(request):
                    response = await study_api._dispatch_action(workspace_id, request, clients)
                    if isinstance(response, RunningPoll):
                        polled = await handle.get_update_handle(
                            str(response.attempt_id),
                            result_type=cast("type[ActionPoll]", ActionPoll),
                        ).result()
                    elif isinstance(response, Response):
                        polled = TypeAdapter(ActionPoll).validate_json(bytes(response.body))
                    else:
                        polled = response
                    assert isinstance(polled, CompletedPoll)
                    assert polled.commit_id is not None
                    record = StudyRepository(workspace_id).record(polled.commit_id)
                    assert polled.attempt == record.record.attempt
                    assert polled.messages[0].label == f"{request.action.upper()}_STARTED"
                    assert polled.messages[-1].level == (
                        "info" if record.record.attempt.outcome.status == "applied" else "error"
                    )
                    return record

                def state(record=None):
                    repository = StudyRepository(workspace_id)
                    return repository.state(
                        record.commit_id if record is not None else repository.head()
                    )

                rejected = await execute(
                    EditModelRequest(expected_revision=None, model=ModelSpec())
                )
                assert isinstance(rejected.record.attempt.outcome, Rejected)
                assert rejected.record.attempt.outcome.reason == "input_unavailable"
                assert rejected.record.attempt.outcome.detail == "Set the study question first"

                question = QuestionSpec(text=_QUESTION, outcome="construct:sleep")
                root = await execute(SetQuestionRequest(question=question))
                assert root.record.attempt.outcome.status == "applied"
                assert state(root).has("question")
                again = await execute(SetQuestionRequest(question=question))
                assert again == root

                # Every edit defines the question's nodes; without them it is an invalid request.
                invalid = await execute(EditModelRequest(expected_revision=None, model=ModelSpec()))
                assert invalid.record.attempt.outcome.status == "rejected"
                assert invalid.record.attempt.outcome.reason == "scientific_inputs"
                prepared = await execute(PrepareDataRequest(input=_PREPARATION))
                assert prepared.record.attempt.outcome.status == "applied", prepared
                assert state(prepared).has("panel")
                assert set(state(prepared).current) == {"question", "raw_data", "panel"}
                from nof1_causal_lab.study.snapshots import ModelReader

                assert ModelReader(workspace_id, at=prepared.commit_id).data_profile is not None
                edited = await execute(
                    EditModelRequest(
                        expected_revision=None,
                        model=ModelSpec.model_validate(_measured_model()),
                    )
                )
                assert edited.record.attempt.outcome.status == "applied", edited
                saved_checks = ModelReader(workspace_id, at=edited.commit_id).checks
                assert saved_checks is not None
                model_revision = state(edited).current["model"].revision
                before = state().current

                # Both numerical actions reject incomplete definitions before any numerical work.
                for request in (
                    FitRequest(
                        model_revision=model_revision, panel_revision=before["panel"].revision
                    ),
                    SimulateRequest(
                        model_revision=model_revision, start="2026-01-01", horizon="1d"
                    ),
                ):
                    raised = await execute(request)
                    assert isinstance(raised.record.attempt.outcome, Rejected), raised
                    assert raised.record.attempt.outcome.reason == "scientific_inputs"
                    assert state().current == before

                status = await study_api.get_timeline(workspace_id, clients)
                previous_state = state()
                previous_run_id = handle.first_execution_run_id
                await handle.terminate()
                handle = await study_api._study_handle(workspace_id, clients)
                assert handle.first_execution_run_id != previous_run_id
                recovered = await study_api.get_timeline(workspace_id, clients)
                assert state() == previous_state
                assert recovered.attempts == status.attempts

                store = ArtifactStore(workspace_id)
                measured = read_model(store, model_revision)
                revised = measured.revised(
                    edges=(measured.edges[0].revised(description="Sleep shapes the response"),)
                )
                rewritten = await execute(
                    EditModelRequest(expected_revision=model_revision, model=revised)
                )
                assert rewritten.record.attempt.outcome.status == "applied", rewritten
                assert (
                    rewritten.record.seq == max(item.record.seq for item in recovered.attempts) + 1
                )
                records = StudyRepository(workspace_id).attempts()
                assert [record.record.attempt.outcome.status for record in records] == [
                    "rejected",
                    "applied",
                    "rejected",
                    "applied",
                    "applied",
                    "rejected",
                    "rejected",
                    "applied",
                ]
                assert [record.record.attempt.action for record in records] == [
                    "edit_model",
                    "set_question",
                    "edit_model",
                    "prepare_data",
                    "edit_model",
                    "fit",
                    "simulate",
                    "edit_model",
                ]
                assert records[4].record.attempt.action == "edit_model"
                assert records[4].record.attempt.request is not None
                assert records[4].record.attempt.request.model == ModelSpec.model_validate(
                    _measured_model()
                )
                assert records[5].record.attempt.action == "fit"
                assert records[5].record.attempt.request is not None
                assert records[5].record.attempt.request.model_revision == model_revision
                assert all("move" not in record.model_dump() for record in records)
                assert read_model(store, state(rewritten).current["model"].revision) == revised
                assert state(rewritten).current["question"] == state(root).current["question"]
                assert ModelReader(workspace_id, at=edited.commit_id).checks == saved_checks
                from nof1_causal_lab.study.view_models import ModelDiffRequest

                head = StudyRepository(workspace_id).head()
                comparison_request = ModelDiffRequest(
                    before=edited.commit_id,
                    after=rewritten.commit_id,
                    reasoning="Check the revised edge before deciding which model to fit.",
                )
                compared = await execute(comparison_request)
                assert compared.record.attempt.action == "model_diff"
                assert compared.record.attempt.outcome.status == "applied"
                assert StudyRepository(workspace_id).head() == head
                saved = await study_api.model_diff(
                    workspace_id, comparison_request.revised(reasoning=None), clients
                )
                assert isinstance(saved, Response)
                completion = CompletedPoll.model_validate_json(bytes(saved.body))
                assert completion.model_comparison is not None
                assert completion.model_comparison.before_model == measured
                assert completion.model_comparison.after_model == revised
                assert completion.attempt.request == comparison_request
                reverse = await execute(
                    ModelDiffRequest(before=rewritten.commit_id, after=edited.commit_id)
                )
                assert reverse.record.attempt.outcome.status == "applied"
                assert reverse.record.attempt.request.reasoning is None
                assert StudyRepository(workspace_id).head() == head
                timeline = await study_api.get_timeline(workspace_id, clients)
                assert {
                    (link.source_seq, link.argument)
                    for link in timeline.dependencies
                    if link.seq == compared.record.seq
                } == {(edited.record.seq, "before"), (rewritten.record.seq, "after")}
                await handle.signal(StudyWorkflow.close)
                await handle.result()
        finally:
            await env.shutdown()

    asyncio.run(scenario())


def test_timeline_reports_only_attempts_a_live_workflow_executes(machine_env, monkeypatch):
    import nof1_causal_lab.utils.openrouter_client as openrouter_client

    workspace_id = machine_env
    ingest = openrouter_client.call_model

    async def scenario():
        from temporalio.testing import WorkflowEnvironment

        from nof1_causal_lab import study_api
        from nof1_causal_lab.actions.results import RunningAction
        from nof1_causal_lab.actions.temporal.client import (
            RUNNING_ACTION_MEMO,
            pydantic_data_converter,
        )
        from nof1_causal_lab.actions.temporal.worker import (
            build_model_checks_worker,
            build_openrouter_worker,
            build_worker,
        )

        # Ingestion waits here, so data preparation stays in flight until released.
        release = asyncio.Event()

        async def held_call_model(*args, **kwargs):
            await release.wait()
            return await ingest(*args, **kwargs)

        monkeypatch.setattr(openrouter_client, "call_model", held_call_model)
        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                create_openrouter_client() as transport,
                build_worker(env.client),
                build_model_checks_worker(env.client),
                build_openrouter_worker(env.client, transport),
                asyncio.timeout(30),
            ):
                monkeypatch.setenv("READ_ONLY_FACADE", "0")
                clients = study_api.TemporalClientProvider()
                monkeypatch.setattr(clients, "get", AsyncMock(return_value=env.client))
                handle = await study_api._study_handle(workspace_id, clients)
                created = await study_api._dispatch_action(
                    workspace_id,
                    SetQuestionRequest(question=QuestionSpec(text=_QUESTION)),
                    clients,
                )
                if isinstance(created, RunningPoll):
                    await handle.get_update_handle(str(created.attempt_id)).result()
                assert (await study_api.get_timeline(workspace_id, clients)).running is None

                preparing = await study_api._dispatch_action(
                    workspace_id, PrepareDataRequest(input=_PREPARATION), clients
                )
                while (running := await study_api._running_action(workspace_id, clients)) is None:
                    await asyncio.sleep(0.05)
                assert isinstance(preparing, RunningPoll)
                assert running.attempt_id == preparing.attempt_id
                assert running.action == "prepare_data"
                assert [message.label for message in running.messages] == ["PREPARE_DATA_STARTED"]

                # A terminated workflow can never finish the attempt its memo still names.
                await handle.terminate()
                memo = await (await handle.describe()).memo_value(
                    RUNNING_ACTION_MEMO, type_hint=RunningAction
                )
                assert memo == running
                assert (await study_api.get_timeline(workspace_id, clients)).running is None
        finally:
            release.set()
            await env.shutdown()

    asyncio.run(scenario())
