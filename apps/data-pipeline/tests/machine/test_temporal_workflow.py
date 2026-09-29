"""Durable four-action dispatch, atomic outcomes, and recovery from Git.

Only ingestion's LLM is stubbed. Model edits supply their definition directly;
fit and simulation readiness failures exercise the real execution boundary.
"""

import asyncio
import dataclasses
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from nof1_causal_lab.actions.contracts import (
    EditModelRequest,
    FitRequest,
    PrepareDataRequest,
    SimulateRequest,
)
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.history import StudyRepository
from nof1_causal_lab.machine.status import ActionOutcome
from nof1_causal_lab.machine.store import ArtifactStore, read_model
from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow
from tests.git_fixtures import git_oid
from tests.helpers import graph_constructs

pytestmark = [pytest.mark.workflow, pytest.mark.timeout(60, method="thread")]
_QUESTION = "does exercise improve sleep?"
_PREPARATION: dict[str, Any] = {
    "source": {"files": ["observations.csv"]},
    "definition": {
        "default_window": "1d",
        "variables": [
            {
                "id": "indicator:sleep",
                "name": "sleep_steps_proxy",
                "measurement_dtype": "continuous",
                "aggregation": "mean",
                "how_to_measure": "Read the steps column",
                "source_columns": ["steps"],
                "extraction_mode": "computed",
            }
        ],
    },
}


def _proposed_model() -> dict[str, Any]:
    return {
        "question": _QUESTION,
        "default_outcome": "construct:sleep",
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
            "id": "indicator:sleep",
            "name": "sleep_steps_proxy",
            "construct_polarity": "positive",
            "measurement_dtype": "continuous",
            "aggregation": "mean",
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

    async def fake_call_model(model_name, messages, tools=None, config=None, log_label=None):
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


def test_episode_workflow_journey(machine_env, monkeypatch):
    workspace_id = machine_env

    async def scenario():
        from temporalio.testing import WorkflowEnvironment

        from nof1_causal_lab import episode_api
        from nof1_causal_lab.machine.temporal.client import pydantic_data_converter
        from nof1_causal_lab.machine.temporal.worker import (
            build_model_checks_worker,
            build_openrouter_worker,
            build_worker,
        )

        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                build_worker(env.client),
                build_model_checks_worker(env.client),
                build_openrouter_worker(env.client),
                asyncio.timeout(30),
            ):
                monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "0")
                monkeypatch.setattr(episode_api, "_client", env.client)
                handle = await episode_api._episode_handle(workspace_id)

                async def execute(request):
                    receipt = await episode_api.execute_scientific_action(workspace_id, request)
                    assert set(receipt.model_dump()) == {"attempt_id"}
                    outcome = await handle.get_update_handle(
                        str(receipt.attempt_id), result_type=ActionOutcome
                    ).result()
                    polled = await episode_api.poll_scientific_action(
                        workspace_id, receipt.attempt_id
                    )
                    assert polled.done
                    assert (polled.body is not None) == (outcome.status == "applied")
                    assert polled.messages[0].label == f"{request.action.upper()}_STARTED"
                    assert polled.messages[-1].level == (
                        "info" if outcome.status == "applied" else "error"
                    )
                    persisted = StudyRepository(workspace_id).dispatched_attempt(receipt.attempt_id)
                    assert persisted is not None
                    assert persisted.commit_id == outcome.commit_id
                    return outcome

                rejected = await execute(
                    EditModelRequest(
                        expected_revision=git_oid(42),
                        model=ModelSpec(question=_QUESTION),
                    )
                )
                assert rejected.status == "rejected"
                assert rejected.reason

                initial = await execute(
                    EditModelRequest(
                        expected_revision=None,
                        model=ModelSpec(question=_QUESTION),
                    )
                )
                assert initial.status == "applied"
                prepared = await execute(PrepareDataRequest(input=_PREPARATION))
                assert prepared.status == "applied", prepared
                assert prepared.state.has("panel")
                assert prepared.state.has("data_profile")
                edited = await execute(
                    EditModelRequest(
                        expected_revision=initial.state.current["model"].revision,
                        model=ModelSpec.model_validate(_measured_model()),
                    )
                )
                assert edited.status == "applied", edited
                model_revision = edited.state.current["model"].revision
                before = (await handle.query(EpisodeWorkflow.get_state)).current

                # Both numerical actions reject incomplete definitions before any numerical work.
                for request in (
                    FitRequest(
                        model_revision=model_revision, panel_revision=before["panel"].revision
                    ),
                    SimulateRequest(model_revision=model_revision, start=0, end=1),
                ):
                    raised = await execute(request)
                    assert raised.status == "raised", raised
                    assert raised.error_type == "IncompleteModelError", raised.error_message
                    assert (await handle.query(EpisodeWorkflow.get_state)).current == before

                status = await handle.query(EpisodeWorkflow.get_status)
                assert set(status.actions) == {"edit_model", "prepare_data", "fit", "simulate"}
                assert not any(artifact.stale for artifact in status.artifacts)

                # The facade must recover both committed science and the latest
                # attempt sequence, including failed attempts after the branch head.
                previous_run_id = handle.first_execution_run_id
                await handle.terminate()
                handle = await episode_api._episode_handle(workspace_id)
                assert handle.first_execution_run_id != previous_run_id
                recovered = await handle.query(EpisodeWorkflow.get_status)
                assert recovered.state == status.state
                assert recovered.seq == status.seq

                store = ArtifactStore(workspace_id)
                revised = read_model(store, model_revision).revised(
                    question="does caffeine harm sleep?"
                )
                rewritten = await execute(
                    EditModelRequest(expected_revision=model_revision, model=revised)
                )
                assert rewritten.status == "applied", rewritten
                assert rewritten.seq == recovered.seq + 1
                status = await handle.query(EpisodeWorkflow.get_status)
                stale = {a.artifact_id for a in status.artifacts if a.stale}
                assert "panel" not in stale
                assert "model" not in stale
                assert "raw_data" not in stale

                records = StudyRepository(workspace_id).attempts()
                assert [record.status for record in records] == [
                    "rejected",
                    "applied",
                    "applied",
                    "applied",
                    "raised",
                    "raised",
                    "applied",
                ]
                assert [record.action for record in records] == [
                    "edit_model",
                    "edit_model",
                    "prepare_data",
                    "edit_model",
                    "fit",
                    "simulate",
                    "edit_model",
                ]
                assert records[3].operation_id is None
                assert ModelSpec.model_validate(
                    records[3].inputs["model"]
                ) == ModelSpec.model_validate(_measured_model())
                assert records[4].inputs["model_revision"] == model_revision
                assert all("move" not in record.model_dump() for record in records)
                assert (
                    read_model(store, initial.state.current["model"].revision).question == _QUESTION
                )
                assert (
                    read_model(store, rewritten.state.current["model"].revision).question
                    == revised.question
                )
                await handle.signal(EpisodeWorkflow.close)
                await handle.result()
        finally:
            await env.shutdown()

    asyncio.run(scenario())


def test_status_reports_only_attempts_a_live_workflow_executes(machine_env, monkeypatch):
    import nof1_causal_lab.utils.openrouter_client as openrouter_client

    workspace_id = machine_env
    ingest = openrouter_client.call_model

    async def scenario():
        from temporalio.testing import WorkflowEnvironment

        from nof1_causal_lab import episode_api
        from nof1_causal_lab.actions.results import RunningAction
        from nof1_causal_lab.machine.temporal.client import (
            RUNNING_ACTION_MEMO,
            pydantic_data_converter,
        )
        from nof1_causal_lab.machine.temporal.worker import (
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
                build_worker(env.client),
                build_model_checks_worker(env.client),
                build_openrouter_worker(env.client),
                asyncio.timeout(30),
            ):
                monkeypatch.setenv("EPISODE_FACADE_READ_ONLY", "0")
                monkeypatch.setattr(episode_api, "_client", env.client)
                handle = await episode_api._episode_handle(workspace_id)
                created = await episode_api.execute_scientific_action(
                    workspace_id,
                    EditModelRequest(expected_revision=None, model=ModelSpec(question=_QUESTION)),
                )
                await handle.get_update_handle(
                    str(created.attempt_id), result_type=ActionOutcome
                ).result()
                assert (await episode_api.get_episode(workspace_id))["running"] is None

                preparing = await episode_api.execute_scientific_action(
                    workspace_id, PrepareDataRequest(input=_PREPARATION)
                )
                while (running := await episode_api._running_action(workspace_id)) is None:
                    await asyncio.sleep(0.05)
                assert running.attempt_id == preparing.attempt_id
                assert (running.action, running.branch) == ("prepare_data", "main")
                assert [message.label for message in running.messages] == ["PREPARE_DATA_STARTED"]

                # A terminated workflow can never finish the attempt its memo still names.
                await handle.terminate()
                memo = await (await handle.describe()).memo_value(
                    RUNNING_ACTION_MEMO, type_hint=RunningAction
                )
                assert memo == running
                assert (await episode_api.get_episode(workspace_id))["running"] is None
        finally:
            release.set()
            await env.shutdown()

    asyncio.run(scenario())
