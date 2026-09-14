"""EpisodeWorkflow end-to-end against a local Temporal dev server.

Covers the durable-shell contract: moves serialize through the propose
update, rejections and typed failures are journaled (not just applied
moves), state only changes on applied transitions, and staleness follows
provenance after an upstream rewrite. Stage runners are stubbed — the
real ones are exercised in their own suites; here we test the machine.
"""

import asyncio
import dataclasses
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.machine.derivations import read_model
from nof1_causal_lab.machine.moves import (
    RunOperation,
    WriteArtifact,
)
from nof1_causal_lab.machine.status import MoveOutcome
from nof1_causal_lab.machine.store import ArtifactStore, EpisodeJournal
from nof1_causal_lab.machine.temporal.messages import EpisodeInit, MoveRequest
from nof1_causal_lab.machine.temporal.model_spec_checkpoints import (
    latest_failed_model_spec_checkpoint_ref,
    read_model_spec_checkpoint,
)
from nof1_causal_lab.machine.temporal.workflow import EpisodeWorkflow
from tests.helpers import complete_test_model, graph_constructs

pytestmark = [pytest.mark.workflow, pytest.mark.timeout(240)]
_QUESTION = "does exercise improve sleep?"


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
            "how_to_measure": "Use the steps column as a placeholder sleep proxy.",
            "construct_polarity": "positive",
            "measurement_dtype": "continuous",
            "aggregation": "mean",
            "source_columns": ["steps"],
            "extraction_mode": "computed",
        }
    ]
    return model


def _statistical_submission() -> dict[str, Any]:
    model = complete_test_model(ModelSpec.model_validate(_measured_model())).model_dump(mode="json")
    return {
        "construct": graph_constructs(model)[0],
        "edges": model["edges"],
        "parameters": model["parameters"],
        "distributions": model["distributions"],
    }


@pytest.fixture
def machine_env(monkeypatch, tmp_path):
    import nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow as construct_flow
    import nof1_causal_lab.models.ssm.construct_admission as construct_admission
    import nof1_causal_lab.utils.openrouter_client as openrouter_client
    from nof1_causal_lab.utils import config as config_module
    from nof1_causal_lab.utils import data as data_module
    from nof1_causal_lab.utils.config import LLMProfileConfig

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
                llm=LLMProfileConfig(harness="none", model="openrouter/mock-raw"),
            ),
            structure_proposal=dataclasses.replace(
                config.structure_proposal,
                llm=LLMProfileConfig(harness="none", model="openrouter/mock-latent"),
            ),
            prior_elicitation=dataclasses.replace(
                config.prior_elicitation,
                llm=LLMProfileConfig(harness="none", model="openrouter/mock-model-spec"),
            ),
        ),
    )

    def fake_admit_construct(state, contribution, *_args, **_kwargs):
        admitted = construct_admission.trial_admission_state(state, contribution)
        report = construct_admission.ConstructAdmissionReport(
            name=contribution.name,
            results=(),
            timings=(),
            outcome="ADMITTED",
            annotations=(),
            admitted=True,
        )
        return admitted, report

    def fail_full_admission_validation(*_args, **_kwargs):
        # Exercise the activity's typed error conversion and durable checkpoint.
        raise ValueError("deliberate full-model validation failure")

    monkeypatch.setattr(construct_flow, "admit_construct", fake_admit_construct)
    monkeypatch.setattr(
        construct_admission,
        "validate_full_admission_state",
        fail_full_admission_validation,
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
                                                "'observations.csv')\n"
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
        tool_name = tools[0].name if tools else ""
        if tool_name == "validate_measurement_structure":
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-measurement",
                            "type": "function",
                            "function": {
                                "name": "validate_measurement_structure",
                                "arguments": json.dumps(
                                    {"model_json": json.dumps(_measured_model())}
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
        if "submit_construct" in tool_names:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-submit-construct",
                            "type": "function",
                            "function": {
                                "name": "submit_construct",
                                "arguments": json.dumps(_statistical_submission()),
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
                        "id": "call-latent",
                        "type": "function",
                        "function": {
                            "name": "validate_latent_structure",
                            "arguments": json.dumps({"model_json": json.dumps(_proposed_model())}),
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

    monkeypatch.setattr(openrouter_client, "call_model", fake_call_model)
    return workspace_id


def test_episode_workflow_journey(machine_env):
    workspace_id = machine_env

    async def scenario():
        from temporalio.testing import WorkflowEnvironment

        from nof1_causal_lab.machine.temporal.client import pydantic_data_converter
        from nof1_causal_lab.machine.temporal.worker import (
            build_model_spec_simulation_worker,
            build_openrouter_worker,
            build_worker,
        )

        env = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter)
        try:
            async with (
                build_worker(env.client, task_queue="test-episodes"),
                build_openrouter_worker(env.client),
                build_model_spec_simulation_worker(env.client),
            ):
                handle = await env.client.start_workflow(
                    EpisodeWorkflow.run,
                    EpisodeInit(workspace_id=workspace_id),
                    id=f"episode-{workspace_id}",
                    task_queue="test-episodes",
                )

                async def propose(move, **kwargs):
                    return await handle.execute_update(
                        "propose",
                        MoveRequest(move=move, **kwargs),
                        result_type=MoveOutcome,
                    )

                # 1. Illegal move first: rejected AND journaled.
                rejected = await propose(RunOperation(operation_id="measurement_structure"))
                assert rejected.status == "rejected"
                assert "model" in rejected.reason

                # 2. Root write enables downstream.
                applied = await propose(
                    WriteArtifact(artifact_id="model", expected_model_version=0),
                    payload={"question": _QUESTION},
                )
                assert applied.status == "applied"
                assert applied.state.has("model")

                # 3. Free navigation through enabled transitions (stubs).
                for artifact_id in (
                    "raw_data",
                    "latent_structure",
                    "measurement_structure",
                    "measurements",
                ):
                    outcome = await propose(RunOperation(operation_id=artifact_id))
                    assert outcome.status == "applied", (artifact_id, outcome)

                # 4. Typed stage failure: raised, state unchanged.
                before = (await handle.query(EpisodeWorkflow.get_state)).current
                raised = await propose(RunOperation(operation_id="statistical_model_spec"))
                assert raised.status == "raised"
                assert raised.error_type == "ModelCompileError", raised.error_message
                assert "deliberate full-model validation failure" in raised.error_message
                assert "report" in raised.diagnostics
                after = (await handle.query(EpisodeWorkflow.get_state)).current
                assert after == before

                # 5. A question revision invalidates extraction, preserving raw data.
                status = await handle.query(EpisodeWorkflow.get_status)
                stale_before = {a.artifact_id for a in status.artifacts if a.stale}
                assert stale_before == set()
                store = ArtifactStore(workspace_id)
                model_version = before["model"].version
                revised = read_model(store, model_version).model_dump(mode="json")
                revised["question"] = "does caffeine harm sleep?"
                rewritten = await propose(
                    WriteArtifact(artifact_id="model", expected_model_version=model_version),
                    payload=revised,
                )
                assert rewritten.status == "applied", rewritten
                status = await handle.query(EpisodeWorkflow.get_status)
                stale = {a.artifact_id for a in status.artifacts if a.stale}
                assert "panel" in stale
                assert "model" not in stale
                assert "raw_data" not in stale

                # 6. Journal recorded every attempt, including the rejection
                #    and the typed failure.
                records = EpisodeJournal(workspace_id).read_all()
                statuses = [record.status for record in records]
                assert statuses == [
                    "rejected",
                    "applied",  # initial model question
                    "applied",  # ingestion
                    "applied",  # latent-structure
                    "applied",  # measurement-structure
                    "applied",  # extraction
                    "raised",  # model-spec
                    "applied",  # model question revision
                ]
                assert records[-2].error_type == "ModelCompileError"
                assert "report" in records[-2].diagnostics
                assert records[-2].resume is not None
                assert records[-2].resume.kind == "model_spec"
                checkpoint_ref = latest_failed_model_spec_checkpoint_ref(workspace_id)
                assert checkpoint_ref is not None
                read_model_spec_checkpoint(workspace_id, checkpoint_ref)

                # Model history keeps the original question and each subsequent revision.
                assert store.list_versions("model") == list(range(1, model_version + 2))
                assert read_model(store, 1).question == _QUESTION
                assert read_model(store, model_version + 1).question == revised["question"]
        finally:
            await env.shutdown()

    asyncio.run(scenario())
