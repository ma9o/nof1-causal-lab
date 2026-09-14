"""Temporal activities for the measurement-structure transition."""

from __future__ import annotations

import json
from typing import Any

from temporalio import activity

from nof1_causal_lab.machine.artifact_files import parquet_filename
from nof1_causal_lab.machine.derivations import read_model
from nof1_causal_lab.machine.graph import transition_spec
from nof1_causal_lab.machine.moves import TransitionEffects, input_pins
from nof1_causal_lab.machine.store import ArtifactStore
from nof1_causal_lab.machine.temporal.activity_errors import (
    as_non_retryable_application_error,
)
from nof1_causal_lab.machine.temporal.latent_structure_activities import (
    _llm_backend_config,
)
from nof1_causal_lab.machine.temporal.llm_subroutine_storage import subroutine_root
from nof1_causal_lab.machine.temporal.messages import (
    SingleLLMTransitionFinalizeInput,
    SingleLLMTransitionPlan,
    SingleLLMTransitionWorkflowInput,
)
from nof1_causal_lab.utils import storage


def _write_measurement_structure_json(path: str, value: Any) -> None:
    storage.write_text(path, json.dumps(value))


@activity.defn
async def plan_measurement_structure_activity(
    input: SingleLLMTransitionWorkflowInput,
) -> SingleLLMTransitionPlan:
    from nof1_causal_lab.flows.pipeline_helpers import format_schema_for_llm
    from nof1_causal_lab.flows.transitions.measurement_structure.prompting import (
        build_measurement_structure_user_prompt,
        templates,
    )
    from nof1_causal_lab.utils.config import get_config

    store = ArtifactStore(input.workspace_id)
    spec = transition_spec("measurement_structure")
    pins = input_pins(input.state, spec)
    run_id = f"seq-{input.seq:06d}"

    model = read_model(store, pins["model"])
    question = model.require_question()
    raw_table = store.read_parquet_table(
        "raw_data",
        pins["raw_data"],
        parquet_filename("raw_data", "raw"),
    )
    model_payload = model.model_dump(mode="json")
    dataset_schema = format_schema_for_llm(raw_table)
    dataset_summary = f"{raw_table.num_rows} rows x {raw_table.num_columns} columns"
    context_ref = storage.join(
        subroutine_root(input.workspace_id, run_id, "measurement-structure"),
        "context.json",
    )
    _write_measurement_structure_json(
        context_ref,
        {
            "system_prompt": templates.SYSTEM,
            "user_messages": [
                build_measurement_structure_user_prompt(
                    question,
                    model_payload,
                    [dataset_schema],
                    dataset_summary,
                ),
                templates.REVIEW,
            ],
            "model": model_payload,
        },
    )

    config = get_config()
    max_tool_turns = config.structure_proposal.measurement_max_tool_turns
    return SingleLLMTransitionPlan(
        workspace_id=input.workspace_id,
        run_id=run_id,
        context_ref=context_ref,
        pins=pins,
        llm=_llm_backend_config(config.structure_proposal.llm, config.llm, max_tool_turns),
        max_tool_turns=max_tool_turns,
    )


@activity.defn
async def finalize_measurement_structure_activity(
    input: SingleLLMTransitionFinalizeInput,
) -> TransitionEffects:
    from nof1_causal_lab.machine.temporal.model_authoring import finalize_model_revision

    try:
        return finalize_model_revision(input, "measurement_structure")
    except Exception as exc:
        raise as_non_retryable_application_error(exc) from exc


MEASUREMENT_STRUCTURE_ACTIVITIES = [
    plan_measurement_structure_activity,
    finalize_measurement_structure_activity,
]
