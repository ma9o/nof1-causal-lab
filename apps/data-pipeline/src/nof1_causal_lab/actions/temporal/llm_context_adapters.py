"""Context adapters for generic Temporal LLM subroutines."""

from __future__ import annotations

from typing import assert_never

from nof1_causal_lab.actions.temporal.llm_subroutine_storage import read_subroutine_json
from nof1_causal_lab.actions.temporal.messages import (
    LLMSubroutineContextKind,
    LLMToolSpec,
    MeasurementChunkContext,
)


def subroutine_context_messages(
    context_kind: LLMSubroutineContextKind,
    context_ref: str,
) -> tuple[str | None, list[str], list[LLMToolSpec]]:
    if context_kind == "measurement_extraction":
        from nof1_causal_lab.workers.messages import WorkerMessages

        spec = read_subroutine_json(context_ref, MeasurementChunkContext)
        messages = WorkerMessages(
            question=spec.question,
            measurement_structure=spec.measurement_structure,
            window_text=spec.window_text,
            n_windows=len(spec.window_starts),
        ).extraction_messages()
        system_prompt = None
        user_messages: list[str] = []
        for message in messages:
            if message["role"] == "system":
                system_prompt = message["content"]
            elif message["role"] == "user":
                user_messages.append(message["content"])
        return (
            system_prompt,
            user_messages,
            [
                LLMToolSpec(
                    name="validate_extractions",
                    description="Validate worker extraction output JSON.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "output_json": {
                                "type": "string",
                                "description": "The JSON string containing the worker output.",
                            }
                        },
                        "required": ["output_json"],
                        "additionalProperties": False,
                    },
                )
            ],
        )

    if context_kind == "raw_data_ingestion":
        from nof1_causal_lab.actions.ingestion.flow import SYSTEM_PROMPT, USER_PROMPT

        return (
            SYSTEM_PROMPT,
            [USER_PROMPT],
            [
                LLMToolSpec(
                    name="list_files",
                    description="List files in the prepared input directory.",
                    kind="read_only",
                    executor="raw_data_list_files",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path within the input directory.",
                                "default": ".",
                            }
                        },
                        "required": [],
                        "additionalProperties": False,
                    },
                ),
                LLMToolSpec(
                    name="read_file_sample",
                    description="Read a sample of lines from a file to understand its format.",
                    kind="read_only",
                    executor="raw_data_read_file_sample",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative path to the file within the input directory.",
                            },
                            "n_lines": {
                                "type": "integer",
                                "description": "Number of lines to read.",
                                "default": 50,
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                ),
                LLMToolSpec(
                    name="execute_python",
                    description="Execute Python code in the local pipeline process to parse files into a Polars DataFrame.",
                    kind="checkpoint",
                    executor="raw_data_execute_python",
                    parameters={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "Python code to execute. Assign the final Polars DataFrame "
                                    "to result_df."
                                ),
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                ),
                LLMToolSpec(
                    name="submit_table",
                    description="Validate and finalize the ingested DataFrame with column descriptions.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "column_descriptions_json": {
                                "type": "string",
                                "description": "JSON object mapping column names to descriptions.",
                            }
                        },
                        "required": ["column_descriptions_json"],
                        "additionalProperties": False,
                    },
                    kind="terminal",
                    executor="raw_data_submit_table",
                ),
            ],
        )

    assert_never(context_kind)
