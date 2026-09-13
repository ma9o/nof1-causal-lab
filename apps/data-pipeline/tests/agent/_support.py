"""Shared fixtures for agent-session utility tests."""

from __future__ import annotations

import json

from nof1_causal_lab.workers.schemas import validate_worker_output


def _make_validation_tool(
    name,
    description,
    param_name,
    param_description,
    validator,
    capture_key,
):
    from nof1_causal_lab.utils.openrouter_client import Tool

    capture = {}

    async def _execute(**kwargs):
        try:
            data = json.loads(kwargs[param_name])
        except json.JSONDecodeError as error:
            return f"JSON parse error: {error}"

        _result, errors = validator(data)
        if errors:
            return "VALIDATION ERRORS:\n" + "\n".join(f"- {error}" for error in errors)
        capture[capture_key] = data
        return "VALID"

    return (
        Tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {param_name: {"type": "string", "description": param_description}},
                "required": [param_name],
                "additionalProperties": False,
            },
            execute=_execute,
            stop_on_success=True,
            success_output="VALID",
        ),
        capture,
    )


def worker_schema():
    return {
        "model_clock": "1d",
        "indicators": [
            {
                "id": "indicator:9866c549bd1c25f0a5d7",
                "construct_id": "construct:cdc0b2958a9512b2abad",
                "name": "sleep_hours",
                "measurement_dtype": "continuous",
                "aggregation": "last",
                "how_to_measure": "Read sleep hours directly from the rows",
            }
        ],
    }


def make_worker_tool(schema=None):
    if schema is None:
        schema = worker_schema()
    return _make_validation_tool(
        name="validate_extractions",
        description="Validate worker extraction output JSON.",
        param_name="output_json",
        param_description="The JSON string containing the worker output.",
        validator=lambda data: validate_worker_output(data, schema),
        capture_key="output",
    )
