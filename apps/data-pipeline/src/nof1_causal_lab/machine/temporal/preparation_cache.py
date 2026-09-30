"""Effective LLM request identities, independent of study and run locations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.machine.temporal.llm_context_adapters import subroutine_context_messages
from nof1_causal_lab.utils.content_cache import cache_path, content_key

if TYPE_CHECKING:
    from nof1_causal_lab.llm_specs import LLMProfileSpec
    from nof1_causal_lab.machine.temporal.messages import LLMSubroutineContextKind

# Bump the corresponding version when prompts, tools, schemas, or validation change.
INGESTION_POLICY_VERSION = "ingestion-v1"
EXTRACTION_POLICY_VERSION = "extraction-v3"


def preparation_cache_path(
    kind: LLMSubroutineContextKind,
    context_ref: str,
    llm: LLMProfileSpec,
    max_tool_turns: int,
    inputs: object,
) -> str:
    system, users, tools = subroutine_context_messages(kind, context_ref)
    policy = {
        "raw_data_ingestion": INGESTION_POLICY_VERSION,
        "measurement_extraction": EXTRACTION_POLICY_VERSION,
    }[kind]
    key = content_key(
        {
            "policy": policy,
            "system": system,
            "users": users,
            "tools": [tool.model_dump(mode="json") for tool in tools],
            "llm": llm.model_dump(mode="json"),
            "max_tool_turns": max_tool_turns,
            "inputs": inputs,
        }
    )
    return cache_path(kind, key)
