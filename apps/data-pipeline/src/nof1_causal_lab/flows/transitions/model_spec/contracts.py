"""model-spec contracts and tool metadata."""

from __future__ import annotations

from nof1_causal_lab.flows.transitions.model_spec.tool_registry import (
    build_model_spec_public_tool_contracts,
)

MODEL_SPEC_TOOL_CONTRACTS = build_model_spec_public_tool_contracts()
