"""Declarative action/context hierarchy semantics."""

import pytest

from nof1_causal_lab.flows.context_tools import CONTEXT_TOOLS
from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
)
from nof1_causal_lab.machine.hierarchy import (
    ACTIONS,
    ACTIONS_BY_ID,
    CONTEXTS,
    describe_actions,
    describe_contexts,
)

pytestmark = pytest.mark.contract


def test_context_tree_is_closed():
    context_ids = {context.context_id for context in CONTEXTS}

    assert "navigator" in context_ids
    assert "episode-machine" in context_ids
    for context in CONTEXTS:
        if context.parent_id is not None:
            assert context.parent_id in context_ids


def test_scientific_surface_has_four_actions_independent_of_recipe_stages():
    assert set(ACTIONS_BY_ID) == {"edit_model", "prepare_data", "fit", "simulate"}
    assert ACTIONS_BY_ID["fit"].consumes == ("model", "panel")
    assert ACTIONS_BY_ID["simulate"].consumes == ("model",)
    assert ACTIONS_BY_ID["edit_model"].consumes == ()


def test_every_transition_declares_a_creation_class():
    valid = {"deterministic", "batch_llm", "judgment"}
    for spec in ARTIFACT_GRAPH:
        assert spec.creation_class in valid


def test_public_context_tools_are_allowed_by_their_context():
    tool_names_by_context = {
        context_id: {tool.name for tool in tools} for context_id, tools in CONTEXT_TOOLS.items()
    }

    for context in CONTEXTS:
        if not context.allowed_tools:
            continue
        declared = tool_names_by_context[context.context_id]
        assert declared.issubset(context.allowed_tools)


def test_registry_descriptions_are_json_ready():
    action_payload = describe_actions()
    context_payload = describe_contexts()

    assert {entry["action_id"] for entry in action_payload} == {
        action.action_id for action in ACTIONS
    }
    assert {entry["context_id"] for entry in context_payload} == {
        context.context_id for context in CONTEXTS
    }
    edit = next(entry for entry in action_payload if entry["action_id"] == "edit_model")
    assert edit["derives"] == (
        "identification_report",
        "validation_report",
    )
    for context in CONTEXTS:
        if context.context_id in {
            "latent-structure",
            "measurement-structure",
            "statistical-model-spec",
        }:
            assert context.layer == "tool"
            assert not context.runtime_state
