"""Execute model staging and action-owned checks in local contract tests."""

from nof1_causal_lab.actions.edit_model import edit_model
from nof1_causal_lab.actions.model_checks import evaluate_model_checks


def edit_and_check(workspace_id, request, state):
    staged = edit_model(workspace_id, request, state)
    return evaluate_model_checks(workspace_id, state, staged, action="edit_model")
