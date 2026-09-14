from nof1_causal_lab.machine.errors import TransitionExecutionError
from nof1_causal_lab.machine.temporal.activity_errors import (
    as_non_retryable_application_error,
)


def test_application_error_preserves_transition_diagnostics():
    error = TransitionExecutionError(
        "model failed",
        transition_id="posterior",
        diagnostics={"reason": "diverged"},
    )

    converted = as_non_retryable_application_error(error)

    assert converted.message == "model failed"
    assert converted.type == "TransitionExecutionError"
    assert converted.non_retryable is True
    assert converted.details == ({"reason": "diverged"},)


def test_application_error_omits_diagnostics_for_untyped_failures():
    converted = as_non_retryable_application_error(ValueError("invalid input"))

    assert converted.message == "invalid input"
    assert converted.type == "ValueError"
    assert converted.non_retryable is True
    assert converted.details == ()
