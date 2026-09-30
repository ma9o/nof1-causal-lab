"""Check execution requirements directly on the current scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.execution import AnchorCertificate
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def collect_measurement_compile_errors(
    model: ModelSpec,
) -> list[str]:
    """Collect deterministic measurement checks best handled at compile time."""
    errors: list[str] = []

    if not model.indicators:
        errors.append("Measurement structure must include at least one indicator.")
        return errors

    outcomes = [
        construct
        for construct in model.constructs
        if model.default_outcome is not None and construct.id == model.default_outcome
    ]
    for outcome in outcomes:
        if not outcome.indicators:
            errors.append(f"Outcome construct '{outcome.name}' must have at least one indicator.")

    return errors


def check_execution(
    model: ModelSpec,
) -> tuple[AnchorCertificate, ...]:
    """Require a complete, executable model and return its latent anchoring evidence."""
    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings

    model.require_priors()
    certificates = numeric.validate_execution(model)
    parameter_bindings(model)
    return certificates
