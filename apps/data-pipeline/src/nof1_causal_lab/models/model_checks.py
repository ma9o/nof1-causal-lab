"""Check execution requirements directly on the current scientific model."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def collect_structure_compile_errors(
    model: ModelSpec,
    *,
    manifest_names: Sequence[str] | None = None,
) -> list[str]:
    """Validate that the retained executable structure can be compiled.

    Under the current compiler/runtime, every retained state must be
    supported by at least one manifest channel, and the loading matrix must be
    able to reach full column rank.
    """
    from nof1_causal_lab.utils.model_structure import get_state_names

    errors: list[str] = []
    try:
        latent_states = get_state_names(model)
    except ValueError as exc:
        return [str(exc)]
    if not latent_states:
        return ["model.state_order is empty"]

    indicator_lookup = {
        model.indicator(identity).name: model.indicator(identity)
        for identity in model.manifest_indicator_order
    }
    used_manifests = (
        list(manifest_names) if manifest_names is not None else list(indicator_lookup.keys())
    )
    covered_state_counts = Counter(
        model.indicator_owner(indicator.id).name
        for manifest_name in used_manifests
        if (indicator := indicator_lookup.get(manifest_name)) is not None
    )
    uncovered_states = sorted(
        state for state in latent_states if covered_state_counts.get(state, 0) == 0
    )
    if uncovered_states:
        errors.append(
            "Retained states have no measurement indicators: "
            f"{uncovered_states}. Add proxy indicators for these constructs or "
            "exclude them from the executable state selection."
        )

    n_manifest = len(used_manifests)
    n_latent = len(latent_states)
    if n_manifest < n_latent:
        errors.append(
            f"Loading matrix is rank-deficient: n_manifest ({n_manifest}) < n_latent ({n_latent})."
        )

    return errors


def check_execution(
    model: ModelSpec,
) -> tuple[AnchorCertificate, ...]:
    """Require a complete, executable model and return its latent anchoring evidence."""
    model.require_execution_structure()
    errors = collect_measurement_compile_errors(model)
    errors.extend(collect_structure_compile_errors(model))
    if errors:
        raise ValueError("ModelSpec failed compiler validation:\n" + "\n".join(errors))

    from nof1_causal_lab.models.ssm import numerics as numeric
    from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
    from nof1_causal_lab.models.ssm.compile.structural import (
        compile_anchor_certificates,
    )

    model.require_priors()
    numeric.validate_execution(model)
    parameter_bindings(model)
    return tuple(compile_anchor_certificates(model))
