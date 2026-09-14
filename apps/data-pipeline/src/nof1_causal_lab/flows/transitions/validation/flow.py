"""validation validation entrypoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec

from nof1_causal_lab.flows.transitions.validation.rules import (
    RULES,
    ValidationContext,
    build_indicator_audits,
    derive_validation_status,
    no_data_validation_result,
    run_rules,
)
from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001

# ══════════════════════════════════════════════════════════════════════════════
# Task
# ══════════════════════════════════════════════════════════════════════════════


def validate_extraction(
    model: ModelSpec,
    dataframes: list[pl.DataFrame],
) -> UncheckedJsonObject:
    """Validate semantic properties of extracted data.

    Runs all ``RULES`` against the extracted data and reduces findings
    into a keyed indicator audit map plus dataset-level issues.

    Args:
        causal_design: The full causal design with measurement structure
        dataframes: List of DataFrames with columns (indicator, value, anchor_time)

    Returns:
        Dict with:
            - is_valid: bool
            - indicators: per-indicator profile + validation
            - dataset_issues: cross-indicator validation findings
    """
    dataframes = [df for df in dataframes if df is not None and not df.is_empty()]
    if not dataframes:
        return no_data_validation_result()

    combined = pl.concat(dataframes, how="vertical")

    if combined.is_empty():
        return no_data_validation_result()

    from nof1_causal_lab.models.model_inputs import identification_input

    inputs = identification_input(model)
    indicators = inputs["observations"]["indicators"]
    indicator_ids: set[str] = {ind["id"] for ind in indicators}
    indicator_lookup = {ind["id"]: ind for ind in indicators}
    unknown = set(combined["indicator_id"].unique()) - indicator_ids
    if unknown:
        raise ValueError(f"Observations reference indicators outside the pinned design: {unknown}")

    constructs = inputs["graph"]["constructs"]
    construct_lookup = {c["id"]: c for c in constructs}

    model_clock_str = inputs["observations"]["model_clock"]
    model_clock_hours: float | None = None
    if model_clock_str:
        import contextlib

        from nof1_causal_lab.artifacts.duration import parse_duration_to_hours

        with contextlib.suppress(ValueError):
            model_clock_hours = parse_duration_to_hours(model_clock_str)

    validation_ctx = ValidationContext(
        combined=combined,
        indicators=indicators,
        indicator_ids=indicator_ids,
        indicator_lookup=indicator_lookup,
        construct_lookup=construct_lookup,
        model_clock_hours=model_clock_hours,
    )

    indicator_issues, indicator_health, dataset_issues = run_rules(
        RULES,
        validation_ctx,
    )

    indicator_audits = build_indicator_audits(
        indicator_ids=indicator_ids,
        indicator_lookup=indicator_lookup,
        model_data=combined,
        indicator_issues=indicator_issues,
        indicator_health=indicator_health,
    )

    all_issues = [*indicator_issues, *dataset_issues]
    status = derive_validation_status(all_issues)

    return {
        "is_valid": status["is_valid"],
        "indicators": indicator_audits,
        "dataset_issues": dataset_issues,
    }
