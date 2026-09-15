"""validation validation entrypoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec

from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact
from nof1_causal_lab.flows.transitions.validation.rules import (
    COMPATIBILITY_RULES,
    DATA_RULES,
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
    *,
    data_profile: DataProfileArtifact | None = None,
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
        COMPATIBILITY_RULES,
        validation_ctx,
    )

    indicator_audits = build_indicator_audits(
        indicator_ids=indicator_ids,
        indicator_lookup=indicator_lookup,
        model_data=combined,
        indicator_issues=indicator_issues,
        indicator_health=indicator_health,
    )

    if unknown:
        dataset_issues.append(
            {
                "indicator_id": None,
                "issue_type": "unknown_indicators",
                "severity": "error",
                "message": f"Observations reference indicators outside the selected model: {sorted(unknown)}",
            }
        )
    profile = (data_profile if data_profile is not None else profile_data(combined)).model_dump(
        mode="json"
    )
    # Reuse stored empirical summaries; add only model-dependent measurements here.
    for identity, audit in indicator_audits.items():
        source = profile["indicators"].get(identity)
        if source is not None:
            audit["issues"] = [*source["issues"], *audit["issues"]]
            audit["checks"] = {**source["checks"], **audit["checks"]}
            if audit["profile"] is not None and source["profile"] is not None:
                semantic = {
                    key: audit["profile"][key]
                    for key in (
                        "measurement_dtype",
                        "time_coverage_ratio",
                        "max_gap_ratio",
                        "dtype_violations",
                        "duplicate_pct",
                        "arithmetic_sequence_detected",
                    )
                }
                audit["profile"] = {**source["profile"], **semantic}
        else:
            audit["checks"]["data_availability"] = "not_evaluated"
    all_issues = [
        issue for audit in indicator_audits.values() for issue in audit["issues"]
    ] + dataset_issues
    status = derive_validation_status(all_issues)

    return {
        "is_valid": status["is_valid"],
        "indicators": indicator_audits,
        "dataset_issues": dataset_issues,
    }


def profile_data(data: pl.DataFrame) -> DataProfileArtifact:
    """Measure observed data without consulting any model or authoring state."""
    if data.is_empty():
        return DataProfileArtifact.model_validate(no_data_validation_result())
    identities = set(data["indicator_id"].unique())
    context = ValidationContext(data, [], identities, {}, {}, None)
    issues, health, dataset_issues = run_rules(DATA_RULES, context)
    audits = build_indicator_audits(
        indicator_ids=identities,
        indicator_lookup={},
        model_data=data,
        indicator_issues=issues,
        indicator_health=health,
    )
    return DataProfileArtifact.model_validate(
        {
            "is_valid": derive_validation_status(issues + dataset_issues)["is_valid"],
            "indicators": audits,
            "dataset_issues": dataset_issues,
        }
    )
