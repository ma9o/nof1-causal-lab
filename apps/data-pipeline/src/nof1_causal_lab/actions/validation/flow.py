"""Empirical data profiles and model-dependent measurement compatibility checks."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from nof1_causal_lab.artifacts.data_preparation import PreparedDataMetadata
from nof1_causal_lab.artifacts.identity import IndicatorId

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.construct import ConstructSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.observations import (
        AuthoredObservationSpec,
        ResolvedObservationSpec,
    )

from nof1_causal_lab.actions.validation.rules import (
    COMPATIBILITY_RULES,
    DATA_RULES,
    ValidationContext,
    build_indicator_audits,
    compute_empirical_profile,
    no_data_validation_result,
    run_rules,
)
from nof1_causal_lab.artifacts.validation_report import DataProfileArtifact, ValidationIssue

# ══════════════════════════════════════════════════════════════════════════════
# Task
# ══════════════════════════════════════════════════════════════════════════════


def validate_extraction(
    model: ModelSpec,
    dataframes: list[pl.DataFrame],
) -> DataProfileArtifact:
    """Assess compatibility with the selected model; empirical profiles belong to preparation."""
    dataframes = [df for df in dataframes if not df.is_empty()]
    if not dataframes:
        return no_data_validation_result()

    combined = pl.concat(dataframes, how="vertical")

    if combined.is_empty():
        return no_data_validation_result()

    indicators = tuple(item.observation for item in model.indicators)
    indicator_ids: set[IndicatorId] = {ind.id for ind in indicators}
    indicator_lookup: dict[IndicatorId, AuthoredObservationSpec | ResolvedObservationSpec] = {
        ind.id: ind for ind in indicators
    }
    combined = combined.filter(pl.col("indicator_id").is_in(list(indicator_ids)))
    if combined.is_empty():
        return no_data_validation_result()

    construct_lookup: dict[str, ConstructSpec] = {
        indicator.observation.id: construct for construct, indicator in model.iter_indicators()
    }

    model_clock = model.measurement_clock
    model_clock_hours: float | None = None
    if model_clock is not None:
        model_clock_hours = model_clock.seconds / 3600

    validation_ctx = ValidationContext(
        combined=combined,
        indicators=indicators,
        indicator_ids=indicator_ids,
        indicator_lookup=indicator_lookup,
        construct_lookup=construct_lookup,
        model_clock_hours=model_clock_hours,
    )

    indicator_issues, indicator_health, dataset_issues = run_rules(
        validation_ctx,
        dataset_rules=COMPATIBILITY_RULES,
    )

    indicator_audits = build_indicator_audits(
        indicator_ids=indicator_ids,
        profiles={},
        indicator_issues=indicator_issues,
        indicator_health=indicator_health,
    )

    return DataProfileArtifact(indicators=indicator_audits, dataset_issues=tuple(dataset_issues))


def profile_data(
    data: pl.DataFrame,
    *,
    definitions: tuple[ResolvedObservationSpec, ...] = (),
    metadata: PreparedDataMetadata | None = None,
) -> DataProfileArtifact:
    """Measure observed data without consulting any model or authoring state."""
    if data.is_empty():
        return no_data_validation_result()
    definitions = metadata.variables if metadata is not None else definitions
    lookup: dict[IndicatorId, ResolvedObservationSpec] = {item.id: item for item in definitions}
    identities = {IndicatorId(value) for value in data["indicator_id"].unique()} | set(lookup)
    context = ValidationContext(data, definitions, identities, lookup, {}, None)
    issues, health, dataset_issues = run_rules(context, indicator_rules=DATA_RULES)
    audits = build_indicator_audits(
        indicator_ids=identities,
        profiles={identity: compute_empirical_profile(identity, data) for identity in identities},
        indicator_issues=issues,
        indicator_health=health,
    )
    if definitions:
        unknown = set(data["indicator_id"].unique()) - set(lookup)
        if unknown:
            dataset_issues.append(
                ValidationIssue(
                    indicator_id=None,
                    issue_type="undeclared_variables",
                    severity="error",
                    message=f"{len(unknown)} observed variables have no definitions in the prepared-data metadata.",
                )
            )
        for variable in definitions:
            levels = (
                variable.ordinal_levels
                if variable.measurement_dtype == "ordinal"
                else variable.categorical_levels
            )
            values = data.filter(pl.col("indicator_id") == variable.id)["value"].drop_nulls()
            if levels and any(
                value < 0 or value >= len(levels) or value != int(value)
                for value in values
                if value is not None and math.isfinite(value)
            ):
                audits[variable.id] = audits[variable.id].with_issue(
                    ValidationIssue(
                        indicator_id=variable.id,
                        issue_type="codebook_bounds",
                        severity="error",
                        message="Observed codes must index the declared codebook",
                    )
                )
            if not values.is_finite().all():
                audits[variable.id] = audits[variable.id].with_issue(
                    ValidationIssue(
                        indicator_id=variable.id,
                        issue_type="nonfinite_values",
                        severity="error",
                        message="Observed numeric values must be finite",
                    )
                )
    if isinstance(metadata, PreparedDataMetadata):
        from nof1_causal_lab.artifacts.data_preparation import check_semantic_collisions

        for variable in metadata.preparation.variables:
            for message in check_semantic_collisions(
                variable.extraction.how_to_measure, variable.observation.aggregation
            ):
                audits[variable.observation.id] = audits[variable.observation.id].with_issue(
                    ValidationIssue(
                        indicator_id=variable.observation.id,
                        issue_type="scoring_semantics",
                        severity="warning",
                        message=message,
                    )
                )
    return DataProfileArtifact(indicators=audits, dataset_issues=tuple(dataset_issues))
