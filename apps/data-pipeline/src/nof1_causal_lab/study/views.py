"""Compose display payloads using immutable versions selected by a Git snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nof1_causal_lab.artifacts.checks import ConvergenceSubject, Evaluated
from nof1_causal_lab.artifacts.construct import ConstructSpec
from nof1_causal_lab.artifacts.effects import HistogramBin
from nof1_causal_lab.artifacts.indicator import IndicatorSpec
from nof1_causal_lab.artifacts.raw_data import column_descriptions
from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact
from nof1_causal_lab.study.state import SourceValidity
from nof1_causal_lab.study.view_models import (
    MeasurementsData,
    ObservationRecord,
    RawDataColumnDescription,
    RawDataData,
    RawDataDateRange,
)
from nof1_causal_lab.utils.histograms import histogram_draws

if TYPE_CHECKING:
    import pyarrow as pa

    from nof1_causal_lab.artifacts.identification import IdentificationReport
    from nof1_causal_lab.artifacts.identity import ConstructId, EdgeId, IndicatorId
    from nof1_causal_lab.artifacts.model_checks import ModelPredictiveReport
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.validation_report import (
        DataProfileArtifact,
    )
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.study.snapshot_models import FitSummary, Sourced


def raw_data_view(table: pa.Table, date_range: RawDataDateRange | None) -> RawDataData:
    """Describe the physical table and sample evenly spaced rows."""
    descriptions = column_descriptions(table)
    frame = pl.DataFrame(table)
    indices = np.linspace(0, frame.height - 1, min(15, frame.height), dtype=int).tolist()
    return RawDataData(
        n_records=frame.height,
        n_columns=frame.width,
        date_range=date_range,
        sample=tuple(
            {key: None if value is None else str(value) for key, value in row.items()}
            for row in frame[indices].to_dicts()
        ),
        column_descriptions=tuple(
            RawDataColumnDescription(name=name, dtype=str(dtype), description=descriptions[name])
            for name, dtype in frame.schema.items()
        ),
    )


def observed_histogram(values: np.ndarray, *, discrete: bool) -> list[HistogramBin]:
    """Histogram the actual observed numeric values, excluding missing observations."""
    values = values[np.isfinite(values)]
    if not values.size:
        return []
    if discrete:
        centers, counts = np.unique(values, return_counts=True)
        return [
            HistogramBin(
                bin_center=float(center),
                bin_start=float(center - 0.5),
                bin_end=float(center + 0.5),
                count=int(count),
            )
            for center, count in zip(centers, counts, strict=True)
        ]
    return histogram_draws(values, max_bins=15)


def measurements_view(
    panel: pl.DataFrame, indicator_ids: set[IndicatorId], sample: tuple[ObservationRecord, ...]
) -> MeasurementsData:
    """Counts owned by the selected model and representative rows from one panel."""
    return MeasurementsData(
        n_observations=len(panel),
        per_indicator_counts=dict(
            panel.filter(pl.col("indicator_id").is_in(indicator_ids))
            .group_by("indicator_id")
            .len()
            .sort("indicator_id")
            .iter_rows()
        ),
        combined_extractions_sample=sample,
    )


def likelihood_histograms(
    selection: StructuralSelection, panel: pl.DataFrame | None
) -> dict[IndicatorId, tuple[HistogramBin, ...]]:
    """Observed histograms keyed by their declared indicator, without copied profiles."""
    if panel is None:
        return {}
    return {
        indicator.observation.id: tuple(
            observed_histogram(
                panel.filter(pl.col("indicator_id") == indicator.observation.id)["value"]
                .cast(pl.Float64, strict=False)
                .to_numpy(),
                discrete=likelihood.law.family.is_discrete,
            )
        )
        for indicator, likelihood in selection.model.iter_likelihoods()
    }


def entity_failures(
    model: ModelSpec | None,
    fit: Sourced[FitSummary] | None,
    predictive: Sourced[ModelPredictiveReport] | None,
    identification: Sourced[IdentificationReport] | None,
    data: Sourced[ValidationReportArtifact] | Sourced[DataProfileArtifact] | None,
) -> dict[ConstructId | EdgeId | IndicatorId, tuple[str, ...]]:
    """Attribute recorded scientific failures before the workbench renders them."""
    if model is None:
        return {}
    entities = (*model.constructs, *model.edges, *model.indicators)
    failures: dict[ConstructId | EdgeId | IndicatorId, tuple[str, ...]] = {}
    for entity in entities:
        messages = []
        identity = entity.observation.id if isinstance(entity, IndicatorSpec) else entity.id
        label = (
            entity.observation.name
            if isinstance(entity, IndicatorSpec)
            else entity.name
            if isinstance(entity, ConstructSpec)
            else entity.id
        )
        parameters = {p.id for p in model.parameters_for(identity)}
        if fit is not None and fit.source.validity == SourceValidity.FRESH:
            for assessment in fit.value.report.convergence.assessments:
                if (
                    isinstance(assessment, Evaluated)
                    and assessment.outcome == "failed"
                    and isinstance(assessment.subject, ConvergenceSubject)
                    and assessment.subject.parameter.parameter_id in parameters
                ):
                    messages.append(f"Parameter convergence: {assessment.subject.label}")
        if (
            predictive is not None
            and predictive.source.validity == SourceValidity.FRESH
            and predictive.value.evaluation.kind == "evaluated"
        ):
            for assessment in predictive.value.evaluation.findings:
                if not isinstance(assessment, Evaluated) or assessment.outcome not in {
                    "failed",
                    "error",
                }:
                    continue
                subject = assessment.subject
                target = subject.target.id if not isinstance(subject.target, str) else None
                if target == identity or (
                    subject.construct_id == identity
                    and (
                        not isinstance(entity, ConstructSpec)
                        or target not in {i.observation.id for i in entity.indicators}
                    )
                ):
                    messages.append(f"Predictive checks: {label}")
            if predictive.value.evaluation.predictive_checks is not None:
                for (
                    assessment
                ) in predictive.value.evaluation.predictive_checks.per_variable_warnings:
                    if (
                        isinstance(assessment, Evaluated)
                        and assessment.outcome in {"failed", "warning", "error"}
                        and assessment.subject.target.id == identity
                    ):
                        messages.append(f"Predictive checks: {label}")
        if (
            data is not None
            and data.source.validity == SourceValidity.FRESH
            and isinstance(entity, IndicatorSpec)
        ):
            profile = (
                data.value.data if isinstance(data.value, ValidationReportArtifact) else data.value
            )
            audit = profile.indicators.get(entity.observation.id)
            if audit is not None and any(issue.severity != "info" for issue in audit.issues):
                messages.append(f"Data quality: {label}")
        if (
            identification is not None
            and identification.source.validity == SourceValidity.FRESH
            and isinstance(entity, ConstructSpec)
        ):
            treatment = identification.value.treatments.get(entity.id)
            if treatment is not None and treatment.status == "not_identified":
                messages.append(f"Identification against ★: {label}")
        failures[identity] = tuple(dict.fromkeys(messages))
    for construct in model.constructs:
        failures[construct.id] = tuple(
            dict.fromkeys(
                (
                    *failures[construct.id],
                    *(
                        message
                        for indicator in construct.indicators
                        for message in failures[indicator.observation.id]
                    ),
                )
            )
        )
    return failures
