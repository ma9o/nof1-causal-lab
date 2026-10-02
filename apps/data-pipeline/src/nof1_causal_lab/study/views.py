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
from nof1_causal_lab.numpyro_json import distribution_shape
from nof1_causal_lab.study.equations import (
    confounder_equations,
    observation_equations,
    state_equations,
)
from nof1_causal_lab.study.prior_views import prior_density
from nof1_causal_lab.study.snapshot_models import SourceValidity
from nof1_causal_lab.study.view_models import (
    LikelihoodDiagnostics,
    MeasurementsData,
    ModelDiagnostics,
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
        ValidationReportArtifact,
    )
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


def model_diagnostics_view(
    model: ModelSpec,
    *,
    panel: pl.DataFrame | None,
    validation: ValidationReportArtifact | None,
) -> ModelDiagnostics:
    """Compose equations and plots from findings already selected by the revision reader."""
    diagnostics = {}
    if panel is not None and validation is not None:
        audits = validation.indicators
        for indicator, likelihood in model.iter_likelihoods():
            observations = panel.filter(pl.col("indicator_id") == indicator.id)["value"]
            numeric = observations.cast(pl.Float64, strict=False).to_numpy()
            audit = audits.get(indicator.id)
            discrete = likelihood.law.family in {
                "poisson",
                "bernoulli",
                "negative_binomial",
                "ordered_logistic",
                "categorical",
            }
            bins = observed_histogram(numeric, discrete=discrete)
            diagnostics[indicator.id] = LikelihoodDiagnostics(
                indicator_id=indicator.id,
                profile=audit.profile if audit else None,
                histogram=tuple(bins),
            )
    return ModelDiagnostics(
        prior_densities={
            parameter.id: prior_density(law)
            for parameter in model.parameters
            if (law := model.distribution_for(parameter.id)) is not None
            and distribution_shape(law) == ((), ())
        },
        confounder_equations=tuple(confounder_equations(model))
        if model.measurement_clock is not None and model.indicators
        else (),
        state_equations=tuple(state_equations(model))
        if model.measurement_clock is not None and model.indicators
        else (),
        observation_equations=observation_equations(model),
        likelihood_diagnostics=diagnostics,
    )


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
        label = entity.name if isinstance(entity, (ConstructSpec, IndicatorSpec)) else entity.id
        parameters = {p.id for p in model.parameters_for(entity.id)}
        if fit is not None and fit.source.validity == SourceValidity.FRESH:
            for assessment in fit.value.report.convergence.assessments:
                if (
                    isinstance(assessment, Evaluated)
                    and assessment.outcome == "failed"
                    and isinstance(assessment.subject, ConvergenceSubject)
                    and assessment.subject.parameter.parameter_id in parameters
                ):
                    messages.append(f"Parameter convergence: {assessment.subject.label}")
        if predictive is not None and predictive.source.validity == SourceValidity.FRESH:
            for assessment in predictive.value.findings:
                if not isinstance(assessment, Evaluated) or assessment.outcome not in {
                    "failed",
                    "error",
                }:
                    continue
                subject = assessment.subject
                target = subject.target.id if not isinstance(subject.target, str) else None
                if target == entity.id or (
                    subject.construct_id == entity.id
                    and (
                        not isinstance(entity, ConstructSpec)
                        or target not in {i.id for i in entity.indicators}
                    )
                ):
                    messages.append(f"Predictive checks: {label}")
            if predictive.value.predictive_checks is not None:
                for assessment in predictive.value.predictive_checks.per_variable_warnings:
                    if (
                        isinstance(assessment, Evaluated)
                        and assessment.outcome in {"failed", "warning", "error"}
                        and assessment.subject.target.id == entity.id
                    ):
                        messages.append(f"Predictive checks: {label}")
        if (
            data is not None
            and data.source.validity == SourceValidity.FRESH
            and isinstance(entity, IndicatorSpec)
        ):
            audit = data.value.indicators.get(entity.id)
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
        failures[entity.id] = tuple(dict.fromkeys(messages))
    for construct in model.constructs:
        failures[construct.id] = tuple(
            dict.fromkeys(
                (
                    *failures[construct.id],
                    *(
                        message
                        for indicator in construct.indicators
                        for message in failures[indicator.id]
                    ),
                )
            )
        )
    return failures
