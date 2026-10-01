"""Compose display payloads using immutable versions selected by a Git snapshot."""

from __future__ import annotations

from datetime import date, datetime
from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS
from nof1_causal_lab.artifacts.effects import HistogramBin
from nof1_causal_lab.artifacts.raw_data import column_descriptions
from nof1_causal_lab.numpyro_json import distribution_shape
from nof1_causal_lab.study.artifact_files import artifact_file_spec
from nof1_causal_lab.study.equations import (
    confounder_equations,
    observation_equations,
    state_equations,
)
from nof1_causal_lab.study.prior_views import prior_density
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
    from pydantic import BaseModel

    from nof1_causal_lab.artifacts.identity import ArtifactId, IndicatorId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact
    from nof1_causal_lab.study.store import ArtifactStore


def read_payload(store: ArtifactStore, artifact_id: ArtifactId, revision: str) -> BaseModel:
    """Validate one immutable primary JSON payload with its production contract."""
    filename = next(iter(artifact_file_spec(artifact_id).json.values()))
    return ARTIFACT_CONTRACTS[artifact_id].model_validate(
        store.read_json_file(artifact_id, revision, filename),
        context={"distribution_array_loader": cache(store.read_array)},
    )


def raw_data_view(table: pa.Table) -> RawDataData:
    """Describe the physical table and sample evenly spaced rows."""
    descriptions = column_descriptions(table)
    frame = pl.DataFrame(table)
    dates: list[str] = []
    for candidate in ("timestamp", "date", "time", "datetime"):
        if candidate not in frame.columns:
            continue
        for value in frame[candidate].drop_nulls():
            if isinstance(value, (date, datetime)):
                dates.append(value.isoformat()[:10])
            elif isinstance(value, str):
                dates.append(datetime.fromisoformat(value).date().isoformat())
        if dates:
            break
    indices = np.linspace(0, frame.height - 1, min(15, frame.height), dtype=int).tolist()
    return RawDataData(
        n_records=frame.height,
        n_columns=frame.width,
        date_range=RawDataDateRange(
            start=min(dates) if dates else "", end=max(dates) if dates else ""
        ),
        sample=[
            {key: None if value is None else str(value) for key, value in row.items()}
            for row in frame[indices].to_dicts()
        ],
        column_descriptions=[
            RawDataColumnDescription(name=name, dtype=str(dtype), description=descriptions[name])
            for name, dtype in frame.schema.items()
        ],
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


def measurements_view(panel: pl.DataFrame, indicator_ids: set[IndicatorId]) -> MeasurementsData:
    """Counts owned by the selected model and representative rows from one panel."""
    sample = []
    for row in panel.head(20).to_dicts():
        record = {
            key: value for key, value in row.items() if key in ObservationRecord.__annotations__
        }
        for key in ("anchor_time", "support_start", "support_end"):
            if record.get(key) is not None:
                record[key] = str(record[key])
        sample.append(TypeAdapter(ObservationRecord).validate_python(record))
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
                histogram=bins,
            )
    return ModelDiagnostics(
        prior_densities={
            parameter.id: prior_density(law)
            for parameter in model.parameters
            if (law := model.distribution_for(parameter.id)) is not None
            and distribution_shape(law) == ((), ())
        },
        confounder_equations=confounder_equations(model)
        if model.measurement_clock is not None and model.indicators
        else [],
        state_equations=state_equations(model)
        if model.measurement_clock is not None and model.indicators
        else [],
        observation_equations=observation_equations(model),
        likelihood_diagnostics=diagnostics,
    )
