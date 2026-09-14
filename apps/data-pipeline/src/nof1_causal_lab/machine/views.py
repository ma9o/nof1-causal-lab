"""Compose display payloads using immutable versions selected by the journal."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.admission import AdmissionReport  # noqa: TC001
from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS
from nof1_causal_lab.artifacts.effects import HistogramBin
from nof1_causal_lab.artifacts.identity import ArtifactRef
from nof1_causal_lab.artifacts.model_spec import ModelSpec  # noqa: TC001
from nof1_causal_lab.artifacts.raw_data import column_descriptions
from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact  # noqa: TC001
from nof1_causal_lab.machine.artifact_files import artifact_file_spec, parquet_filename
from nof1_causal_lab.machine.equations import (
    confounder_equations,
    observation_equations,
    state_equations,
)
from nof1_causal_lab.machine.moves import is_stale
from nof1_causal_lab.machine.prior_views import prior_density
from nof1_causal_lab.machine.view_models import (
    ArtifactViews,
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

    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.store import ArtifactStore


def read_payload(store: ArtifactStore, ref: ArtifactRef) -> BaseModel:
    """Validate one immutable primary JSON payload with its production contract."""
    filename = next(iter(artifact_file_spec(ref.artifact_id).json.values()))
    return ARTIFACT_CONTRACTS[ref.artifact_id].model_validate(
        store.read_json_file(ref.artifact_id, ref.version, filename)
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


def read_artifact_views(store: ArtifactStore, state: EpisodeState) -> ArtifactViews:
    """Project only compatible selected artifacts; every join checks its input pins."""
    payloads = {
        aid: read_payload(store, ArtifactRef(artifact_id=aid, version=info.version))
        for aid, info in state.current.items()
        if aid in ARTIFACT_CONTRACTS
    }

    values = {
        aid: payloads[aid]
        for aid in (
            "model",
            "admission_report",
            "validation_report",
            "baseline_report",
        )
        if aid in payloads
    }
    if state.has("raw_data"):
        values["raw_data"] = raw_data_view(
            store.read_parquet_table(
                "raw_data", state.current["raw_data"].version, parquet_filename("raw_data", "raw")
            ),
        )
    if state.has("model"):
        from nof1_causal_lab.artifacts.posterior import InferenceReport
        from nof1_causal_lab.machine.inference import inference_report_record
        from nof1_causal_lab.machine.store import EpisodeJournal

        record = inference_report_record(EpisodeJournal(store.workspace_id).read_all(), state)
        if record is not None:
            values["inference_report"] = InferenceReport.model_validate(
                record.diagnostics["report"]
            )
    panel = None
    if state.has("panel"):
        panel = store.read_parquet_file(
            "panel", state.current["panel"].version, parquet_filename("panel", "panel")
        )
    if panel is not None:
        sample = []
        for row in panel.head(20).to_dicts():
            record = {
                key: value for key, value in row.items() if key in ObservationRecord.__annotations__
            }
            for key in ("anchor_time", "support_start", "support_end"):
                if record.get(key) is not None:
                    record[key] = str(record[key])
            sample.append(TypeAdapter(ObservationRecord).validate_python(record))
        values["measurements"] = MeasurementsData(
            n_observations=len(panel),
            per_indicator_counts=dict(
                panel.group_by("indicator_id").len().sort("indicator_id").iter_rows()
            ),
            combined_extractions_sample=sample,
        )
    if "model" in payloads:
        model = cast("ModelSpec", payloads["model"])
        admission = (
            cast("AdmissionReport", payloads["admission_report"])
            if "admission_report" in payloads
            else None
        )
        diagnostics = {}
        if panel is not None and state.matches_inputs("validation_report", "panel", "model"):
            audits = cast("ValidationReportArtifact", payloads["validation_report"]).indicators
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
                prior = (
                    (admission.prior_predictive_samples or {}).get(indicator.id)
                    if admission is not None and not is_stale(state, "admission_report")
                    else None
                )
                prior_counts = None
                outside = None
                if prior and bins:
                    if likelihood.law.family in {"bernoulli", "ordered_logistic", "categorical"}:
                        present = {bin.bin_center for bin in bins}
                        bins += [
                            HistogramBin(
                                bin_center=value,
                                bin_start=value - 0.5,
                                bin_end=value + 0.5,
                                count=0,
                            )
                            for value in sorted(set(prior) - present)
                        ]
                        bins.sort(key=lambda bin: bin.bin_center)
                    samples = np.asarray(prior)
                    counts = [
                        int(
                            np.count_nonzero(
                                (samples >= bin.bin_start)
                                & (
                                    samples <= bin.bin_end
                                    if index == len(bins) - 1
                                    else samples < bin.bin_end
                                )
                            )
                        )
                        for index, bin in enumerate(bins)
                    ]
                    n_obs = sum(bin.count for bin in bins)
                    prior_counts = [count / len(prior) * n_obs for count in counts]
                    outside = 1 - sum(counts) / len(prior)
                diagnostics[indicator.id] = LikelihoodDiagnostics(
                    indicator_id=indicator.id,
                    profile=audit.profile if audit else None,
                    histogram=bins,
                    prior_counts=prior_counts,
                    prior_outside_fraction=outside,
                )
        values["model_diagnostics"] = ModelDiagnostics(
            prior_densities={
                parameter.id: prior_density(parameter.distribution)
                for parameter in model.parameters
                if parameter.distribution is not None
                and not isinstance(parameter.distribution, str)
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
    return ArtifactViews.model_validate(values)
