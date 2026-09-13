"""Compose display payloads using immutable versions selected by the journal."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.baseline_report import SavedScenariosArtifact  # noqa: TC001
from nof1_causal_lab.artifacts.catalog import ARTIFACT_CONTRACTS
from nof1_causal_lab.artifacts.causal_design import CausalDesignArtifact  # noqa: TC001
from nof1_causal_lab.artifacts.compiled_ssm import CompiledSSMArtifact  # noqa: TC001
from nof1_causal_lab.artifacts.effects import HistogramBin
from nof1_causal_lab.artifacts.identity import ArtifactRef
from nof1_causal_lab.artifacts.measurements import MeasurementsArtifact  # noqa: TC001
from nof1_causal_lab.artifacts.raw_data import RawDataArtifact  # noqa: TC001
from nof1_causal_lab.artifacts.statistical_model_spec import (
    StatisticalModelSpecArtifact,  # noqa: TC001
)
from nof1_causal_lab.artifacts.structural_plan import StructuralPlanArtifact  # noqa: TC001
from nof1_causal_lab.artifacts.validation_report import ValidationReportArtifact  # noqa: TC001
from nof1_causal_lab.machine.artifact_files import artifact_file_spec, parquet_filename
from nof1_causal_lab.machine.equations import state_equations
from nof1_causal_lab.machine.view_models import (
    ArtifactViews,
    MeasurementsData,
    MeasurementStructureViewData,
    ModelSpecLikelihoodDiagnostics,
    ObservationRecord,
    RawDataColumnDescription,
    RawDataData,
    RawDataDateRange,
    StatisticalModelSpecData,
)
from nof1_causal_lab.utils.histograms import histogram_draws

if TYPE_CHECKING:
    from pydantic import BaseModel

    from nof1_causal_lab.artifacts.identity import ArtifactId
    from nof1_causal_lab.machine.artifacts import EpisodeState
    from nof1_causal_lab.machine.store import ArtifactStore


def read_payload(store: ArtifactStore, ref: ArtifactRef) -> BaseModel:
    """Validate one immutable primary JSON payload with its production contract."""
    filename = next(iter(artifact_file_spec(ref.artifact_id).json.values()))
    return ARTIFACT_CONTRACTS[ref.artifact_id].model_validate(
        store.read_json_file(ref.artifact_id, ref.version, filename)
    )


def raw_data_view(payload: RawDataArtifact, frame: pl.DataFrame) -> RawDataData:
    """Describe the physical table and sample evenly spaced rows."""
    descriptions = {column.name: column.description for column in payload.column_descriptions}
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


def read_artifact_views(
    store: ArtifactStore, state: EpisodeState, installed_at: dict[ArtifactId, int]
) -> ArtifactViews:
    """Project only compatible selected artifacts; every join checks its input pins."""
    payloads = {
        aid: read_payload(store, ArtifactRef(artifact_id=aid, version=info.version))
        for aid, info in state.current.items()
        if aid != "panel"
    }

    def compatible(output: ArtifactId, *inputs: ArtifactId) -> bool:
        info = state.get(output)
        return info is not None and all(
            (selected := state.get(aid)) is not None
            and info.derived_from.get(aid) == selected.version
            for aid in inputs
        )

    values = {
        aid: payloads[aid]
        for aid in ("latent_structure", "validation_report", "posterior", "baseline_report")
        if aid in payloads
    }
    if "raw_data" in payloads:
        values["raw_data"] = raw_data_view(
            cast("RawDataArtifact", payloads["raw_data"]),
            store.read_parquet_file(
                "raw_data", state.current["raw_data"].version, parquet_filename("raw_data", "raw")
            ),
        )
    if compatible("causal_design", "measurement_structure") and compatible(
        "structural_plan", "causal_design"
    ):
        values["measurement_structure"] = MeasurementStructureViewData(
            causal_design=cast("CausalDesignArtifact", payloads["causal_design"]).causal_design,
            structural_plan=cast(
                "StructuralPlanArtifact", payloads["structural_plan"]
            ).structural_plan,
        )
    panel = None
    if state.has("panel"):
        panel = store.read_parquet_file(
            "panel", state.current["panel"].version, parquet_filename("panel", "panel")
        )
    if (
        panel is not None
        and state.has("measurements")
        and (
            state.current["panel"].derived_from == state.current["measurements"].derived_from
            and installed_at["panel"] == installed_at["measurements"]
        )
    ):
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
            workers=cast("MeasurementsArtifact", payloads["measurements"]).workers,
            per_indicator_counts=dict(
                panel.group_by("indicator_id").len().sort("indicator_id").iter_rows()
            ),
            combined_extractions_sample=sample,
        )
    if "statistical_model_spec" in payloads:
        spec = cast("StatisticalModelSpecArtifact", payloads["statistical_model_spec"])
        diagnostics = {}
        if panel is not None and compatible("statistical_model_spec", "panel", "validation_report"):
            audits = cast("ValidationReportArtifact", payloads["validation_report"]).indicators
            for likelihood in spec.statistical_model_spec.likelihoods:
                observations = panel.filter(pl.col("indicator_id") == likelihood.indicator_id)[
                    "value"
                ]
                numeric = observations.cast(pl.Float64, strict=False).to_numpy()
                audit = audits.get(likelihood.indicator_id)
                discrete = likelihood.distribution in {
                    "poisson",
                    "bernoulli",
                    "negative_binomial",
                    "ordered_logistic",
                    "categorical",
                }
                bins = observed_histogram(numeric, discrete=discrete)
                prior = (spec.prior_predictive_samples or {}).get(likelihood.indicator_id)
                prior_counts = None
                outside = None
                if prior and bins:
                    if likelihood.distribution in {"bernoulli", "ordered_logistic", "categorical"}:
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
                diagnostics[likelihood.indicator_id] = ModelSpecLikelihoodDiagnostics(
                    indicator_id=likelihood.indicator_id,
                    profile=audit.profile if audit else None,
                    histogram=bins,
                    prior_counts=prior_counts,
                    prior_outside_fraction=outside,
                )
        values["statistical_model_spec"] = StatisticalModelSpecData(
            **spec.model_dump(),
            state_equations=state_equations(
                spec.statistical_model_spec,
                cast("StructuralPlanArtifact", payloads["structural_plan"]).structural_plan,
            )
            if compatible("statistical_model_spec", "structural_plan")
            else [],
            likelihood_diagnostics=diagnostics,
            structural_plan=cast(
                "StructuralPlanArtifact", payloads["structural_plan"]
            ).structural_plan
            if compatible("statistical_model_spec", "structural_plan")
            else None,
            parameters=cast("CompiledSSMArtifact", payloads["compiled_ssm"]).parameters
            if compatible("compiled_ssm", "statistical_model_spec")
            else [],
        )
    if "baseline_report" in values and "saved_scenarios" in payloads:
        values["baseline_report"] = values["baseline_report"].model_copy(
            update={
                "saved_scenarios": cast(
                    "SavedScenariosArtifact", payloads["saved_scenarios"]
                ).scenarios
            }
        )
    return ArtifactViews.model_validate(values)
