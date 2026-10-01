"""Pure preparation helpers for executable SSM models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl

from nof1_causal_lab.artifacts.scenarios import InterventionSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.counterfactual.orchestration import ResolvedIntervention
from nof1_causal_lab.models.ssm.execution.planning import (
    InferenceStructurePlan,
    plan_inference_structure,
)
from nof1_causal_lab.models.ssm.model import SSMModel
from nof1_causal_lab.models.ssm.observation_support import (
    ObservationSupportRuntime,
    augment_wide_data_with_support_boundaries,
    compile_observation_support_runtime,
    validate_discrete_manifest_metadata,
    validate_observation_support,
)
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightError
from nof1_causal_lab.utils.data import pivot_to_wide
from nof1_causal_lab.utils.time_coordinates import serialization_origin

if TYPE_CHECKING:
    from datetime import datetime

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs
    from nof1_causal_lab.models.ssm.parameter_layout import SSMParameterLayout
    from nof1_causal_lab.sampler_config import (
        SamplerConfig,
        SamplerConfigInput,
    )

logger = logging.getLogger(__name__)


def replay_input_events(
    spec: ModelSpec,
    panel: pl.DataFrame | None,
    *,
    time_origin: datetime | None,
    start: float,
    end: float,
    indicator_column: str = "indicator_id",
) -> tuple[ResolvedIntervention, ...]:
    """Read constant input levels over their windows, then hold to the next reading."""
    events = []
    for index in np.flatnonzero(numeric.input_mask(spec)):
        construct = spec.get_construct(spec.state_order[index])
        if panel is None or time_origin is None:
            raise ValueError(f"Input {construct.name!r} requires a panel with a time origin")
        origin = serialization_origin(time_origin)
        readings: dict[float, float] = {}
        for indicator in construct.indicators:
            identity = indicator.id if indicator_column == "indicator_id" else indicator.name
            rows = panel.filter(pl.col(indicator_column) == identity).drop_nulls("value")
            for row in rows.iter_rows(named=True):
                at = row["support_start"]
                time = (at.replace(tzinfo=None) - origin).total_seconds() / 86400.0
                value = float(row["value"])
                if indicator.summary_operator in {"count", "sum"}:
                    duration = (row["support_end"] - row["support_start"]).total_seconds() / 86400.0
                    value /= duration
                if not np.isfinite(value):
                    raise ValueError(f"Input {construct.name!r} requires finite readings")
                if time in readings and readings[time] != value:
                    raise ValueError(f"Input {construct.name!r} has conflicting readings at {time}")
                readings[time] = value
        eligible = sorted(time for time in readings if time <= start)
        if not eligible:
            raise ValueError(
                f"Input {construct.name!r} has no value at the start of the requested history"
            )
        record = [(start, readings[eligible[-1]])]
        record.extend((time, readings[time]) for time in sorted(readings) if start < time <= end)
        if not construct.is_dynamic and len({value for _, value in record}) != 1:
            raise ValueError(f"Time-invariant input {construct.name!r} has varying readings")
        events.extend(
            ResolvedIntervention(
                int(index), InterventionSpec(target=construct.id, time=time, value=value)
            )
            for time, value in record
        )
    return tuple(sorted(events, key=lambda event: (event.spec.time, event.index)))


def replay_input_values(spec: ModelSpec, times, events) -> jnp.ndarray:
    """Expand dated readings on every fit grid point; modeled coordinates stay unknown."""
    grid = np.asarray(times)
    values = np.full((len(grid), numeric.n_states(spec)), np.nan, dtype=grid.dtype)
    for event in events:
        values[grid >= np.asarray(event.spec.time, dtype=grid.dtype), event.index] = (
            event.spec.value
        )
    return jnp.asarray(values)


def _standardize_manifest_columns(
    wide_data: pl.DataFrame,
    manifest_cols: list[str],
    manifest_standardized: list[bool],
) -> pl.DataFrame:
    """Apply deterministic standardization to manifest columns marked standardized.

    Flagged columns become (y - mean) / sd so their link-scale spread is exactly 1,
    matching the standardized-latent convention the priors are authored under. When
    sd is 0 or undefined every centered value is already 0, so any divisor yields
    identical data; 1 is the canonical completion, not a fallback.
    """
    if not any(manifest_standardized):
        return wide_data

    standardized_exprs = []
    for manifest_name, standardized in zip(manifest_cols, manifest_standardized, strict=False):
        base_expr = pl.col(manifest_name).cast(pl.Float64, strict=False)
        if standardized:
            centered_expr = base_expr - base_expr.mean()
            scale_expr = base_expr.std()
            standardized_exprs.append(
                (
                    centered_expr
                    / pl.when(scale_expr > 0.0).then(scale_expr).otherwise(pl.lit(1.0))
                ).alias(manifest_name)
            )
        else:
            standardized_exprs.append(base_expr.alias(manifest_name))
    passthrough = [col_name for col_name in wide_data.columns if col_name not in set(manifest_cols)]
    return wide_data.select(*passthrough, *standardized_exprs)


@dataclass
class PreparedModelRuntime:
    """Canonical prepared runtime context shared by validation and inference."""

    model: SSMModel
    sampler_config: SamplerConfigInput
    wide_data: pl.DataFrame
    observation_data: pl.DataFrame | None
    observation_support: ObservationSupportRuntime | None
    inference_structure: InferenceStructurePlan
    observations: jnp.ndarray
    times: jnp.ndarray

    @property
    def spec(self) -> ModelSpec:
        return self.model.spec

    @property
    def parameter_layout(self) -> SSMParameterLayout:
        return self.model.parameter_layout


def get_default_sampler_config() -> SamplerConfig:
    """Return default sampler configuration from config.yaml."""
    from nof1_causal_lab.utils.config import get_config

    return get_config().inference.to_sampler_config()


def build_ssm_model(wide_data: pl.DataFrame, *, inputs: CompiledFitInputs) -> SSMModel:
    """Check the later panel boundary against compiled fit inputs."""
    if wide_data.is_empty():
        raise ValueError("Cannot build SSM model from empty data")
    validate_discrete_manifest_metadata(inputs.spec, wide_data)
    validate_observation_support(inputs.spec, wide_data)
    return SSMModel(inputs)


def prepare_fit_inputs(
    spec: ModelSpec,
    wide_data: pl.DataFrame,
) -> tuple[jnp.ndarray, jnp.ndarray, list[str], pl.DataFrame]:
    """Extract observations, times, manifest order, and standardized wide data."""
    manifest_cols = numeric.observation_names(spec)
    manifest_standardized = numeric.observation_standardized(spec)
    standardized_data = _standardize_manifest_columns(
        wide_data, manifest_cols, manifest_standardized
    )
    observations = jnp.array(standardized_data.select(manifest_cols).to_numpy(), dtype=jnp.float32)
    if "time" in standardized_data.columns:
        times = jnp.array(standardized_data["time"].to_numpy(), dtype=jnp.float32)
    else:
        times = jnp.arange(standardized_data.height, dtype=jnp.float32)
    return observations, times, manifest_cols, standardized_data


def prepare_wide_model_runtime(
    wide_data: pl.DataFrame,
    *,
    inputs: CompiledFitInputs,
    sampler_config: SamplerConfigInput | None = None,
    observation_data: pl.DataFrame | None = None,
    time_origin: datetime | None,
) -> PreparedModelRuntime:
    """Prepare panel-dependent runtime state for one compiled model."""
    resolved_sampler_config = sampler_config or get_default_sampler_config()
    model = build_ssm_model(wide_data, inputs=inputs)

    spec = model.spec
    manifest_names = numeric.observation_names(spec)
    wide_data = augment_wide_data_with_support_boundaries(
        observation_data,
        wide_data,
        manifest_names,
        time_origin=time_origin,
    )
    observations, times, manifest_names, wide_data = prepare_fit_inputs(spec, wide_data)
    observation_support = compile_observation_support_runtime(
        observation_data,
        wide_data,
        manifest_names,
        time_origin=time_origin,
    )
    model.set_observation_support(observation_support)
    model.input_events = replay_input_events(
        spec,
        observation_data,
        time_origin=time_origin,
        start=float(times[0]),
        end=float(times[-1]),
        indicator_column="indicator",
    )
    if model.input_events:
        model.input_values = replay_input_values(spec, times, model.input_events)
        input_columns = [
            index
            for index, indicator in enumerate(numeric.observed_indicators(spec))
            if spec.indicator_owner(indicator.id).role == "exogenous"
        ]
        observations = observations.at[:, jnp.asarray(input_columns)].set(jnp.nan)
    inference_structure = plan_inference_structure(
        spec,
        observation_support=observation_support,
        method_override=resolved_sampler_config.get("method"),
        n_timepoints=int(times.shape[0]),
    )
    if observation_support is not None and observation_support.requires_interval_summary_handling:
        interval_summary_desc = ", ".join(
            f"{name} ({operator})"
            for name, operator, support_kind in zip(
                observation_support.manifest_names,
                observation_support.summary_operators,
                observation_support.support_kinds,
                strict=False,
            )
            if support_kind == "interval" and operator is not None
        )
        logger.info(
            "Prepared runtime compiled support-aware observation semantics for %s.",
            interval_summary_desc,
        )
    return PreparedModelRuntime(
        model=model,
        sampler_config=resolved_sampler_config,
        wide_data=wide_data,
        observation_data=observation_data,
        observation_support=observation_support,
        inference_structure=inference_structure,
        observations=observations,
        times=times,
    )


def prepare_model_runtime(
    data_for_model: pl.DataFrame,
    *,
    inputs: CompiledFitInputs,
    time_origin: datetime | None,
    sampler_config: SamplerConfigInput | None = None,
) -> PreparedModelRuntime:
    """Canonical entry point for preparing stage data for model work."""
    wide_data, runtime_rows = project_observation_data(
        data_for_model, model_spec=inputs.spec, time_origin=time_origin
    )
    return prepare_wide_model_runtime(
        wide_data,
        inputs=inputs,
        sampler_config=sampler_config,
        observation_data=runtime_rows,
        time_origin=time_origin,
    )


def project_observation_data(
    data_for_model: pl.DataFrame, *, model_spec: ModelSpec, time_origin: datetime | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Resolve indicator identities without compiling parameter laws or fitting."""
    labels = {indicator.id: indicator.name for indicator in model_spec.indicators}
    selected = data_for_model.filter(pl.col("indicator_id").is_in(list(labels)))
    present = set(selected["indicator_id"])
    missing = [
        f"{name} ({identity})" for identity, name in labels.items() if identity not in present
    ]
    if missing:
        raise ObservationPreflightError(
            "Prepared observations are missing model indicators: " + ", ".join(missing)
        )
    wide_data = pivot_to_wide(selected, time_origin=time_origin)
    runtime_rows = selected.rename({"indicator_id": "indicator"})
    wide_data = wide_data.rename(
        {iid: name for iid, name in labels.items() if iid in wide_data.columns}
    )
    runtime_rows = runtime_rows.with_columns(pl.col("indicator").replace_strict(labels))
    return wide_data, runtime_rows
