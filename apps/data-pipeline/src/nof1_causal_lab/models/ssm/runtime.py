"""Pure preparation helpers for executable SSM models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl

from nof1_causal_lab.artifacts.scenarios import InterventionSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.counterfactual.orchestration import ResolvedIntervention
from nof1_causal_lab.models.ssm.observation_support import (
    ObservationSupportRuntime,
    augment_wide_data_with_support_boundaries,
    compile_observation_support_runtime,
    validate_discrete_manifest_metadata,
    validate_observation_support,
)
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightError
from nof1_causal_lab.utils.observation_rows import (
    ensure_datetime_column,
    observation_row_schema,
    pivot_to_wide,
)
from nof1_causal_lab.utils.time_coordinates import ObservationInstant

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    import jax
    import pyarrow as pa

    from nof1_causal_lab.artifacts.identity import IndicatorId
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel


def replay_input_events(
    spec: CompiledModel,
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
        construct = spec.states[index]
        if panel is None:
            raise ObservationPreflightError(
                f"Input {construct.name!r} requires a panel with a time origin"
            )
        origin = ObservationInstant.origin(time_origin)
        readings: dict[float, float] = {}
        for indicator in spec.observations:
            if indicator.state_index != index:
                continue
            identity = indicator.id if indicator_column == "indicator_id" else indicator.name
            rows = panel.filter(pl.col(indicator_column) == identity).drop_nulls("value")
            for row in rows.iter_rows(named=True):
                at = row["support_start"]
                time = ObservationInstant(at).relative_to(origin).days
                value = float(row["value"])
                if indicator.support.summary_operator in {"count", "sum"}:
                    duration = (
                        ObservationInstant(row["support_end"])
                        .relative_to(ObservationInstant(row["support_start"]))
                        .days
                    )
                    value /= duration
                if not np.isfinite(value):
                    raise ObservationPreflightError(
                        f"Input {construct.name!r} requires finite readings"
                    )
                if time in readings and readings[time] != value:
                    raise ObservationPreflightError(
                        f"Input {construct.name!r} has conflicting readings at {time}"
                    )
                readings[time] = value
        eligible = sorted(time for time in readings if time <= start)
        if not eligible:
            raise ObservationPreflightError(
                f"Input {construct.name!r} has no value at the start of the requested history"
            )
        record = [(start, readings[eligible[-1]])]
        record.extend((time, readings[time]) for time in sorted(readings) if start < time <= end)
        if construct.time_invariant and len({value for _, value in record}) != 1:
            raise ObservationPreflightError(
                f"Time-invariant input {construct.name!r} has varying readings"
            )
        events.extend(
            ResolvedIntervention(
                int(index), InterventionSpec(target=construct.id, time=time, value=value)
            )
            for time, value in record
        )
    return tuple(sorted(events, key=lambda event: (event.spec.time, event.index)))


def replay_input_values(
    spec: CompiledModel,
    times: np.ndarray | jax.Array | Sequence[float],
    events: tuple[ResolvedIntervention, ...],
) -> jnp.ndarray:
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
    manifest_cols: Sequence[str],
    manifest_standardized: Sequence[bool],
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


@dataclass(frozen=True)
class BoundPanel:
    """Immutable observations and exogenous bindings on one compiled model's axes."""

    model: CompiledModel
    rows: pa.Table
    time_origin: datetime
    values: jax.Array
    times: jax.Array
    observation_support: ObservationSupportRuntime
    input_values: jax.Array
    input_events: tuple[ResolvedIntervention, ...]

    @property
    def indicator_ids(self) -> tuple[IndicatorId, ...]:
        return tuple(observation.id for observation in self.model.observations)

    @property
    def observation_mask(self) -> jax.Array:
        return ~jnp.isnan(self.values)

    @property
    def observations(self) -> jax.Array:
        """Only endogenous measurements contribute to the fitted likelihood."""
        input_columns = jnp.asarray(
            tuple(
                self.model.states[observation.state_index].is_input
                for observation in self.model.observations
            )
        )
        return jnp.where(input_columns[None, :], jnp.nan, self.values)


@dataclass(frozen=True)
class PanelPreparationFailure:
    """The supplied history cannot be bound to the compiled model."""

    message: str


def prepare_fit_inputs(
    spec: CompiledModel,
    wide_data: pl.DataFrame,
) -> tuple[jnp.ndarray, jnp.ndarray, tuple[str, ...], pl.DataFrame]:
    """Extract numerical observations on the explicit bound time grid."""
    manifest_cols = numeric.observation_names(spec)
    standardized_data = _standardize_manifest_columns(
        wide_data, manifest_cols, numeric.observation_standardized(spec)
    )
    observations = jnp.array(standardized_data.select(manifest_cols).to_numpy(), dtype=jnp.float32)
    times = jnp.array(standardized_data["time"].to_numpy(), dtype=jnp.float32)
    return observations, times, manifest_cols, standardized_data


def bind_panel(
    data_for_model: pl.DataFrame,
    *,
    model: CompiledModel,
    time_origin: datetime | None,
) -> BoundPanel | PanelPreparationFailure:
    """Own compatibility once, retaining the exact identity-bearing input rows."""
    time_origin = ObservationInstant.origin(time_origin).value
    if data_for_model.is_empty():
        return PanelPreparationFailure("Cannot bind an empty observation panel")
    missing = observation_row_schema().keys() - set(data_for_model.columns)
    if missing:
        return PanelPreparationFailure(
            f"Observation table is missing canonical columns: {sorted(missing)}"
        )
    try:
        wide, selected = project_observation_data(
            data_for_model, model_spec=model, time_origin=time_origin
        )
        rows = selected
        for column in ("anchor_time", "support_start", "support_end"):
            rows = ensure_datetime_column(rows, column)
        if rows["anchor_time"].null_count():
            return PanelPreparationFailure("Observations require non-null anchor_time")
        labels = {observation.id: observation.name for observation in model.observations}
        support_rows = rows.with_columns(
            pl.col("indicator_id").replace_strict(labels).alias("indicator")
        )
        wide = augment_wide_data_with_support_boundaries(
            support_rows, wide, time_origin=time_origin
        )
        validate_discrete_manifest_metadata(model, wide)
        validate_observation_support(model, wide)
        observations, times, names, wide = prepare_fit_inputs(model, wide)
        support = compile_observation_support_runtime(
            support_rows, wide, names, time_origin=time_origin
        )
        events = replay_input_events(
            model, rows, time_origin=time_origin, start=float(times[0]), end=float(times[-1])
        )
        input_values = replay_input_values(model, times, events)
        return BoundPanel(
            model,
            selected.to_arrow(),
            time_origin,
            observations,
            times,
            support,
            input_values,
            events,
        )
    except ObservationPreflightError as exc:
        return PanelPreparationFailure(str(exc))


def project_observation_data(
    data_for_model: pl.DataFrame, *, model_spec: CompiledModel, time_origin: datetime | None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Project selected identities to display columns without changing the exact rows."""
    labels = {indicator.id: indicator.name for indicator in model_spec.observations}
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
    return wide_data.rename({str(identity): name for identity, name in labels.items()}), selected
