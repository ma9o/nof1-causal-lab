"""Pure preparation helpers for executable SSM models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl

from nof1_causal_lab.artifacts.observation_data import (
    ObservationDataset,
    SelectedObservations,
)
from nof1_causal_lab.artifacts.scenarios import StateAssignment
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs
from nof1_causal_lab.models.ssm.counterfactual.orchestration import ResolvedIntervention
from nof1_causal_lab.models.ssm.observation_support import (
    ObservationSupportRuntime,
    augment_wide_data_with_support_boundaries,
    compile_observation_support_runtime,
    validate_observation_support,
)
from nof1_causal_lab.models.ssm.preflight import ObservationPreflightFailure
from nof1_causal_lab.utils.observation_rows import (
    pivot_to_wide,
)
from nof1_causal_lab.utils.time_coordinates import ObservationInstant

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    import jax
    import pyarrow as pa

    from nof1_causal_lab.artifacts.identity import IndicatorId
    from nof1_causal_lab.models.ssm.compile.inputs import (
        CompilationFailure,
        CompiledModel,
        FitCompilationResult,
    )


def reading_level(
    value: float, support_start: datetime, support_end: datetime, summary_operator: str
) -> float:
    """An input reading's constant level: its value, or a count or sum per model day."""
    if summary_operator in {"count", "sum"}:
        return value / (
            ObservationInstant(support_end).relative_to(ObservationInstant(support_start)).days
        )
    return value


def input_trajectory_events(
    spec: CompiledModel,
    *,
    time_origin: datetime | None,
    start: float,
    end: float,
) -> tuple[ResolvedIntervention, ...] | ObservationPreflightFailure:
    """Resolve model-owned step functions on the execution clock, holding their final value."""
    events = []
    for trajectory in spec.input_trajectories:
        origin = trajectory.layout.time_origin
        offset = (
            ObservationInstant(origin).relative_to(ObservationInstant.origin(time_origin)).days
            if not isinstance(origin, str)
            else 0.0
        )
        times = tuple(time + offset for time in trajectory.layout.time_points)
        index = int(np.searchsorted(times, start, side="right")) - 1
        if index < 0:
            return ObservationPreflightFailure.rejected(
                f"Input {spec.states[trajectory.index].name!r} has no value at the start of the requested history"
            )
        points = (
            (start, trajectory.values[index]),
            *(
                (time, value)
                for time, value in zip(times, trajectory.values, strict=True)
                if start < time <= end
            ),
        )
        events.extend(
            ResolvedIntervention(
                trajectory.index,
                StateAssignment(target=trajectory.construct_id, time=time, value=value),
            )
            for time, value in points
        )
    return tuple(sorted(events, key=lambda event: (event.spec.time, event.index)))


def input_trajectory_values(
    spec: CompiledModel,
    times: np.ndarray | jax.Array | Sequence[float],
    events: tuple[ResolvedIntervention, ...],
) -> jnp.ndarray:
    """Expand deterministic input paths on the execution grid; endogenous states stay unknown."""
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
        """Observation identities in the column order used by this bound panel."""
        return tuple(observation.id for observation in self.model.observations)

    @property
    def observation_mask(self) -> jax.Array:
        """Boolean mask of recorded panel entries, with NaN values treated as missing."""
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


@dataclass(frozen=True)
class PreparedFit:
    """Fit laws and observations bound to the same compiled model."""

    inputs: CompiledFitInputs
    panel: BoundPanel


def prepare_fit(
    inputs: FitCompilationResult,
    observations: ObservationDataset,
    *,
    time_origin: datetime | None,
) -> PreparedFit | CompilationFailure | PanelPreparationFailure:
    """Forward compilation failures or bind the resolved fitting inputs once."""
    if not isinstance(inputs, CompiledFitInputs):
        return inputs
    panel = bind_panel(observations, model=inputs.compiled, time_origin=time_origin)
    if isinstance(panel, PanelPreparationFailure):
        return panel
    return PreparedFit(inputs, panel)


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
    observations: ObservationDataset,
    *,
    model: CompiledModel,
    time_origin: datetime | None,
) -> BoundPanel | PanelPreparationFailure:
    """Own compatibility once, retaining the exact identity-bearing input rows."""
    observation_selection = observations.select(
        tuple(item.observation for item in model.observations)
    )
    if not isinstance(observation_selection, SelectedObservations):
        return PanelPreparationFailure(observation_selection.message)
    data_for_model = observation_selection.frame
    time_origin = ObservationInstant.origin(time_origin).value
    if data_for_model.is_empty():
        return PanelPreparationFailure("Cannot bind an empty observation panel")
    projected = project_observation_data(data_for_model, model_spec=model, time_origin=time_origin)
    if isinstance(projected, ObservationPreflightFailure):
        return PanelPreparationFailure(projected.message)
    wide, selected = projected
    rows = selected
    labels = {observation.id: observation.name for observation in model.observations}
    support_rows = rows.with_columns(
        pl.col("indicator_id").replace_strict(labels).alias("indicator")
    )
    wide = augment_wide_data_with_support_boundaries(support_rows, wide, time_origin=time_origin)
    support_failure = validate_observation_support(model, wide)
    if support_failure is not None:
        return PanelPreparationFailure(support_failure.message)
    events = input_trajectory_events(
        model, time_origin=time_origin, start=float(wide["time"][0]), end=float(wide["time"][-1])
    )
    if isinstance(events, ObservationPreflightFailure):
        return PanelPreparationFailure(events.message)
    if events:
        boundaries = pl.DataFrame({"time": sorted({event.spec.time for event in events})})
        wide = wide.join(boundaries, on="time", how="full", coalesce=True).sort("time")
    values, times, names, wide = prepare_fit_inputs(model, wide)
    support = compile_observation_support_runtime(
        support_rows, wide, names, time_origin=time_origin
    )
    if isinstance(support, ObservationPreflightFailure):
        return PanelPreparationFailure(support.message)
    input_values = input_trajectory_values(model, times, events)
    channels = tuple(
        index
        for index, observation in enumerate(model.observations)
        if model.states[observation.state_index].is_input
    )
    if channels:
        from nof1_causal_lab.models.ssm.execution.observation_operator import (
            compile_observation_operator,
        )

        operator = compile_observation_operator(support, held_channels=channels)
        response = input_values[:, jnp.asarray([item.state_index for item in model.observations])]
        expected, _ = operator.project_response_trajectory(response)
        for channel in channels:
            recorded = np.asarray(values[:, channel])
            consistent = np.isnan(recorded) | np.isclose(
                recorded, np.asarray(expected[:, channel]), rtol=1e-6, atol=1e-6
            )
            if not consistent.all():
                return PanelPreparationFailure(
                    f"Exact readings for {model.observations[channel].name!r} conflict with its deterministic trajectory law"
                )
    return BoundPanel(
        model, selected.to_arrow(), time_origin, values, times, support, input_values, events
    )


def project_observation_data(
    data_for_model: pl.DataFrame, *, model_spec: CompiledModel, time_origin: datetime | None
) -> tuple[pl.DataFrame, pl.DataFrame] | ObservationPreflightFailure:
    """Project selected identities to display columns without changing the exact rows."""
    labels = {indicator.id: indicator.name for indicator in model_spec.observations}
    selected = data_for_model.filter(pl.col("indicator_id").is_in(list(labels)))
    present = set(selected["indicator_id"])
    missing = [
        f"{name} ({identity})" for identity, name in labels.items() if identity not in present
    ]
    if missing:
        return ObservationPreflightFailure.rejected(
            "Prepared observations are missing model indicators: " + ", ".join(missing)
        )
    wide_data = pivot_to_wide(selected, time_origin=time_origin)
    return wide_data.rename({str(identity): name for identity, name in labels.items()}), selected
