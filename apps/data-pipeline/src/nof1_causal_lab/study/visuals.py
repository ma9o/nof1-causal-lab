"""Lossless reads of recorded data and draws for the model workbench."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

from nof1_causal_lab.artifacts.availability import Available
from nof1_causal_lab.study.visual_models import (
    EmpiricalPoint,
    ObservationHistory,
    PathSeries,
    RecordedPath,
    SimulationPaths,
)
from nof1_causal_lab.utils.time_coordinates import ObservationInstant

if TYPE_CHECKING:
    from datetime import datetime

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.observations import ResolvedObservationSpec
    from nof1_causal_lab.artifacts.simulation import SimulationEvidence, SimulationReport
    from nof1_causal_lab.study.visual_models import ObservationData


def finite_values(values: np.ndarray) -> tuple[float | None, ...]:
    """Convert numerical values to JSON-safe scalars, representing non-finite entries as ``None``."""
    return tuple(float(value) if np.isfinite(value) else None for value in values)


def empirical_points(values: np.ndarray) -> tuple[EmpiricalPoint, ...]:
    """Compute the empirical CDF at distinct finite values, retaining duplicate frequencies."""
    unique, counts = np.unique(values[np.isfinite(values)], return_counts=True)
    cumulative = np.cumsum(counts) / counts.sum() if counts.size else []
    return tuple(
        EmpiricalPoint(
            value=value,
            probability=float(probability),
        )
        for value, probability, count in zip(unique, cumulative, counts, strict=True)
    )


def observation_history(
    time_origin: datetime | None, variable: ResolvedObservationSpec, panel: pl.DataFrame
) -> ObservationHistory:
    """Project recorded rows into model-day coordinates with their support intervals and empirical CDF."""
    origin = ObservationInstant.origin(time_origin)

    def days(column: str) -> tuple[float | None, ...]:
        return tuple(
            ObservationInstant(value).relative_to(origin).days if value is not None else None
            for value in panel[column]
        )

    values = panel["value"].cast(pl.Float64).to_numpy()
    return ObservationHistory(
        label=variable.name,
        times=tuple(
            ObservationInstant(value).relative_to(origin).days for value in panel["anchor_time"]
        ),
        values=finite_values(values),
        support_start=days("support_start"),
        support_end=days("support_end"),
        time_origin=time_origin,
        levels=variable.ordinal_levels or variable.categorical_levels,
        empirical=empirical_points(values),
    )


def simulation_observation_histories(
    evidence: SimulationEvidence,
    observations: np.ndarray,
    mask: np.ndarray,
    support_start: np.ndarray,
    support_end: np.ndarray,
) -> tuple[ObservationData, ...]:
    """Project every saved replicate into the same history type as prepared user data."""
    return tuple(
        {
            variable.id: ObservationHistory(
                label=variable.name,
                times=evidence.times,
                values=finite_values(values),
                support_start=finite_values(support_start[:, column]),
                support_end=finite_values(support_end[:, column]),
                time_origin=evidence.time_origin,
                levels=variable.ordinal_levels or variable.categorical_levels,
                empirical=empirical_points(values),
            )
            for column, variable in enumerate(evidence.observation_layout.variables)
            for values in (
                np.where(mask[replicate, :, column], observations[replicate, :, column], np.nan),
            )
        }
        for replicate in range(evidence.draws)
    )


def recorded_simulation_paths(
    report: SimulationReport,
    model: ModelSpec,
    latent: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray,
    reference: np.ndarray | None,
    reference_observed: np.ndarray | None,
    *,
    start: int,
    count: int,
) -> SimulationPaths:
    """Project draw pages and surviving labels from the full retained buffers."""
    import jax.numpy as jnp

    from nof1_causal_lab.actions.simulation_summaries import category_probabilities
    from nof1_causal_lab.models.ssm.counterfactual.estimands import summarize_draws

    stop = min(start + count, report.evidence.draws)

    def paths(values: np.ndarray) -> tuple[RecordedPath, ...]:
        return tuple(
            RecordedPath(draw=start + index, values=finite_values(row))
            for index, row in enumerate(values[start:stop])
        )

    effect = None
    summary = None
    reference_mean = None
    manifest = {}
    if isinstance(report.causal, Available):
        # Certification owns both reference keys; the reader hydrates those exact buffers.
        reference = cast("np.ndarray", reference)
        reference_observed = cast("np.ndarray", reference_observed)
        outcome = report.evidence.state_ids.index(report.causal.value.outcome)
        differences = latent[:, :, outcome] - reference[:, :, outcome]
        effect = PathSeries.from_paths(
            label=report.causal.value.labels[report.causal.value.outcome],
            action=paths(differences),
        )
        summary = summarize_draws(jnp.asarray(differences[:, -1]))
        reference_mean = float(reference[:, -1, outcome].mean())
        difference = observed - reference_observed
        manifest = {
            identity: float(difference[:, -1, index].mean())
            for index, identity in enumerate(report.evidence.observation_layout.indicator_ids)
            if np.isfinite(difference[:, -1, index]).all()
        }

    variables = report.evidence.observation_layout.variables
    category_levels = {
        index: ("0", "1")
        if variable.measurement_dtype == "binary"
        else variable.ordinal_levels or variable.categorical_levels
        for index, variable in enumerate(variables)
        if variable.measurement_dtype == "binary"
        or variable.ordinal_levels
        or variable.categorical_levels
    }
    return SimulationPaths(
        times=report.evidence.times,
        time_origin=report.evidence.time_origin,
        total_draws=report.evidence.draws,
        start=start,
        count=stop - start,
        states={
            identity: PathSeries.from_paths(
                label=model.get_construct(identity).name,
                action=paths(latent[:, :, index]),
                reference=paths(reference[:, :, index]) if reference is not None else (),
            )
            for index, identity in enumerate(report.evidence.state_ids)
        },
        indicators={
            variable.id: PathSeries.from_paths(
                label=variable.name,
                action=paths(np.where(mask[:, :, index], observed[:, :, index], np.nan)),
                reference=paths(
                    np.where(mask[:, :, index], reference_observed[:, :, index], np.nan)
                )
                if reference_observed is not None
                else (),
                levels=variable.ordinal_levels or variable.categorical_levels,
            )
            for index, variable in enumerate(variables)
        },
        effect=effect,
        effect_summary=summary,
        reference_mean=reference_mean,
        manifest_effects=manifest,
        action_category_probabilities={
            variables[index].id: category_probabilities(
                observed[:, :, index], mask[:, :, index], levels
            )
            for index, levels in category_levels.items()
            if levels is not None
        },
        reference_category_probabilities={
            variables[index].id: category_probabilities(
                reference_observed[:, :, index], mask[:, :, index], levels
            )
            for index, levels in category_levels.items()
            if levels is not None
        }
        if reference_observed is not None
        else {},
    )
