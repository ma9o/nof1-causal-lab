"""Materialize simulation summaries from the generated draws at publication time."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.simulation import (
    CategoryProbabilitySummary,
    SimulationPredictiveReport,
    SimulationSeriesSummary,
    TrajectorySummary,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nof1_causal_lab.artifacts.effects import EffectTrajectoryPoint

# Fixed equal-tail 95% intervals for saved predictive and paired-effect summaries.
SIMULATION_QUANTILES = (0.025, 0.975)


def paired_effect_trajectory(
    times: Sequence[float], differences: np.ndarray
) -> tuple[EffectTrajectoryPoint, ...]:
    """Fixed 95% intervals of paired draw-wise effects on absolute model days."""
    from nof1_causal_lab.artifacts.effects import EffectTrajectoryPoint

    return tuple(
        EffectTrajectoryPoint(
            day=float(time),
            effect=float(values.mean()),
            lower_95=float(np.quantile(values, SIMULATION_QUANTILES[0])),
            upper_95=float(np.quantile(values, SIMULATION_QUANTILES[1])),
        )
        for time, values in zip(times, differences.T, strict=True)
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.artifacts.observations import ObservationSpec
    from nof1_causal_lab.artifacts.simulation import FitReliability, PredictiveSummary


def _summarize(
    values: np.ndarray, mask: np.ndarray, levels: tuple[str, ...] | None = None
) -> PredictiveSummary:
    mask = mask & np.isfinite(values)
    counts = tuple(int(count) for count in mask.sum(axis=0))
    selected = [samples[present] for samples, present in zip(values.T, mask.T, strict=True)]
    if levels is not None:
        observed = values[mask]
        if not np.all(
            (observed == np.floor(observed)) & (observed >= 0) & (observed < len(levels))
        ):
            raise ValueError("Simulation emissions must use their declared category codes")
        return CategoryProbabilitySummary(
            probabilities={
                level: tuple(
                    float(np.mean(samples == index)) if samples.size else None
                    for samples in selected
                )
                for index, level in enumerate(levels)
            },
            n_draws=counts,
        )
    return TrajectorySummary(
        mean=tuple(float(samples.mean()) if samples.size else None for samples in selected),
        lower=tuple(
            float(np.quantile(samples, SIMULATION_QUANTILES[0])) if samples.size else None
            for samples in selected
        ),
        upper=tuple(
            float(np.quantile(samples, SIMULATION_QUANTILES[1])) if samples.size else None
            for samples in selected
        ),
        n_draws=counts,
    )


def summarize_simulation(
    model: ModelSpec,
    *,
    state_ids: tuple[ConstructId, ...],
    variables: tuple[ObservationSpec, ...],
    latent_paths: np.ndarray,
    observations: np.ndarray,
    mask: np.ndarray,
    reference_latent_paths: np.ndarray | None,
    reference_observations: np.ndarray | None,
    fit_reliability: FitReliability,
) -> SimulationPredictiveReport:
    """Reduce both arms without generating, changing masks, or certifying effects."""
    if latent_paths.ndim != 3 or latent_paths.shape[-1] != len(state_ids):
        raise ValueError("Simulation state draws do not match their layout")
    if (
        observations.shape != (*latent_paths.shape[:2], len(variables))
        or mask.shape != observations.shape
        or mask.dtype != np.bool_
    ):
        raise ValueError("Simulation observation draws and masks do not match their layout")
    if (reference_latent_paths is None) != (reference_observations is None):
        raise ValueError("Reference states and observations must be paired")
    if reference_latent_paths is not None and (
        reference_latent_paths.shape != latent_paths.shape
        or reference_observations is None
        or reference_observations.shape != observations.shape
    ):
        raise ValueError("Reference draws must align with the intervention arm")
    state_mask = np.ones(latent_paths.shape[:2], dtype=bool)
    return SimulationPredictiveReport(
        fit_reliability=fit_reliability,
        states={
            identity: SimulationSeriesSummary(
                label=model.get_construct(identity).name,
                action=_summarize(latent_paths[:, :, index], state_mask),
                reference=_summarize(reference_latent_paths[:, :, index], state_mask)
                if reference_latent_paths is not None
                else None,
            )
            for index, identity in enumerate(state_ids)
        },
        indicators={
            variable.id: SimulationSeriesSummary(
                label=variable.name,
                action=_summarize(observations[:, :, index], mask[:, :, index], levels),
                reference=_summarize(reference_observations[:, :, index], mask[:, :, index], levels)
                if reference_observations is not None
                else None,
            )
            for index, variable in enumerate(variables)
            for levels in [
                ("0", "1")
                if variable.measurement_dtype == "binary"
                else variable.ordinal_levels or variable.categorical_levels
            ]
        },
    )
