"""Pure read projections of retained simulation evidence."""

from __future__ import annotations

import numpy as np

from nof1_causal_lab.artifacts.simulation import CategoryProbabilitySummary


def category_probabilities(
    values: np.ndarray, mask: np.ndarray, levels: tuple[str, ...]
) -> CategoryProbabilitySummary:
    """Keep the full categorical distribution and per-time effective draw count."""
    present = mask & np.isfinite(values)
    selected = tuple(samples[keep] for samples, keep in zip(values.T, present.T, strict=True))
    return CategoryProbabilitySummary(
        probabilities={
            level: tuple(
                float(np.mean(samples == index)) if samples.size else None for samples in selected
            )
            for index, level in enumerate(levels)
        },
        n_draws=tuple(int(count) for count in present.sum(axis=0)),
    )
