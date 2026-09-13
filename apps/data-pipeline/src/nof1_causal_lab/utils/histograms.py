"""Backend binning for reported sample distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.effects import HistogramBin

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def histogram_draws(draws: ArrayLike, *, max_bins: int = 25) -> list[HistogramBin]:
    """Bin finite scalar samples into equal-width, inclusive-final-edge bins."""
    values = np.asarray(draws)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Histogram draws must be a nonempty finite vector")
    counts, edges = np.histogram(values, bins=min(max_bins, int(np.ceil(np.sqrt(values.size)))))
    return [
        HistogramBin(
            bin_start=float(left),
            bin_end=float(right),
            bin_center=float((left + right) / 2),
            count=int(count),
        )
        for left, right, count in zip(edges[:-1], edges[1:], counts, strict=True)
    ]
