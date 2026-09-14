"""Visualization and JSON-serialization helpers for inference diagnostics.

Functions for converting inference results (posterior samples, MCMC diagnostics,
LOO-CV, energy) into JSON-serializable dicts for the web frontend.

Extracted from inference.py to separate visualization concerns from inference logic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator  # noqa: TC003
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
from nof1_causal_lab.json_types import JsonObject  # noqa: TC001

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult

logger = logging.getLogger(__name__)

# Histogram binning edge padding to avoid artifacts at distribution tails
HIST_PADDING_RATIO = 0.05
HIST_PADDING_DEFAULT = 0.5


def sample_coordinates(
    samples: dict[str, jnp.ndarray], *, sample_dims: int
) -> Iterator[tuple[ParameterCoordinate, jnp.ndarray]]:
    """Iterate every scalar site element while preserving its original array coordinates."""
    for name, values in samples.items():
        for indices in np.ndindex(values.shape[sample_dims:]):
            coordinate = ParameterCoordinate(site_name=name, indices=indices)
            yield coordinate, values[(slice(None),) * sample_dims + indices]


def build_trace_data(
    chain_samples: dict[str, jnp.ndarray], max_points: int = 200
) -> list[JsonObject]:
    """Build thinned scalar traces with their runtime coordinates."""
    traces: list[JsonObject] = []
    for coordinate, values in sample_coordinates(chain_samples, sample_dims=2):
        step = max(1, values.shape[1] // max_points)
        thinned = values[:, ::step]
        traces.append(
            {
                "parameter": coordinate.label,
                "coordinate": coordinate.model_dump(mode="json"),
                "chains": [
                    {"chain": chain, "values": [float(v) for v in thinned[chain]]}
                    for chain in range(values.shape[0])
                ],
            }
        )
    return traces


def build_rank_histograms(
    chain_samples: dict[str, jnp.ndarray], n_bins: int = 20
) -> list[JsonObject]:
    """Build rank histograms for each scalar coordinate, including array-valued sites."""
    histograms: list[JsonObject] = []
    for coordinate, values in sample_coordinates(chain_samples, sample_dims=2):
        n_chains, n_samples = values.shape
        total = n_chains * n_samples
        ranks = (jnp.argsort(jnp.argsort(values.reshape(-1))) + 1).reshape(values.shape)
        chains: list[JsonObject] = []
        for chain in range(n_chains):
            hist, _ = jnp.histogram(ranks[chain], bins=n_bins, range=(1, total + 1))
            chains.append({"chain": chain, "counts": [int(v) for v in hist]})
        histograms.append(
            {
                "parameter": coordinate.label,
                "coordinate": coordinate.model_dump(mode="json"),
                "n_bins": n_bins,
                "expected_per_bin": float(n_samples / n_bins),
                "chains": list(chains),
            }
        )
    return histograms


def param_marginal(
    coordinate: ParameterCoordinate, values: jnp.ndarray, n_bins: int = 50
) -> JsonObject:
    """Compute histogram-based marginal density for a scalar parameter.

    Returns:
        {parameter, coordinate, x_values, density, mean, sd, lower, upper, interval_kind, interval_mass}
    """
    v_min, v_max = float(jnp.min(values)), float(jnp.max(values))
    padding = (v_max - v_min) * HIST_PADDING_RATIO if v_max > v_min else HIST_PADDING_DEFAULT
    counts, edges = jnp.histogram(values, bins=n_bins, range=(v_min - padding, v_max + padding))
    bin_width = float(edges[1] - edges[0])
    density = counts / (float(jnp.sum(counts)) * bin_width)
    x_centers = (edges[:-1] + edges[1:]) / 2.0

    # HDI (highest density interval) at 94%
    sorted_vals = jnp.sort(values)
    n = len(sorted_vals)
    ci_size = int(jnp.ceil(0.94 * n))
    if ci_size < n:
        widths = sorted_vals[ci_size:] - sorted_vals[: n - ci_size]
        best = int(jnp.argmin(widths))
        hdi_lo = float(sorted_vals[best])
        hdi_hi = float(sorted_vals[best + ci_size])
    else:
        hdi_lo, hdi_hi = v_min, v_max

    return {
        "parameter": coordinate.label,
        "coordinate": coordinate.model_dump(mode="json"),
        "x_values": [float(v) for v in x_centers],
        "density": [float(v) for v in density],
        "mean": float(jnp.mean(values)),
        "sd": float(jnp.std(values)),
        "interval_kind": "hdi",
        "interval_mass": 0.94,
        "lower": hdi_lo,
        "upper": hdi_hi,
    }


def build_energy_diagnostics(energy: jnp.ndarray, n_bins: int = 40) -> JsonObject:
    """Build Hamiltonian energy diagnostics (Betancourt 2017).

    Computes marginal energy (E) and energy transition (dE) histograms.
    """
    e_flat = energy.reshape(-1)

    if energy.ndim == 2:
        de_per_chain = jnp.diff(energy, axis=1)
        de_flat = de_per_chain.reshape(-1)
        bfmi = [
            float(jnp.var(de_per_chain[c]) / jnp.var(energy[c]))
            if float(jnp.var(energy[c])) > 0
            else 0.0
            for c in range(energy.shape[0])
        ]
    else:
        de_flat = jnp.diff(e_flat)
        var_e = float(jnp.var(e_flat))
        bfmi = [float(jnp.var(de_flat) / var_e) if var_e > 0 else 0.0]

    def _hist(vals: jnp.ndarray) -> JsonObject:
        lo, hi = float(jnp.min(vals)), float(jnp.max(vals))
        pad = (hi - lo) * HIST_PADDING_RATIO if hi > lo else HIST_PADDING_DEFAULT
        counts, edges = jnp.histogram(vals, bins=n_bins, range=(lo - pad, hi + pad))
        bw = float(edges[1] - edges[0])
        total = float(jnp.sum(counts))
        density = counts / (total * bw) if total > 0 else counts
        centers = (edges[:-1] + edges[1:]) / 2.0
        return {
            "bin_centers": [float(v) for v in centers],
            "density": [float(v) for v in density],
        }

    return {
        "energy_hist": _hist(e_flat),
        "energy_transition_hist": _hist(de_flat),
        "bfmi": list(bfmi),
    }


def compute_posterior_marginals(
    samples: dict[str, jnp.ndarray], n_bins: int = 50
) -> list[JsonObject]:
    """Compute marginal posterior density data for visualization."""
    return [
        param_marginal(coordinate, values, n_bins)
        for coordinate, values in sample_coordinates(samples, sample_dims=1)
    ]


def compute_posterior_pairs(
    samples: dict[str, jnp.ndarray],
    mcmc: TrajectoryMCMCResult | None,
    max_params: int = 6,
    max_samples: int = 200,
) -> list[JsonObject]:
    """Compute pairwise scatter data for joint posterior visualization."""
    from itertools import islice

    scalars = list(islice(sample_coordinates(samples, sample_dims=1), max_params))
    n_draws = scalars[0][1].shape[0] if scalars else 0
    step = max(1, n_draws // max_samples)

    div_mask: list[bool] | None = None
    if mcmc is not None:
        try:
            extra = mcmc.get_extra_fields()
            if "diverging" in extra:
                div_flat = extra["diverging"].reshape(-1)
                div_mask = [bool(v) for v in div_flat[::step]]
        except (AttributeError, ValueError, RuntimeError):
            logger.warning(
                "Divergence mask extraction failed; pair plots will not show divergent transitions",
                exc_info=True,
            )

    pairs: list[JsonObject] = []
    for i in range(len(scalars)):
        for j in range(i + 1, len(scalars)):
            coordinate_x, vals_x = scalars[i]
            coordinate_y, vals_y = scalars[j]
            entry: JsonObject = {
                "param_x": coordinate_x.label,
                "coordinate_x": coordinate_x.model_dump(mode="json"),
                "param_y": coordinate_y.label,
                "coordinate_y": coordinate_y.model_dump(mode="json"),
                "x_values": [float(v) for v in vals_x[::step]],
                "y_values": [float(v) for v in vals_y[::step]],
            }
            if div_mask is not None and any(div_mask):
                entry["divergent"] = list(div_mask)
            pairs.append(entry)

    return pairs
