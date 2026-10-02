"""Typed plot evidence reduced directly from retained native posterior buffers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from itertools import islice
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.identity import ParameterRef
from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
from nof1_causal_lab.artifacts.posterior_diagnostics import (
    DensityHistogram,
    EnergyDiagnostics,
    ParetoKPoint,
    PosteriorMarginal,
    PosteriorPair,
    RankHistogram,
    TraceSeries,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult

type ParameterReferences = Mapping[ParameterCoordinate, tuple[str, ParameterRef] | None]

HIST_PADDING_RATIO = 0.05
HIST_PADDING_DEFAULT = 0.5


def sample_coordinates(
    samples: Mapping[str, jnp.ndarray], *, sample_dims: int
) -> Iterator[tuple[ParameterCoordinate, jnp.ndarray]]:
    for name, values in samples.items():
        for indices in np.ndindex(values.shape[sample_dims:]):
            coordinate = ParameterCoordinate(site_name=name, indices=indices)
            yield coordinate, values[(slice(None),) * sample_dims + indices]


def build_trace_data(
    chain_samples: Mapping[str, jnp.ndarray], references: ParameterReferences
) -> tuple[TraceSeries, ...]:
    """Keep every draw; auxiliary coordinates have an explicit None binding."""
    traces = []
    for coordinate, values in sample_coordinates(chain_samples, sample_dims=2):
        reference = references[coordinate]
        if reference is not None:
            label, subject = reference
            traces.append(
                TraceSeries(
                    parameter=label,
                    subject=subject,
                    chains=tuple(tuple(float(v) for v in chain) for chain in values),
                )
            )
    return tuple(traces)


def build_rank_histograms(
    chain_samples: Mapping[str, jnp.ndarray], references: ParameterReferences, n_bins: int = 20
) -> tuple[RankHistogram, ...]:
    histograms = []
    for coordinate, values in sample_coordinates(chain_samples, sample_dims=2):
        reference = references[coordinate]
        if reference is None:
            continue
        label, subject = reference
        n_chains, n_samples = values.shape
        total = n_chains * n_samples
        ranks = (jnp.argsort(jnp.argsort(values.reshape(-1))) + 1).reshape(values.shape)
        chains = tuple(
            tuple(int(v) for v in jnp.histogram(ranks[chain], bins=n_bins, range=(1, total + 1))[0])
            for chain in range(n_chains)
        )
        histograms.append(
            RankHistogram(
                parameter=label,
                subject=subject,
                n_bins=n_bins,
                expected_per_bin=float(n_samples / n_bins),
                chains=chains,
            )
        )
    return tuple(histograms)


def param_marginal(
    parameter: str, subject: ParameterRef, values: jnp.ndarray, n_bins: int = 50
) -> PosteriorMarginal:
    v_min, v_max = float(jnp.min(values)), float(jnp.max(values))
    padding = (v_max - v_min) * HIST_PADDING_RATIO if v_max > v_min else HIST_PADDING_DEFAULT
    counts, edges = jnp.histogram(values, bins=n_bins, range=(v_min - padding, v_max + padding))
    bin_width = float(edges[1] - edges[0])
    density = counts / (float(jnp.sum(counts)) * bin_width)
    centers = (edges[:-1] + edges[1:]) / 2.0
    sorted_vals = jnp.sort(values)
    n = len(sorted_vals)
    ci_size = int(jnp.ceil(0.94 * n))
    if ci_size < n:
        widths = sorted_vals[ci_size:] - sorted_vals[: n - ci_size]
        best = int(jnp.argmin(widths))
        low, high = float(sorted_vals[best]), float(sorted_vals[best + ci_size])
    else:
        low, high = v_min, v_max
    return PosteriorMarginal(
        parameter=parameter,
        subject=subject,
        x_values=tuple(float(v) for v in centers),
        density=tuple(float(v) for v in density),
        mean=float(jnp.mean(values)),
        sd=float(jnp.std(values)),
        interval_kind="hdi",
        interval_mass=0.94,
        lower=low,
        upper=high,
    )


def build_energy_diagnostics(energy: jnp.ndarray, n_bins: int = 40) -> EnergyDiagnostics:
    e_flat = energy.reshape(-1)
    if energy.ndim == 2:
        de = jnp.diff(energy, axis=1)
        de_flat = de.reshape(-1)
        bfmi = tuple(
            float(jnp.var(de[c]) / jnp.var(energy[c])) if float(jnp.var(energy[c])) > 0 else 0.0
            for c in range(energy.shape[0])
        )
    else:
        de_flat = jnp.diff(e_flat)
        var = float(jnp.var(e_flat))
        bfmi = (float(jnp.var(de_flat) / var) if var > 0 else 0.0,)

    def _hist(values: jnp.ndarray) -> DensityHistogram:
        lo, hi = float(jnp.min(values)), float(jnp.max(values))
        pad = (hi - lo) * HIST_PADDING_RATIO if hi > lo else HIST_PADDING_DEFAULT
        counts, edges = jnp.histogram(values, bins=n_bins, range=(lo - pad, hi + pad))
        bw, total = float(edges[1] - edges[0]), float(jnp.sum(counts))
        density = counts / (total * bw) if total > 0 else counts
        centers = (edges[:-1] + edges[1:]) / 2.0
        return DensityHistogram(
            bin_centers=tuple(float(v) for v in centers), density=tuple(float(v) for v in density)
        )

    return EnergyDiagnostics(
        energy_hist=_hist(e_flat), energy_transition_hist=_hist(de_flat), bfmi=bfmi
    )


def compute_posterior_marginals(
    samples: Mapping[str, jnp.ndarray], references: ParameterReferences, n_bins: int = 50
) -> tuple[PosteriorMarginal, ...]:
    return tuple(
        param_marginal(reference[0], reference[1], values, n_bins)
        for coordinate, values in sample_coordinates(samples, sample_dims=1)
        if (reference := references[coordinate]) is not None
    )


def compute_posterior_pairs(
    samples: Mapping[str, jnp.ndarray],
    references: ParameterReferences,
    mcmc: TrajectoryMCMCResult,
    max_params: int = 6,
) -> tuple[PosteriorPair, ...]:
    """Select axes, retaining all original joint draws and their divergence flags."""
    scalars = list(
        islice(
            (
                (reference, values)
                for coordinate, values in sample_coordinates(samples, sample_dims=1)
                if (reference := references[coordinate]) is not None
            ),
            max_params,
        )
    )
    extra = mcmc.get_extra_fields()
    divergent = (
        tuple(bool(v) for v in extra["diverging"].reshape(-1)) if "diverging" in extra else None
    )
    return tuple(
        PosteriorPair(
            param_x=left[0],
            subject_x=left[1],
            param_y=right[0],
            subject_y=right[1],
            x_values=tuple(float(v) for v in x),
            y_values=tuple(float(v) for v in y),
            divergent=divergent,
        )
        for i, (left, x) in enumerate(scalars)
        for right, y in scalars[i + 1 :]
    )


def pareto_k_points(values: Sequence[float], timesteps: Sequence[int]) -> tuple[ParetoKPoint, ...]:
    """PSIS influence classes and rank retain each original observation row."""
    return tuple(
        ParetoKPoint(
            rank=rank + 1,
            timestep=timesteps[index],
            k=value
            if np.isfinite(value)
            else "undefined"
            if np.isnan(value)
            else "infinity"
            if value > 0
            else "-infinity",
            status="not_evaluated"
            if np.isnan(value)
            else "failed"
            if value > 0.7
            else "warning"
            if value > 0.5
            else "passed",
        )
        for rank, (index, value) in enumerate(
            sorted(
                enumerate(values),
                key=lambda item: (not np.isnan(item[1]), item[1] if not np.isnan(item[1]) else 0),
                reverse=True,
            )
        )
    )
