"""Diagnostic reductions checked against small, deterministic reference values."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts.identity import ParameterElementId, ParameterId, ParameterRef
from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
from nof1_causal_lab.artifacts.posterior_diagnostics import PosteriorEstimate
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    build_energy_diagnostics,
    build_rank_histograms,
    build_trace_data,
    param_marginal,
)
from nof1_causal_lab.models.ssm.inference.types import ProductionDiagnostics


def references(coordinates):
    return {
        ParameterCoordinate(site_name=name, indices=indices): (
            ParameterCoordinate(site_name=name, indices=indices).label,
            ParameterRef(
                parameter_id=ParameterId("parameter:" + str(index).zfill(64)),
                element_id=ParameterElementId("element:" + str(index).zfill(64)),
            ),
        )
        for index, (name, indices) in enumerate(coordinates)
    }


@pytest.mark.contract
def test_traces_preserve_all_draws_chains_and_scientific_subjects():
    scalar = np.arange(16).reshape(2, 8)
    matrix = np.arange(64).reshape(2, 8, 2, 2)
    expected = [("scalar", (), scalar)] + [
        ("matrix", ij, matrix[:, :, *ij]) for ij in np.ndindex(2, 2)
    ]
    refs = references([(name, indices) for name, indices, _ in expected])
    traces = build_trace_data({"scalar": jnp.asarray(scalar), "matrix": jnp.asarray(matrix)}, refs)
    assert len(traces) == len(expected)
    for trace, (name, indices, values) in zip(traces, expected, strict=True):
        label, subject = refs[ParameterCoordinate(site_name=name, indices=indices)]
        assert trace.parameter == label
        assert trace.subject == subject
        assert trace.chains == tuple(tuple(row) for row in values.tolist())


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("values", "counts"),
    [
        ([[0, 2, 4, 6], [1, 3, 5, 7]], [[1, 1, 1, 1], [1, 1, 1, 1]]),
        ([[0, 1, 2, 3], [4, 5, 6, 7]], [[2, 2, 0, 0], [0, 0, 2, 2]]),
    ],
    ids=["interleaved", "separated"],
)
def test_rank_histograms_use_pooled_ranks_for_each_coordinate(values, counts):
    scalar = jnp.asarray(values)
    matrix = scalar[:, :, None, None] + jnp.asarray([[0, 10], [20, 30]])
    coords = [("scalar", ())] + [("matrix", ij) for ij in np.ndindex(2, 2)]
    refs = references(coords)
    histograms = build_rank_histograms({"scalar": scalar, "matrix": matrix}, refs, n_bins=4)
    coordinates = [("scalar", ())] + [("matrix", ij) for ij in np.ndindex(2, 2)]
    assert len(histograms) == len(coordinates)
    for histogram, (name, ij) in zip(histograms, coordinates, strict=True):
        coordinate = ParameterCoordinate(site_name=name, indices=ij)
        assert histogram.parameter == coordinate.label
        assert histogram.subject == refs[coordinate][1]
        assert histogram.n_bins == 4
        assert histogram.expected_per_bin == 1.0
        assert histogram.chains == tuple(tuple(row) for row in counts)


@pytest.mark.inference(concern="sampling")
def test_marginal_normalizes_density_and_finds_shortest_interval():
    # The shortest 94% interval excludes the isolated upper-tail value.
    values = np.concatenate([np.arange(39), [100]])
    coordinate = ParameterCoordinate(site_name="matrix", indices=(1, 0))
    marginal = param_marginal(
        coordinate.label,
        references([("matrix", (1, 0))])[coordinate][1],
        jnp.asarray(values),
        n_bins=8,
    )
    assert marginal.subject == references([("matrix", (1, 0))])[coordinate][1]
    assert marginal.mean == pytest.approx(float(values.mean()))
    assert marginal.sd == pytest.approx(float(values.std()))
    assert (marginal.lower, marginal.upper) == (0.0, 38.0)
    assert marginal.interval_kind == "hdi"
    assert marginal.interval_mass == 0.94
    assert len(marginal.x_values) == len(marginal.density) == 8
    assert all(density >= 0 for density in marginal.density)
    bin_width = marginal.x_values[1] - marginal.x_values[0]
    assert sum(marginal.density) * bin_width == pytest.approx(1.0)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("energy", "bfmi"),
    [
        ([[0, 1, 0, 1], [10, 12, 10, 12]], [32 / 9, 32 / 9]),
        ([0, 1, 0, 1], [32 / 9]),
        ([[1, 1, 1, 1], [1, 1, 1, 1]], [0.0, 0.0]),
    ],
    ids=["separate-chains", "single-chain", "constant"],
)
def test_energy_diagnostics_preserve_chain_boundaries_and_normalize_histograms(energy, bfmi):
    result = build_energy_diagnostics(jnp.asarray(energy, dtype=float), n_bins=4)
    np.testing.assert_allclose(result.bfmi, bfmi, rtol=1e-6)
    for key in ("energy_hist", "energy_transition_hist"):
        histogram = getattr(result, key)
        assert len(histogram.bin_centers) == len(histogram.density) == 4
        assert all(density >= 0 for density in histogram.density)
        bin_width = histogram.bin_centers[1] - histogram.bin_centers[0]
        assert sum(histogram.density) * bin_width == pytest.approx(1.0)


@pytest.mark.contract
@pytest.mark.parametrize(
    "invalid",
    [{"lower": 3.0}, {"interval_mass": 0.0}, {"interval_mass": 1.0}, {"mean": float("nan")}],
)
def test_posterior_estimate_rejects_invalid_intervals(invalid):
    payload = {
        "mean": 1.0,
        "lower": 0.0,
        "upper": 2.0,
        "interval_kind": "hdi",
        "interval_mass": 0.94,
    }
    with pytest.raises(ValidationError):
        PosteriorEstimate.model_validate({**payload, **invalid})


@pytest.fixture
def particle_posterior():
    """Fixed input draws for report integration; no inference or shared mutable results."""
    from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
    from nof1_causal_lab.models.ssm.inference.types import (
        JointPosteriorDraws,
        ParticleMCMCPosterior,
    )

    rng = np.random.default_rng(1)
    samples = {
        "alpha": jnp.asarray(1.0 + 0.15 * rng.normal(size=(2, 64))),
        "beta": jnp.asarray([2.5, -0.2]) + jnp.asarray(0.08 * rng.normal(size=(2, 64, 2))),
        "sigma": jnp.asarray(0.5 + 0.04 * rng.normal(size=(2, 64))),
    }
    divergences = np.zeros((2, 64), dtype=bool)
    divergences[0, 4] = True
    mcmc = TrajectoryMCMCResult(
        chain_samples=samples,
        chain_extra_fields={
            "diverging": jnp.asarray(divergences),
            "num_steps": jnp.full((2, 64), 4),
            "accept_prob": jnp.full((2, 64), 0.84),
            "energy": jnp.asarray(4.0 + rng.normal(size=(2, 64))),
        },
        num_chains=2,
        num_samples=64,
        backend="marginal_particle_gibbs",
    )
    return ParticleMCMCPosterior(
        draws=JointPosteriorDraws(parameters=mcmc.get_samples()),
        diagnostics=ProductionDiagnostics(
            mcmc=mcmc, observation_log_probs=jnp.zeros((mcmc.num_chains, mcmc.num_samples, 0))
        ),
    )


@pytest.mark.inference(concern="sampling")
def test_mcmc_report_preserves_coordinate_metrics_chains_and_sampler_statistics(particle_posterior):
    coordinates = [("alpha", ()), ("beta", (0,)), ("beta", (1,)), ("sigma", ())]
    refs = references(coordinates)
    report = particle_posterior.get_mcmc_diagnostics(refs)
    traces, ranks = particle_posterior.get_chain_detail(refs)
    assert report.num_chains == 2
    assert report.num_samples == 64
    assert report.num_divergences == 1
    assert report.divergence_rate == pytest.approx(1 / 128)
    assert report.tree_depth_mean == report.tree_depth_max == 4
    assert report.accept_prob_mean == pytest.approx(0.84)
    assert len(report.energy.bfmi) == 2
    assert all(np.isfinite(value) and value > 0 for value in report.energy.bfmi)
    assert len(report.per_parameter) == len(traces) == len(ranks) == 4
    for metric, trace, hist, (name, indices) in zip(
        report.per_parameter, traces, ranks, coordinates, strict=True
    ):
        coordinate = ParameterCoordinate(site_name=name, indices=indices)
        for entry in (metric, trace, hist):
            assert entry.parameter == coordinate.label
            assert entry.subject == refs[coordinate][1]
        for key in ("r_hat", "ess_bulk", "ess_tail", "mcse_mean"):
            assert np.isfinite(getattr(metric, key))
            assert getattr(metric, key) > 0
        expected = np.asarray(
            particle_posterior.get_samples()[name][(slice(None), *indices)]
        ).reshape(2, 64)
        assert trace.chains == tuple(tuple(row) for row in expected.tolist())
        assert hist.expected_per_bin == 64 / hist.n_bins
        assert len(hist.chains) == 2
        assert all(
            len(row) == hist.n_bins and min(row) >= 0 and sum(row) == 64 for row in hist.chains
        )


@pytest.mark.inference(concern="sampling")
def test_posterior_plots_preserve_all_joint_draws_and_divergences(particle_posterior):
    coords = [("alpha", ()), ("beta", (0,)), ("beta", (1,)), ("sigma", ())]
    refs = references(coords)
    marginals = particle_posterior.get_posterior_marginals(refs, n_bins=8)
    assert len(marginals) == 4
    by_subject = {subject.element_id: coordinate for coordinate, (_, subject) in refs.items()}
    for marginal in marginals:
        coordinate = by_subject[marginal.subject.element_id]
        values = particle_posterior.get_samples()[coordinate.site_name][
            (slice(None), *coordinate.indices)
        ]
        assert marginal.mean == pytest.approx(float(jnp.mean(values)))
        assert marginal.lower < marginal.mean < marginal.upper
        assert len(marginal.x_values) == len(marginal.density) == 8
        assert min(marginal.density) >= 0
    pairs = particle_posterior.get_posterior_pairs(refs, max_params=3)
    assert len(pairs) == 3
    expected_divergences = [False] * 128
    expected_divergences[4] = True
    for pair in pairs:
        assert pair.divergent == tuple(expected_divergences)
        for axis in ("x", "y"):
            coordinate = by_subject[getattr(pair, "subject_" + axis).element_id]
            values = particle_posterior.get_samples()[coordinate.site_name][
                (slice(None), *coordinate.indices)
            ]
            np.testing.assert_array_equal(getattr(pair, axis + "_values"), values)


@pytest.mark.inference(concern="sampling")
def test_loo_report_accepts_joint_particle_emission_factors(particle_posterior):
    import numpyro.distributions as dist

    x = jnp.linspace(-2, 2, 6)
    observations = 1.0 + 2.5 * x + jnp.asarray(np.random.default_rng(0).normal(size=6) * 0.5)
    samples = particle_posterior.diagnostics.mcmc.get_samples(group_by_chain=True)
    mean = samples["alpha"][..., None] + samples["beta"][..., 0, None] * x
    factors = dist.Normal(mean, samples["sigma"][..., None]).log_prob(observations)
    particle_posterior = replace(
        particle_posterior,
        diagnostics=replace(particle_posterior.diagnostics, observation_log_probs=factors),
    )

    estimate, points = particle_posterior.get_loo_diagnostics(observations=observations[:, None])
    assert estimate.observation_unit == "measurement_row"
    assert estimate.prediction_task == "interpolation_given_other_measurements"
    assert estimate.n_data_points == len(points) == 6
    assert np.isfinite(estimate.elpd_loo)
    assert all(isinstance(point.k, float) and np.isfinite(point.k) for point in points)
    assert sorted(point.timestep for point in points) == list(range(1, 7))
