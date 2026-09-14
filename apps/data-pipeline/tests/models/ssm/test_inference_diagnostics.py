"""Tests for enriched MCMC diagnostics extraction (trace data, rank histograms, ESS-tail, LOO)."""

from typing import Any

import jax.numpy as jnp
import jax.random as random
import pytest
from pydantic import ValidationError

from nof1_causal_lab.artifacts.parameter import ParameterCoordinate
from nof1_causal_lab.artifacts.posterior_diagnostics import (
    PosteriorEstimate,
)
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    build_energy_diagnostics as _build_energy_diagnostics,
)
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    build_rank_histograms as _build_rank_histograms,
)
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    build_trace_data as _build_trace_data,
)
from nof1_causal_lab.models.ssm.inference.diagnostics_viz import (
    param_marginal as _param_marginal,
)


class TestBuildTraceData:
    def test_trace_data_shape(self):
        chain_samples = {
            "a": jnp.ones((2, 100)),
            "b": jnp.ones((2, 100, 3)),
        }
        traces: Any = _build_trace_data(chain_samples, max_points=50)
        # "a" is scalar -> 1 trace, "b" is 3-dim -> 3 traces
        assert len(traces) == 4
        assert traces[0]["parameter"] == "a"
        assert len(traces[0]["chains"]) == 2
        assert len(traces[0]["chains"][0]["values"]) == 50

    def test_trace_data_thinning(self):
        chain_samples = {"x": jnp.ones((2, 1000))}
        traces: Any = _build_trace_data(chain_samples, max_points=100)
        assert len(traces[0]["chains"][0]["values"]) == 100


class TestBuildRankHistograms:
    def test_rank_histogram_structure(self):
        key = random.PRNGKey(42)
        chain_samples = {"x": random.normal(key, (2, 200))}
        hists: Any = _build_rank_histograms(chain_samples, n_bins=10)
        assert len(hists) == 1
        assert hists[0]["parameter"] == "x"
        assert hists[0]["n_bins"] == 10
        assert hists[0]["expected_per_bin"] == 20.0  # 200 / 10
        assert len(hists[0]["chains"]) == 2
        assert len(hists[0]["chains"][0]["counts"]) == 10
        # Total counts should equal n_samples
        assert sum(hists[0]["chains"][0]["counts"]) == 200

    def test_preserves_matrix_coordinates(self):
        chain_samples = {"matrix": jnp.ones((2, 100, 3, 3))}
        hists: Any = _build_rank_histograms(chain_samples)
        assert len(hists) == 9
        assert hists[5]["coordinate"] == ParameterCoordinate(
            site_name="matrix", indices=(1, 2)
        ).model_dump(mode="json")
        assert sum(hists[5]["chains"][0]["counts"]) == 100


class TestParamMarginal:
    def test_marginal_structure(self):
        values = random.normal(random.PRNGKey(0), (500,))
        m: Any = _param_marginal(
            ParameterCoordinate(site_name="test", indices=()), values, n_bins=30
        )
        assert m["parameter"] == "test"
        assert len(m["x_values"]) == 30
        assert len(m["density"]) == 30
        assert m["lower"] < m["mean"] < m["upper"]
        assert m["interval_kind"] == "hdi"
        assert m["interval_mass"] == 0.94
        # Density should be non-negative
        assert all(d >= 0 for d in m["density"])


class TestBuildEnergyDiagnostics:
    def test_energy_shape_2d(self):
        energy = jnp.ones((2, 200))
        result: Any = _build_energy_diagnostics(energy, n_bins=20)
        assert len(result["bfmi"]) == 2
        assert len(result["energy_hist"]["bin_centers"]) == 20
        assert len(result["energy_hist"]["density"]) == 20
        assert len(result["energy_transition_hist"]["bin_centers"]) == 20

    def test_energy_shape_1d(self):
        energy = jnp.ones(400)
        result: Any = _build_energy_diagnostics(energy, n_bins=15)
        assert len(result["bfmi"]) == 1
        assert len(result["energy_hist"]["bin_centers"]) == 15

    def test_bfmi_reasonable(self):
        # Random energy => BFMI should be a positive finite number
        key = random.PRNGKey(99)
        energy = random.normal(key, (2, 500))
        result: Any = _build_energy_diagnostics(energy)
        for b in result["bfmi"]:
            assert b > 0
            assert b < 10  # sanity bound


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
