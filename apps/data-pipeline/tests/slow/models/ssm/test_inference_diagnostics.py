"""Slow diagnostics tests for MCMC-shaped inference results."""

import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.models.ssm.inference import ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws

pytestmark = [pytest.mark.slow, pytest.mark.cpu_expensive]


@pytest.fixture(scope="module")
def mcmc_result():
    draw_key = random.PRNGKey(1)
    k_alpha, k_beta, k_sigma, k_energy = random.split(draw_key, 4)
    chain_samples = {
        "alpha": 1.0 + 0.15 * random.normal(k_alpha, (2, 200)),
        "beta": 2.5 + 0.08 * random.normal(k_beta, (2, 200)),
        "sigma": 0.5 + 0.04 * random.normal(k_sigma, (2, 200)),
    }
    extra_fields = {
        "diverging": jnp.zeros((2, 200), dtype=bool),
        "num_steps": jnp.full((2, 200), 4),
        "accept_prob": jnp.full((2, 200), 0.84),
        "energy": 4.0 + random.normal(k_energy, (2, 200)),
    }
    mcmc = TrajectoryMCMCResult(
        chain_samples=chain_samples,
        chain_extra_fields=extra_fields,
        num_chains=2,
        num_samples=200,
        backend="marginal_particle_gibbs",
    )

    return ParticleMCMCPosterior(
        draws=JointPosteriorDraws(parameters=mcmc.get_samples()),
        diagnostics={"mcmc": mcmc},
    )


class TestMCMCDiagnostics:
    def test_basic_diagnostics(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert diag is not None
        assert "per_parameter" in diag
        assert len(diag["per_parameter"]) == 3
        for p in diag["per_parameter"]:
            assert "parameter" in p
            assert "r_hat" in p
            assert "ess_bulk" in p

    def test_ess_tail_values_positive(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        for p in diag["per_parameter"]:
            assert "ess_tail" in p
            assert p["ess_tail"] > 0

    def test_mcse_values_positive(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        for p in diag["per_parameter"]:
            assert "mcse_mean" in p
            assert p["mcse_mean"] > 0

    def test_trace_data_has_finite_values(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert len(diag["trace_data"]) == 3
        assert {t["parameter"] for t in diag["trace_data"]} == {"alpha", "beta", "sigma"}
        for trace in diag["trace_data"]:
            assert len(trace["chains"]) == 2
            for chain in trace["chains"]:
                assert len(chain) > 0

    def test_rank_histograms_have_valid_bins(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert len(diag["rank_histograms"]) == 3
        assert {h["parameter"] for h in diag["rank_histograms"]} == {"alpha", "beta", "sigma"}
        for hist in diag["rank_histograms"]:
            assert hist["n_bins"] > 0
            for chain_entry in hist["chains"]:
                counts = chain_entry["counts"]
                assert len(counts) == hist["n_bins"]
                assert all(c >= 0 for c in counts)
                assert sum(counts) > 0

    def test_sampler_stats(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert "num_divergences" in diag
        assert "tree_depth_mean" in diag
        assert "accept_prob_mean" in diag
        assert diag["num_chains"] == 2
        assert diag["num_samples"] == 200


class TestLOODiagnostics:
    def test_loo_uses_joint_posterior_emission_factors(self, mcmc_result):
        n_obs = 30
        x = jnp.linspace(-2, 2, n_obs)
        y = 1.0 + 2.5 * x + 0.5 * random.normal(random.PRNGKey(0), (n_obs,))
        samples = mcmc_result.diagnostics["mcmc"].get_samples(group_by_chain=True)
        mean = samples["alpha"][..., None] + samples["beta"][..., None] * x
        factors = dist.Normal(mean, samples["sigma"][..., None]).log_prob(y)
        mcmc_result.diagnostics["observation_log_probs"] = factors
        estimate = mcmc_result.get_loo_diagnostics(observations=y[:, None])
        assert estimate is not None
        assert estimate["observation_unit"] == "measurement_row"
        assert estimate["prediction_task"] == "interpolation_given_other_measurements"
        assert len(estimate["pareto_k"]) == n_obs
        assert estimate["n_bad_k"] == 0

    def test_loo_omits_fully_missing_rows(self, mcmc_result):
        mcmc_result.diagnostics["observation_log_probs"] = jnp.zeros((2, 200, 3))
        assert mcmc_result.get_loo_diagnostics(observations=jnp.full((3, 1), jnp.nan)) is None

    def test_loo_requires_particle_emission_factors(self):
        posterior = ParticleMCMCPosterior(draws=JointPosteriorDraws(parameters={}), diagnostics={})
        with pytest.raises(KeyError, match="observation_log_probs"):
            posterior.get_loo_diagnostics(observations=jnp.ones((3, 1)))


class TestPosteriorMarginals:
    def test_marginals(self, mcmc_result):
        marginals = mcmc_result.get_posterior_marginals()
        assert len(marginals) == 3
        assert {m["parameter"] for m in marginals} == {"alpha", "beta", "sigma"}
        for m in marginals:
            assert len(m["x_values"]) == len(m["density"])
            assert all(d >= 0 for d in m["density"])
            assert m["lower"] < m["mean"] < m["upper"]


class TestPosteriorPairs:
    def test_pairs(self, mcmc_result):
        pairs = mcmc_result.get_posterior_pairs()
        assert len(pairs) == 3
        for p in pairs:
            assert "param_x" in p
            assert "param_y" in p
            assert len(p["x_values"]) == len(p["y_values"])
            assert len(p["x_values"]) <= 200

    def test_divergent_field(self, mcmc_result):
        pairs = mcmc_result.get_posterior_pairs()
        for p in pairs:
            if "divergent" in p:
                assert len(p["divergent"]) == len(p["x_values"])
                assert not any(p["divergent"])


class TestEnergyInMCMCDiagnostics:
    def test_energy_present(self, mcmc_result):
        diag = mcmc_result.get_mcmc_diagnostics()
        assert "energy" in diag
        energy = diag["energy"]
        assert "energy_hist" in energy
        assert "energy_transition_hist" in energy
        assert "bfmi" in energy
        assert len(energy["bfmi"]) == 2
