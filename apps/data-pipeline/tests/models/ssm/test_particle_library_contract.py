"""Cheap checks of the app/library boundary; no fitting or sampler execution."""

import ast
from importlib import import_module
from pathlib import Path
from typing import Never
from unittest.mock import Mock

import jax.numpy as jnp
import numpy as np
import pytest
from dynestyx.inference.particle_runtime import Parameterization, ParticleRuntime

from nof1_causal_lab.artifacts.posterior_diagnostics import LOODiagnostics
from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from nof1_causal_lab.models.ssm.inference.methods._pmcmc_shared.extraction import (
    extract_grouped_public_samples,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.runner import (
    _initialize_chain_state,
)
from nof1_causal_lab.models.ssm.inference.problem import ParticleProblem
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws, ParticleMCMCPosterior
from nof1_causal_lab.models.ssm.inference.utils import extract_constrained_samples


def _no_numerical_execution(*_args) -> Never:
    raise AssertionError("This boundary check must not execute the numerical model")


def test_inference_uses_dynestyx_targets_without_external_sampler_implementations():
    source_root = Path(__file__).resolve().parents[3] / "src/nof1_causal_lab/models/ssm/inference"
    forbidden = (
        "cuthbert",
        "dynestyx.inference.particle_mcmc",
        "dynestyx.inference.particle_chain",
        "dynestyx.inference.particle_profiling",
    )
    violations = []
    for source in source_root.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
            else:
                continue
            if any(
                module == prefix or module.startswith(f"{prefix}.")
                for module in modules
                for prefix in forbidden
            ):
                violations.append(f"{source.relative_to(source_root)}:{node.lineno}")
    assert not violations, f"Inference implementation must stay in the app: {violations}"


def test_library_parameter_output_preserves_public_vector_sites_and_chain_order():
    positions = jnp.arange(12.0).reshape(2, 3, 2)
    parameters = Parameterization(
        initial_position=jnp.zeros(2),
        unravel=_no_numerical_execution,
        constrain=lambda z: {"beta": 2 * z, "beta_decentered": z},
        log_prior=_no_numerical_execution,
    )
    runtime = ParticleRuntime(
        parameters=parameters,
        context=_no_numerical_execution,
        model=_no_numerical_execution,
        schedule=_no_numerical_execution,
        observations=jnp.zeros((4, 1)),
        times=jnp.arange(4.0),
    )
    problem = ParticleProblem(
        runtime=runtime,
        site_info={},
        public_sites={"beta"},
        latent_transition_kind="euler_maruyama",
    )
    samples = extract_grouped_public_samples(positions, bundle=problem, num_chains=2, num_samples=3)
    assert set(samples) == {"beta"}
    np.testing.assert_array_equal(samples["beta"], 2 * positions)
    mcmc = TrajectoryMCMCResult(samples, {}, num_chains=2, num_samples=3)
    np.testing.assert_array_equal(mcmc.get_samples()["beta"], (2 * positions).reshape(6, 2))


def test_supplied_initial_path_does_not_compute_a_predictive_rollout():
    target = Mock(
        spec=ParticleRuntime,
        context=Mock(),
        initial_path=Mock(side_effect=_no_numerical_execution),
        initial_moments=Mock(return_value=(jnp.zeros(1), jnp.eye(1))),
        log_posterior_from_context=Mock(return_value=(jnp.array(-2.0), jnp.array(-1.0))),
    )
    initial_path = jnp.array([[0.1], [0.2], [0.3]])
    state = _initialize_chain_state(
        jnp.zeros(1),
        observations=jnp.zeros((3, 1)),
        times=jnp.arange(3.0),
        target=target,
        initial_latent_delta=jnp.full(3, 0.1),
        param_step_size=0.01,
        param_min_scale=1e-6,
        param_max_scale=1.0,
        param_target_accept=0.35,
        initial_latent_trajectory=initial_path,
    )
    target.initial_path.assert_not_called()
    np.testing.assert_array_equal(state.latent_trajectory, initial_path)
    target.log_posterior_from_context.assert_called_once()


def test_fixed_model_samples_replay_deterministics_with_zero_parameter_dimension():
    parameters = Parameterization(
        initial_position=jnp.zeros(0),
        unravel=_no_numerical_execution,
        constrain=lambda _: {"lambda": jnp.ones((2, 1)), "internal": jnp.array(5.0)},
        log_prior=_no_numerical_execution,
    )
    samples = extract_constrained_samples(jnp.zeros((3, 0)), parameters, {"lambda"})
    assert set(samples) == {"lambda"}
    np.testing.assert_array_equal(samples["lambda"], np.ones((3, 2, 1)))


def test_loo_uses_joint_emissions_and_omits_only_completely_missing_rows(monkeypatch):
    samples = {"beta": jnp.arange(12.0).reshape(2, 3, 2)}
    factors = jnp.arange(24.0).reshape(2, 3, 4) / -10
    observations = jnp.array([[1.0, 2.0], [jnp.nan, jnp.nan], [3.0, jnp.nan], [jnp.nan, 4.0]])
    mcmc = TrajectoryMCMCResult(samples, {}, num_chains=2, num_samples=3)
    posterior = ParticleMCMCPosterior(
        draws=JointPosteriorDraws(parameters=mcmc.get_samples()),
        diagnostics={"mcmc": mcmc, "observation_log_probs": factors},
    )

    def estimate(idata):
        np.testing.assert_array_equal(
            idata.log_likelihood["measurement_row"].values, factors[:, :, [0, 2, 3]]
        )
        np.testing.assert_array_equal(idata.posterior["beta"].values, samples["beta"])
        return Mock(
            elpd=-7.0,
            p=0.4,
            se=0.2,
            n_data_points=3,
            pareto_k=Mock(values=np.array([0.2, 0.8, 0.4])),
        )

    # Test the scientific factor boundary without computing PSIS or fitting.
    monkeypatch.setattr(import_module("arviz_stats.loo"), "loo", estimate)
    result = LOODiagnostics.model_validate(posterior.get_loo_diagnostics(observations=observations))
    assert result.n_data_points == 3
    assert result.n_bad_k == 1
    assert result.observation_unit == "measurement_row"
    assert result.prediction_task == "interpolation_given_other_measurements"
    assert result.likelihood_source == "exact_emission_on_joint_particle_draws"


def test_loo_all_missing_rows_have_no_predictive_estimate():
    posterior = ParticleMCMCPosterior(
        draws=JointPosteriorDraws(parameters={}),
        diagnostics={"observation_log_probs": jnp.zeros((2, 3, 4))},
    )
    assert posterior.get_loo_diagnostics(observations=jnp.full((4, 2), jnp.nan)) is None


def test_loo_requires_particle_evidence():
    posterior = ParticleMCMCPosterior(draws=JointPosteriorDraws(parameters={"beta": jnp.zeros(3)}))
    with pytest.raises(KeyError, match="observation_log_probs"):
        posterior.get_loo_diagnostics(observations=jnp.ones((4, 2)))
