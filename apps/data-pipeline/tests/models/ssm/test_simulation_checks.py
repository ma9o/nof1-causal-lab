"""Shared exact simulation reductions, without construct admission."""

from __future__ import annotations

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from pathlib import Path

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.likelihood import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.execution.observation_distributions import mean_observation_variance
from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
from nof1_causal_lab.models.ssm.simulation_checks import (
    ConstructSimulationTarget,
    DesignInfo,
    measure_construct_simulation,
)
from tests.dynamics_fixtures import hill_term, potential_term
from tests.helpers import (
    fixture_entity_id,
)
from tests.model_fixtures import default_t0_means_block

pytestmark = pytest.mark.inference(concern="predictive")


@pytest.mark.parametrize(
    ("family", "means", "scale", "extra", "expected"),
    [
        ("delta", [0.0, 2.0], 0.0, {}, [0.0, 0.0]),
        ("gaussian", [0.0, 2.0], 0.5, {}, [0.25, 0.25]),
        (
            "student_t",
            [0.0, 1.0, 2.0],
            1.0,
            {"obs_df": [5.0, 1.5, 0.5]},
            [5.0 / 3.0, np.inf, np.nan],
        ),
        ("poisson", [0.0, 1e-12, 2.0], 1.0, {}, [0.0, 1e-12, 2.0]),
        ("gamma", [1.0, 2.0], 1.0, {"obs_shape": 1e-10}, [1e10, 4e10]),
        ("bernoulli", [0.0, 1e-12, 0.5, 1.0], 1.0, {}, [0.0, 1e-12, 0.25, 0.0]),
        ("negative_binomial", [0.0, 1.0, 2.0], 1.0, {"obs_r": 1e-10}, [0.0, 1e10, 4e10]),
        ("beta", [1e-12, 0.5], 1.0, {"obs_concentration": 9.0}, [1e-13, 0.025]),
    ],
)
def test_conditional_variance_uses_observation_family_moments(
    family, means, scale, extra, expected
):
    actual = mean_observation_variance(
        DistributionFamily(family),
        jnp.asarray(means),
        scale,
        {name: jnp.asarray(value) for name, value in extra.items()},
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0.0, equal_nan=True)


def test_time_invariant_construct_omits_temporal_transmission_check():
    draws = 200
    times = 20
    static_values = np.linspace(-1.5, 1.5, draws)
    latent = np.broadcast_to(static_values[:, None, None], (draws, times, 1))
    expected = latent.copy()
    pred = PredictiveDraws(
        parameters={"manifest_cov": jnp.broadcast_to(jnp.array([[[0.25]]]), (draws, 1, 1))},
        likelihood_parameters={},
        trajectory=PredictiveTrajectory(
            latents=jnp.asarray(latent),
            linear_predictors=jnp.asarray(expected),
            observations=jnp.asarray(expected),
            expected_observations=jnp.asarray(expected),
            observations_mask=jnp.ones(expected.shape, dtype=bool),
        ),
    )
    spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / 'simulation_checks/time_invariant_construct_omits_temporal_transmission_check_model_fixture.json').read_text())
    obs_idx = np.arange(times)
    design = DesignInfo(
        manifest_ids=(fixture_entity_id("indicator", "static_indicator"),),
        t_grid=jnp.arange(times, dtype=float),
        obs_index_by_indicator={fixture_entity_id("indicator", "static_indicator"): obs_idx},
        values_by_indicator={
            fixture_entity_id("indicator", "static_indicator"): np.linspace(-1.0, 1.0, times)
        },
    )
    target = ConstructSimulationTarget(
        construct=spec.get_construct(numeric.state_ids(spec)[0]),
    )

    results, _timings = measure_construct_simulation(spec, pred, design, target)
    checks = {result.check for result in results}
    assert {"C5a location reach", "C5b width"} <= checks
    assert (
        next(result for result in results if result.check == "C5c transmission").reason
        == "STATIC_CONSTRUCT"
    )


@pytest.mark.parametrize(('dynamics', 'measurement', 'model_fixture_payload'), [
    pytest.param(True, True, 'simulation_checks/fixed_hill_coefficients_participate_in_checks_and_edge_off_model_fixture_true-true.json', id='True-True'),
    pytest.param(True, False, 'simulation_checks/fixed_hill_coefficients_participate_in_checks_and_edge_off_model_fixture_true-false.json', id='True-False'),
    pytest.param(False, True, 'simulation_checks/fixed_hill_coefficients_participate_in_checks_and_edge_off_model_fixture_false-true.json', id='False-True'),
    pytest.param(False, False, 'simulation_checks/fixed_hill_coefficients_participate_in_checks_and_edge_off_model_fixture_false-false.json', id='False-False'),
])
def test_fixed_hill_coefficients_participate_in_checks_and_edge_off(
    monkeypatch, dynamics, measurement
, model_fixture_payload):
    """Fixed coefficients still affect the Hill checks and the exact edge-off contrast."""
    from nof1_causal_lab.artifacts.expressions import LiteralExpression, hill_applications
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from nof1_causal_lab.models.ssm.predictive import registry_runtime

    draws, ticks = 4, 21
    spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[2] / "fixtures/models" / model_fixture_payload).read_text())
    latents = np.broadcast_to(np.linspace(0.2, 2, ticks)[None, :, None], (draws, ticks, 2)).copy()
    predictive = PredictiveDraws(
        parameters={"manifest_cov": jnp.broadcast_to(jnp.eye(2) * 0.25, (draws, 2, 2))},
        likelihood_parameters={},
        trajectory=PredictiveTrajectory(
            latents=jnp.asarray(latents),
            linear_predictors=jnp.asarray(latents),
            observations=jnp.asarray(latents),
            expected_observations=jnp.asarray(latents),
            observations_mask=jnp.ones(latents.shape, dtype=bool),
        ),
    )
    captured = []
    original = numeric.dynamics_expressions(spec)[2]

    def exact_resimulation(intervened_spec, samples, _times, **_kwargs):
        component = _kwargs["dynamics"].components[2]
        assert component.expression == LiteralExpression(value=0)
        assert samples is predictive.parameters
        for name, values in predictive.parameters.items():
            np.testing.assert_array_equal(samples[name], jnp.asarray(values))
        captured.append(intervened_spec)
        return jnp.asarray(latents * 0.8), jnp.asarray(latents[:, :, 1:])

    monkeypatch.setattr(
        registry_runtime, "_simulate_vector_field_predictive_latents", exact_resimulation
    )
    indicator_id = fixture_entity_id("indicator", "y1")
    design = DesignInfo(
        manifest_ids=(fixture_entity_id("indicator", "x1"), indicator_id),
        t_grid=jnp.arange(ticks, dtype=float),
        obs_index_by_indicator={
            identity: np.arange(ticks) for identity in numeric.observation_ids(spec)
        },
        values_by_indicator={
            identity: np.linspace(0.2, 2, ticks) for identity in numeric.observation_ids(spec)
        },
        n_draws=draws,
    )
    results, _ = measure_construct_simulation(
        spec,
        predictive,
        design,
        ConstructSimulationTarget(
            construct=spec.get_construct(numeric.state_ids(spec)[1]),
            edge_parents=("X",),
            hill_parents=("X",),
        ),
        dynamics=dynamics,
        measurement=measurement,
    )
    assert len(captured) == int(dynamics)
    assert numeric.dynamics_expressions(spec)[2] == original
    assert len(tuple(hill_applications(original.expression))) == 1
    checks = {result.check: result for result in results}
    assert ("C5c transmission" in checks) == measurement
    assert ("C3 resolvability" in checks) == dynamics
    if dynamics:
        assert checks["C3 resolvability"].passed
        assert checks["C4c saturation"].passed
        evidence = checks["C4c saturation"].evidence
        assert evidence is not None
        np.testing.assert_array_equal(evidence["hill_n"], np.full(draws, 2))
