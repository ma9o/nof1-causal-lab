"""Test-owned ModelSpec construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import dynestyx as dsx
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
from dynestyx.inference.configs.discretizer import ExactAffineConfig

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.models.ssm.autoreparam import Strategy, _minimal_reparam
from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

if TYPE_CHECKING:
    from numpyro.primitives import Message

    from nof1_causal_lab.artifacts.identity import ConstructId


def affine_test_evolution(A, covariance, b=None, B=None):
    """Library-owned exact affine reference, restricted to test data and comparisons."""
    return dsx.discretize_state_evolution(
        dsx.StochasticContinuousTimeStateEvolution(
            drift=dsx.AffineDrift(A=A, b=b, B=B),
            diffusion=dsx.FullDiffusion(jnp.linalg.cholesky(covariance)),
        ),
        ExactAffineConfig(covariance_jitter=0.0),
    )


class MinimalReparam(Strategy):
    """Test-owned minimal reparameterization strategy."""

    @override
    def configure(self, msg: Message):
        return _minimal_reparam(msg["fn"], is_observed=msg.get("is_observed", False))


def make_lgss_data(
    *,
    T: int = 100,
    dt: float = 1.0,
    decay_diag: float = -0.3,
    diff_sd: float = 0.3,
    obs_sd: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Build 1D linear-Gaussian SSM data plus a free-parameter ModelSpec.

    Returns a dict with ``observations``, ``times``, ``spec``, the true
    parameter values, and ``n_latent`` for convenience. Used by recovery
    checks that fit the same canonical 1D model with different inference
    methods.
    """
    n_latent, n_manifest = 1, 1

    true_dynamics = jnp.array([[decay_diag]])
    true_diff_cov = jnp.array([[diff_sd**2]])
    true_obs_var = jnp.array([[obs_sd**2]])

    parameters = affine_test_evolution(true_dynamics, true_diff_cov).params_at(0.0, dt)
    Ad, Qd = parameters.A, parameters.cov
    Qd_chol = jla.cholesky(Qd + jnp.eye(n_latent) * 1e-8, lower=True)
    R_chol = jla.cholesky(true_obs_var, lower=True)

    key = random.PRNGKey(seed)
    states = [jnp.zeros(n_latent)]
    for _ in range(T - 1):
        key, nk = random.split(key)
        states.append(Ad @ states[-1] + Qd_chol @ random.normal(nk, (n_latent,)))
    latent = jnp.stack(states)

    key, obs_key = random.split(key)
    observations = latent + random.normal(obs_key, (T, n_manifest)) @ R_chol.T
    times = jnp.arange(T, dtype=float) * dt

    spec = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[0]
            / "fixtures/models"
            / "model_fixtures/make_lgss_data_model_fixture.json"
        ).read_text()
    )

    return {
        "observations": observations,
        "times": times,
        "spec": spec,
        "true_decay_diag": decay_diag,
        "true_diff_diag": diff_sd,
        "true_obs_sd": obs_sd,
        "n_latent": n_latent,
    }


def make_observation_support_runtime(**kwargs: Any) -> ObservationSupportRuntime:
    """Build ObservationSupportRuntime while accepting 2D interval coefficient inputs."""
    support_kinds = kwargs["support_kinds"]
    kwargs.setdefault(
        "summary_operators",
        ["mean" if kind == "interval" else "last" for kind in support_kinds],
    )
    kwargs.setdefault(
        "anchor_policies",
        [
            "support_start" if operator == "first" else "support_end"
            for operator in kwargs["summary_operators"]
        ],
    )
    prev = np.asarray(kwargs["interval_prev_coeffs"], dtype=np.float64)
    curr = np.asarray(kwargs["interval_curr_coeffs"], dtype=np.float64)
    weights = np.asarray(kwargs["interval_weights"], dtype=np.float64)
    if prev.ndim == 2:
        prev = prev[..., None]
        curr = curr[..., None]
        weights = weights[..., None]
    kwargs["interval_prev_coeffs"] = prev
    kwargs["interval_curr_coeffs"] = curr
    kwargs["interval_weights"] = weights
    emission_slots = kwargs.get("emission_slot_indices")
    if emission_slots is None:
        support_end = np.asarray(kwargs["support_end_times"])
        emission_slots = np.where(np.isfinite(support_end), 0, -1).astype(np.int64)
    kwargs["emission_slot_indices"] = emission_slots
    return ObservationSupportRuntime.assembled(**kwargs)


def parameter_draws(model: ModelSpec, n_draws: int) -> dict[str, jnp.ndarray]:
    """Repeat the authored prior reference point without invoking inference."""
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import compile_priors
    from nof1_causal_lab.prior_distributions import prior_reference_value

    priors, _, _ = compile_priors(compile_model_fixture(model), StructuralSelection(model, None))
    return {
        name: jnp.broadcast_to(value, (n_draws, *value.shape))
        for name, law in priors.items()
        for value in [jnp.asarray(prior_reference_value(law))]
    }


def compile_fit_fixture(spec: ModelSpec, outcome: ConstructId | None = None):
    """Require real compilation in fixtures instead of forging fit evidence."""
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import (
        CompiledFitInputs,
        compile_ssm_inputs_from_model,
    )

    inputs = compile_ssm_inputs_from_model(StructuralSelection(spec, outcome))
    assert isinstance(inputs, CompiledFitInputs), inputs
    return inputs


def compile_model_fixture(spec: ModelSpec, outcome: ConstructId | None = None):
    """Compile native execution facts without imposing the fitting law restrictions."""
    from nof1_causal_lab.models.model_structure import StructuralSelection
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model

    return compile_executable_model(StructuralSelection(spec, outcome))


def bind_panel_fixture(model, observations, times, *, support=None):
    """Publish a complete numerical test panel, including its identity-bearing rows."""
    from datetime import UTC, datetime, timedelta

    import polars as pl

    from nof1_causal_lab.models.ssm.observation_support import simulation_observation_support
    from nof1_causal_lab.models.ssm.runtime import BoundPanel, bind_panel

    observations, times = jnp.asarray(observations), jnp.asarray(times)
    support = (
        simulation_observation_support(model, np.asarray(times)) if support is None else support
    )
    origin = datetime(1970, 1, 1, tzinfo=UTC)
    rows = []
    for i, observation in enumerate(model.observations):
        for t, at in enumerate(np.asarray(times)):
            start, end = support.support_start_times[t, i], support.support_end_times[t, i]
            rows.append(
                {
                    "indicator_id": str(observation.id),
                    "value": float(observations[t, i]),
                    "anchor_time": origin + timedelta(days=float(at)),
                    "support_start": origin + timedelta(days=float(start))
                    if np.isfinite(start)
                    else None,
                    "support_end": origin + timedelta(days=float(end))
                    if np.isfinite(end)
                    else None,
                    "support_kind": support.support_kinds[i],
                    "summary_operator": support.summary_operators[i],
                    "anchor_policy": support.anchor_policies[i],
                    "observation_window": support.observation_windows[i],
                }
            )
    panel = bind_panel(pl.DataFrame(rows), model=model, time_origin=origin)
    assert isinstance(panel, BoundPanel), panel
    return panel
