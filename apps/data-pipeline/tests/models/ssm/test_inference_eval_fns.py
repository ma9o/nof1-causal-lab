"""Focused behavioral matrix for shared inference evaluators."""

import functools
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
from dynestyx.inference.particle_runtime import Parameterization
from numpyro import handlers

import nof1_causal_lab.models.ssm.inference.utils as inference_utils
from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm import SSMModel
from nof1_causal_lab.models.ssm.autoreparam import AutoReparam
from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.inference.utils import _build_eval_fns, prepare_model_parameters
from nof1_causal_lab.models.ssm.structure import SparseVectorBlockSpec
from tests.model_fixtures import dense_matrix_dynamics_spec, diagonal_diffusion_block, model_fixture


class _RecordingBackend:
    checkpoint_loglik = False

    def __init__(self, lnc: jnp.ndarray) -> None:
        self.lnc = lnc
        self.calls: list[dict[str, Any]] = []

    def _evaluate(self, *args: Any, with_aux: bool, **kwargs: Any) -> Any:
        self.calls.append({"args": args, "kwargs": kwargs, "with_aux": with_aux})
        if with_aux:
            return self.lnc, {"latent_state": jnp.asarray([7.0])}
        return self.lnc

    def compute_log_likelihood(self, *args: Any, **kwargs: Any) -> jnp.ndarray:
        return self._evaluate(*args, with_aux=False, **kwargs)

    def compute_log_likelihood_with_aux(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        return self._evaluate(*args, with_aux=True, **kwargs)


def _build_test_evaluators(monkeypatch, *, runtime: bool, backend: _RecordingBackend):
    registry = object()
    assembled_samples: list[dict[str, jnp.ndarray]] = []
    bound_observations = jnp.asarray([[1.0], [2.0], [3.0]])
    bound_times = jnp.asarray([0.0, 0.5, 1.5])
    model = SimpleNamespace(
        spec=object(),
    )

    monkeypatch.setattr(inference_utils, "build_site_registry", lambda _spec: registry)

    def assemble(samples, spec, *, registry: object):
        assert spec is model.spec
        assert registry is not None
        assembled_samples.append(samples)
        return "dynamics", "measurement", "initial", {"obs_df": 5.0}

    monkeypatch.setattr(inference_utils, "assemble_likelihood_inputs", assemble)
    parameters = Parameterization(
        initial_position=jnp.asarray(0.0),
        unravel=lambda z: {"theta": z},
        constrain=lambda z: {"theta": 11.0 + 2.0 * z},
        log_prior=lambda z: -(z**2),
    )
    functions = inference_utils._build_eval_fns(
        model,
        bound_observations,
        bound_times,
        parameters,
        backend,
        include_likelihood_aux=True,
        runtime_observations_times=runtime,
    )
    return functions, assembled_samples, bound_observations, bound_times


@pytest.mark.parametrize("runtime", [False, True])
def test_eval_fns_share_preparation_and_backend_semantics(monkeypatch, runtime: bool) -> None:
    backend = _RecordingBackend(jnp.asarray([1.0, 2.0, 4.0]))
    (log_lik, _log_prior, log_lik_with_aux), assembled, bound_obs, bound_times = (
        _build_test_evaluators(monkeypatch, runtime=runtime, backend=backend)
    )
    z = jnp.asarray(2.0)
    if runtime:
        observations = jnp.asarray([[8.0], [9.0]])
        times = jnp.asarray([2.0, 2.25])
        value = log_lik(z, observations, times)
        value_with_aux, aux = log_lik_with_aux(
            z,
            observations,
            times,
            latent_mode_init=jnp.asarray([3.0]),
        )
    else:
        observations = bound_obs
        times = bound_times
        value = log_lik(z)
        value_with_aux, aux = log_lik_with_aux(z, latent_mode_init=jnp.asarray([3.0]))

    assert float(value) == 4.0
    assert float(value_with_aux) == 4.0
    np.testing.assert_array_equal(aux["latent_state"], [7.0])
    assert len(assembled) == 2
    np.testing.assert_allclose(assembled[0]["theta"], 15.0)

    first_call, aux_call = backend.calls
    assert first_call["with_aux"] is False
    assert aux_call["with_aux"] is True
    assert "latent_mode_init" not in first_call["kwargs"]
    np.testing.assert_array_equal(aux_call["kwargs"]["latent_mode_init"], [3.0])
    np.testing.assert_array_equal(first_call["args"][3], observations)
    np.testing.assert_allclose(
        first_call["args"][4],
        [MIN_DT, *np.diff(np.asarray(times))],
    )
    assert first_call["kwargs"]["extra_params"] == {"obs_df": 5.0}


@pytest.mark.parametrize(
    ("lnc", "expected"),
    [(jnp.asarray(2.5), 2.5), (jnp.asarray([0.0, jnp.nan]), -jnp.inf)],
)
def test_eval_fns_normalize_scalar_and_nonfinite_results(monkeypatch, lnc, expected) -> None:
    backend = _RecordingBackend(lnc)
    (log_lik, _log_prior, _log_lik_with_aux), *_ = _build_test_evaluators(
        monkeypatch,
        runtime=False,
        backend=backend,
    )

    result = log_lik(jnp.asarray(0.0))

    if jnp.isneginf(expected):
        assert jnp.isneginf(result)
    else:
        assert float(result) == expected


def test_aux_evaluator_is_not_checkpointed(monkeypatch) -> None:
    checkpointed: list[Any] = []
    monkeypatch.setattr(
        inference_utils.jax,
        "checkpoint",
        lambda fn: checkpointed.append(fn) or fn,
    )
    backend = _RecordingBackend(jnp.asarray(1.0))
    backend.checkpoint_loglik = True

    _build_test_evaluators(monkeypatch, runtime=False, backend=backend)

    assert len(checkpointed) == 1


class _InputFingerprintBackend:
    """Exercise real assembly/reparameterization without running a numerical solver."""

    checkpoint_loglik = False

    def compute_log_likelihood(
        self, dynamics, measurement, initial, observations, intervals, *, extra_params
    ):
        inputs = (
            dynamics,
            measurement,
            initial.mean,
            initial.covariance_matrix,
            observations,
            intervals,
            extra_params,
        )
        arrays = jax.tree.leaves(eqx.filter(inputs, eqx.is_array))
        return sum((i + 1) * jnp.sum(value**2) for i, value in enumerate(arrays))


def _apply_reparam(model_fn, reparam_config):
    if reparam_config is None:
        return model_fn
    return handlers.reparam(model_fn, config=reparam_config)


def _eval_model(model_fn, params_dict, observations, times):
    with handlers.seed(rng_seed=0), handlers.substitute(data=params_dict):
        trace = handlers.trace(model_fn).get_trace(observations, times)

    log_lik = 0.0
    log_prior = 0.0
    for name, site in trace.items():
        if site["type"] != "sample":
            continue
        if name == "log_likelihood":
            log_lik = site["fn"].log_factor
        elif not site.get("is_observed", False):
            log_prior = log_prior + jnp.sum(site["fn"].log_prob(site["value"]))
    return log_lik, log_prior


class TestPureJaxLikelihoodEvaluator:
    """The pure-JAX likelihood path should match NumPyro replay exactly."""

    @staticmethod
    def _build_poisson_case():
        spec = model_fixture(
            n_latent=1,
            n_manifest=1,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=1,
                decay_support=np.ones(1, dtype=bool),
                edge_support=np.zeros((1, 1), dtype=bool),
                coupling_template=jnp.zeros((1, 1), dtype=jnp.float32),
                intercept_support=np.zeros(1, dtype=bool),
                cint_template=jnp.zeros(1, dtype=jnp.float32),
            ),
            diffusion_block=diagonal_diffusion_block(1),
            manifest_means_block=SparseVectorBlockSpec(
                n=1,
                free_support=np.zeros(1, dtype=bool),
                template=jnp.array([jnp.log(4.0)], dtype=jnp.float32),
                free_site_name="manifest_means_free",
                det_site_name="manifest_means",
                support=SupportClass.REAL,
                site_kind=SiteKind.MANIFEST_MEANS,
                assembly_group="manifest",
                fixed_spec_field="manifest_means",
                priors_field="manifest_means",
            ),
            manifest_dists=[DistributionFamily.POISSON],
            manifest_links=[LinkFunction.LOG],
        )
        model = SSMModel(spec)
        observations = jnp.array([[4.0], [3.0], [5.0], [6.0]], dtype=jnp.float32)
        times = jnp.arange(observations.shape[0], dtype=jnp.float32) * 0.5
        return model, observations, times

    @staticmethod
    def _assert_log_likelihood_match(reparam) -> None:
        model, observations, times = TestPureJaxLikelihoodEvaluator._build_poisson_case()
        backend = _InputFingerprintBackend()
        parameters, site_info, _ = prepare_model_parameters(
            model, observations, times, random.PRNGKey(0), reparam
        )
        z0, unravel_fn = parameters.initial_position, parameters.unravel
        log_lik_fn, _ = _build_eval_fns(
            model,
            observations,
            times,
            parameters,
            likelihood_backend=backend,
        )

        base_model_fn = functools.partial(model.model, likelihood_backend=backend)
        replay_model_fn = _apply_reparam(base_model_fn, reparam)
        constrained = {
            name: site_info[name]["transform"](unravel_fn(z0)[name]) for name in site_info
        }
        replay_ll, _ = _eval_model(replay_model_fn, constrained, observations, times)

        np.testing.assert_allclose(
            np.asarray(log_lik_fn(z0)),
            np.asarray(replay_ll),
            rtol=1e-6,
            atol=1e-6,
        )

        def _replay_likelihood(z):
            constrained = {
                name: site_info[name]["transform"](unravel_fn(z)[name]) for name in site_info
            }
            return _eval_model(replay_model_fn, constrained, observations, times)[0]

        gradients = jax.grad(log_lik_fn)(z0)
        np.testing.assert_allclose(
            gradients, jax.grad(_replay_likelihood)(z0), rtol=1e-5, atol=1e-5
        )
        assert np.linalg.norm(gradients) > 0

    def test_log_likelihood_matches_model_replay_without_reparam(self):
        self._assert_log_likelihood_match(reparam=None)

    def test_log_likelihood_matches_model_replay_with_fixed_autoreparam(self):
        self._assert_log_likelihood_match(reparam=AutoReparam(centered=0.0))
