"""Small parameter-contract checks without fitting or forward simulation."""

import hashlib
from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.models.ssm import parameterization
from nof1_causal_lab.models.ssm.model import sample_parameters
from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor
from tests.model_fixtures import compile_fit_fixture


@pytest.mark.inference(concern="predictive")
def test_prior_draws_keep_native_values_and_the_existing_random_streams():
    sites = [
        SiteDescriptor(
            name="tiny",
            shape=(),
            support=SupportClass.POSITIVE,
            assembly_group="diffusion",
            site_kind=SiteKind.DIFFUSION_DIAG,
        ),
        SiteDescriptor(
            name="rho",
            shape=(),
            support=SupportClass.CORRELATION,
            assembly_group="t0",
            site_kind=SiteKind.T0_VAR_LOWER,
        ),
        SiteDescriptor(
            name="normal",
            shape=(2,),
            support=SupportClass.REAL,
            assembly_group="t0",
            site_kind=SiteKind.T0_MEANS,
        ),
    ]
    laws = {
        "tiny": dist.Delta(jnp.array(1e-35)),
        "rho": dist.Delta(jnp.array(1.0 - 1e-7)),
        "normal": dist.Normal(jnp.zeros(2), 1.0),
    }
    key = random.PRNGKey(11)
    samples = parameterization.sample_prior_parameters(key, sites, laws, n_samples=2)
    # Native support values survive without log/exp rounding or correlation clipping.
    np.testing.assert_array_equal(samples["tiny"], jnp.full(2, 1e-35))
    np.testing.assert_array_equal(samples["rho"], jnp.full(2, 1.0 - 1e-7))
    digest = hashlib.sha256(b"normal").digest()
    for index in range(2):
        draw_key = random.fold_in(key, index)
        draw_key = random.fold_in(draw_key, int.from_bytes(digest[:4], "little"))
        draw_key = random.fold_in(draw_key, int.from_bytes(digest[4:8], "little"))
        np.testing.assert_array_equal(samples["normal"][index], laws["normal"].sample(draw_key))


@pytest.mark.inference(concern="sampling")
def test_parameter_trace_preserves_site_order_shapes_and_public_deterministics():
    spec = ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "common/two_state_gaussian_model.json"
        ).read_text()
    )
    model = compile_fit_fixture(spec)
    values = {
        "diffusion_diag_free": jnp.array([0.4, 0.6]),
        "diffusion_lower_free": jnp.array([0.25]),
        "manifest_var_diag_free": jnp.array([0.7, 0.8]),
        "t0_means_free": jnp.array([1.0, -1.0]),
        "t0_var_diag_free": jnp.array([2.0, 3.0]),
        "t0_var_lower_free": jnp.array([0.25]),
    }
    with handlers.substitute(data=values):
        trace = handlers.trace(sample_parameters).get_trace(
            model.compiled, model.prior_runtime_bundle
        )
    sampled_names = [name for name in trace if name in values]
    assert sampled_names == list(values)
    for name, value in values.items():
        np.testing.assert_array_equal(trace[name]["value"], value)
    assert {name for name, site in trace.items() if site["type"] == "deterministic"} == {
        "diffusion",
        "lambda",
        "manifest_means",
        "manifest_cov",
        "t0_means",
        "t0_cov",
    }
    np.testing.assert_allclose(trace["t0_cov"]["value"], [[4.0, 1.5], [1.5, 9.0]], atol=1e-6)
    np.testing.assert_allclose(trace["diffusion"]["value"], [[0.4, 0.0], [0.25, 0.6]])
    np.testing.assert_allclose(trace["manifest_cov"]["value"], np.diag([0.49, 0.64]), atol=1e-6)
