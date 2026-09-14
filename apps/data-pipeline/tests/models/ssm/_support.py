"""Shared fixtures for SSM contracts, inference, and predictive test suites."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from tests.model_fixtures import (
    default_static_state_sd_block,
    dense_matrix_dynamics_spec,
    model_fixture,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec

# ══════════════════════════════════════════════════════════════════════════════
# AUTOREPARAM
# ══════════════════════════════════════════════════════════════════════════════


def simple_normal_model():
    x = numpyro.sample("x", dist.Normal(0.0, 1.0))
    y = numpyro.sample("y", dist.Normal(x, 0.5))
    numpyro.sample("obs", dist.Normal(y, 0.1), obs=jnp.array(1.0))


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION FAMILY MATRIX
# ══════════════════════════════════════════════════════════════════════════════


def complex_mixed_family_config() -> tuple[list[str], list[str], list[int], list[str]]:
    manifest_dists = [
        "gaussian",
        "bernoulli",
        "poisson",
        "student_t",
        "gamma",
        "beta",
        "ordered_logistic",
        "categorical",
        "negative_binomial",
        "gaussian",
    ]
    manifest_links = [
        LinkFunction.IDENTITY.value,
        LinkFunction.LOGIT.value,
        LinkFunction.LOG.value,
        LinkFunction.IDENTITY.value,
        LinkFunction.LOG.value,
        LinkFunction.LOGIT.value,
        LinkFunction.CUMULATIVE_LOGIT.value,
        LinkFunction.SOFTMAX.value,
        LinkFunction.LOG.value,
        LinkFunction.IDENTITY.value,
    ]
    manifest_level_counts = [0, 0, 0, 0, 0, 0, 4, 4, 0, 0]
    manifest_names = [
        "stress_cont",
        "adherence_flag",
        "steps_count",
        "fatigue_t",
        "screen_gap",
        "sleep_efficiency",
        "symptom_severity",
        "coping_style",
        "rumination_count",
        "focus_cont",
    ]
    return manifest_dists, manifest_links, manifest_level_counts, manifest_names


# ══════════════════════════════════════════════════════════════════════════════
# PRIOR-PREDICTIVE RUNTIME SPEC
# ══════════════════════════════════════════════════════════════════════════════


def complex_mixed_runtime_spec() -> ModelSpec:
    n_latent = 4
    n_manifest = 10
    coupling_template = jnp.array(
        [
            [-0.45, 0.0, 0.0, 0.0],
            [0.08, -0.35, 0.0, 0.0],
            [0.02, 0.06, -0.4, 0.0],
            [0.0, 0.03, 0.05, -0.3],
        ],
        dtype=jnp.float32,
    )
    lambda_template = jnp.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.3, 0.4, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0],
            [0.2, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.5, 0.3],
            [0.0, 0.0, 0.7, 0.0],
            [0.0, 0.2, 0.5, 0.0],
            [0.0, 0.0, 0.0, 0.9],
            [0.1, 0.0, 0.2, 0.8],
        ],
        dtype=jnp.float32,
    )
    manifest_means_template = jnp.array(
        [0.0, -0.3, 0.4, 0.0, -0.2, 0.0, 0.0, 0.1, 0.2, -0.1],
        dtype=jnp.float32,
    )
    manifest_chol_template = jnp.diag(
        jnp.array(
            [0.12, 0.08, 0.1, 0.18, 0.1, 0.05, 0.08, 0.08, 0.11, 0.12],
            dtype=jnp.float32,
        )
        ** 2
    )
    t0_chol_template = jnp.eye(n_latent, dtype=jnp.float32) * 0.25
    diffusion_template = jnp.diag(jnp.array([0.2, 0.18, 0.16, 0.14], dtype=jnp.float32))
    return model_fixture(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dense_matrix_dynamics_spec(
            n_latent=n_latent,
            decay_support=np.zeros(n_latent, dtype=bool),
            edge_support=np.zeros((n_latent, n_latent), dtype=bool),
            coupling_template=coupling_template,
            intercept_support=np.zeros(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent, dtype=jnp.float32),
        ),
        diffusion_block=DiffusionBlockSpec(
            n_latent=n_latent,
            diffusion_chol_support=np.tri(n_latent, dtype=bool),
            diffusion_chol_template=diffusion_template,
        ),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            free_support=np.zeros((n_manifest, n_latent), dtype=bool),
            template=lambda_template,
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        ),
        manifest_means_block=SparseVectorBlockSpec(
            n=n_manifest,
            free_support=np.zeros(n_manifest, dtype=bool),
            template=manifest_means_template,
            free_site_name="manifest_means_free",
            det_site_name="manifest_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.MANIFEST_MEANS,
            assembly_group="manifest",
            fixed_spec_field="manifest_means",
            priors_field="manifest_means",
        ),
        manifest_chol_block=ManifestCholBlockSpec(
            n_manifest=n_manifest,
            diag_support=np.ones(n_manifest, dtype=bool),
            template=manifest_chol_template,
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            free_support=np.ones(n_latent, dtype=bool),
            template=jnp.zeros(n_latent, dtype=jnp.float32),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        ),
        t0_chol_block=T0CholBlockSpec(
            n_latent=n_latent,
            diag_support=np.ones(n_latent, dtype=bool),
            correlation_support=np.tri(n_latent, k=-1, dtype=bool),
            template=t0_chol_template,
        ),
        static_state_sd_block=default_static_state_sd_block(),
        manifest_dists=[
            DistributionFamily.GAUSSIAN,
            DistributionFamily.BERNOULLI,
            DistributionFamily.POISSON,
            DistributionFamily.STUDENT_T,
            DistributionFamily.GAMMA,
            DistributionFamily.BETA,
            DistributionFamily.ORDERED_LOGISTIC,
            DistributionFamily.CATEGORICAL,
            DistributionFamily.NEGATIVE_BINOMIAL,
            DistributionFamily.GAUSSIAN,
        ],
        manifest_links=[
            LinkFunction.IDENTITY,
            LinkFunction.LOGIT,
            LinkFunction.LOG,
            LinkFunction.IDENTITY,
            LinkFunction.LOG,
            LinkFunction.LOGIT,
            LinkFunction.CUMULATIVE_LOGIT,
            LinkFunction.SOFTMAX,
            LinkFunction.LOG,
            LinkFunction.IDENTITY,
        ],
        manifest_level_counts=[0, 0, 0, 0, 0, 0, 4, 4, 0, 0],
        latent_names=["stress", "adherence", "sleep", "focus"],
        manifest_names=[
            "stress_cont",
            "adherence_flag",
            "steps_count",
            "fatigue_t",
            "screen_gap",
            "sleep_efficiency",
            "symptom_severity",
            "coping_style",
            "rumination_count",
            "focus_cont",
        ],
    )
