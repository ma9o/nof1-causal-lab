"""Fixed model parameters do not declare sample sites."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest

from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.models.ssm import numerics as numeric
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

pytestmark = pytest.mark.contract

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def _all_fixed_spec(n_latent: int = 2) -> ModelSpec:
    n_manifest = 2
    return model_fixture(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dense_matrix_dynamics_spec(
            n_latent=n_latent,
            decay_support=np.zeros(n_latent, dtype=bool),
            edge_support=np.zeros((n_latent, n_latent), dtype=bool),
            coupling_template=jnp.zeros((n_latent, n_latent)),
            intercept_support=np.zeros(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent),
        ),
        diffusion_block=DiffusionBlockSpec(
            n_latent=n_latent,
            diffusion_chol_support=np.zeros((n_latent, n_latent), dtype=bool),
            diffusion_chol_template=jnp.eye(n_latent),
        ),
        lambda_block=SparseMatrixBlockSpec(
            n_rows=n_manifest,
            n_cols=n_latent,
            free_support=np.zeros((n_manifest, n_latent), dtype=bool),
            template=jnp.eye(n_manifest, n_latent),
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
            template=jnp.zeros(n_manifest),
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
            diag_support=np.zeros(n_manifest, dtype=bool),
            template=jnp.eye(n_manifest),
        ),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            free_support=np.zeros(n_latent, dtype=bool),
            template=jnp.zeros(n_latent),
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
            diag_support=np.zeros(n_latent, dtype=bool),
            correlation_support=np.zeros((n_latent, n_latent), dtype=bool),
            template=jnp.eye(n_latent),
        ),
        static_state_sd_block=default_static_state_sd_block(),
    )


def test_all_fixed_spec_yields_no_sites():
    spec = _all_fixed_spec()
    assert list(numeric.iter_sample_sites(spec)) == []
