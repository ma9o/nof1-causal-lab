"""Shared parameter sampling and matrix assembly for the scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro

from nof1_causal_lab.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
    stabilize_covariance_for_cholesky,
    symmetrize,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpyro.distributions as dist

    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor


def sample_sites(
    sites: Iterable[SiteDescriptor], prior: Callable[[str], dist.Distribution]
) -> dict[str, jnp.ndarray]:
    """Interpret the declared site order through NumPyro's sampling handlers."""
    return {site.name: jnp.asarray(numpyro.sample(site.name, prior(site.name))) for site in sites}


def assemble_model_matrices(
    spec: SSMSpec, samples: dict[str, jnp.ndarray]
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """Assemble one parameter draw, including the initial covariance constraint.

    Every declared free site is required. Missing entries in ``values`` below
    therefore correspond only to fixed blocks, whose templates own their values.
    The returned minimum eigenvalue defines the NumPyro constraint factor.
    """
    values = {
        site.name: samples[site.name]
        for block in spec.parameter_blocks
        for site in block.iter_sites()
    }
    manifest_chol = spec.manifest_chol_block.assemble(values.get("manifest_var_diag_free"))
    static_sds = spec.static_state_sd_block.assemble(values.get("static_state_sd_free"))
    covariance = spec.t0_chol_block.assemble_cov(
        values.get("t0_var_diag_free"), values.get("t0_var_lower_free")
    )
    if static_sds.size:
        loadings = jnp.asarray(spec.static_factor_loadings)
        covariance = covariance + loadings @ jnp.diag(static_sds**2) @ loadings.T
    initial_covariance, min_eigenvalue = stabilize_covariance_for_cholesky(
        symmetrize(covariance), min_eigenvalue=INITIAL_STATE_COV_MIN_EIGENVALUE
    )
    return {
        "diffusion": spec.diffusion_block.assemble(
            values.get("diffusion_diag_free"), values.get("diffusion_lower_free")
        ),
        "lambda": spec.lambda_block.assemble(values.get("lambda_free")),
        "manifest_means": spec.manifest_means_block.assemble(values.get("manifest_means_free")),
        "manifest_cov": manifest_chol @ manifest_chol.T,
        "t0_means": spec.t0_means_block.assemble(values.get("t0_means_free")),
        "t0_cov": initial_covariance,
        "input_effect": spec.input_effect_block.assemble(values.get("input_effect_free")),
        "static_state_sds": static_sds,
    }, min_eigenvalue
