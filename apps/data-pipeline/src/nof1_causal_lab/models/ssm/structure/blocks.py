"""Block-spec abstraction for every param-bearing concern of an SSM.

Each ``*BlockSpec`` is the canonical declarative representation of one
SSM concern (process-noise Cholesky, manifest mean, initial-state
covariance, …). ``SSMSpec`` stores these blocks as its only
param-bearing fields: there are no flat-field duplicates.

Each block is a frozen dataclass with:

- Its structural data (free supports + templates) — direct fields
- An ``iter_sites()`` method declaring names, shapes, supports, and prior bindings
  interpreted by the shared NumPyro site sampler
- Assembly delegated to ``structure.assembly`` (single algorithmic
  source of truth shared with ``SSMParameterLayout``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.models.ssm.structure.sites import make_site

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import jax.numpy as jnp
    import numpy as np
    import numpyro.distributions as dist

    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor

    PriorFn = Callable[[str], dist.Distribution]


# Position extractors live in ``structure.assembly`` — the single
# canonical implementation also used by ``SSMParameterLayout``. The
# block specs below import them lazily to avoid an eager dependency at
# module-load time.


# ---------------------------------------------------------------------------
# Diffusion block: process-noise Cholesky factor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class DiffusionBlockSpec:
    """Process-noise Cholesky factor ``L_Q`` block.

    Free entries on the lower-Cholesky support are sampled from the canonical
    site-prior registry. Time-invariant latents have their diagonal forced
    toward a tiny epsilon so their diffusion is effectively zero.
    """

    n_latent: int
    diffusion_chol_support: np.ndarray
    diffusion_chol_template: jnp.ndarray
    time_invariant_mask: np.ndarray | None = None

    @property
    def diffusion_diag_positions(self) -> list[int]:
        from nof1_causal_lab.models.ssm.structure.assembly import chol_diag_positions

        return chol_diag_positions(self.diffusion_chol_support, self.n_latent)

    @property
    def diffusion_lower_positions(self) -> list[tuple[int, int]]:
        from nof1_causal_lab.models.ssm.structure.assembly import strict_lower_positions

        return strict_lower_positions(self.diffusion_chol_support, self.n_latent)

    @property
    def n_diffusion_diag(self) -> int:
        return len(self.diffusion_diag_positions)

    @property
    def n_diffusion_lower(self) -> int:
        return len(self.diffusion_lower_positions)

    def iter_sites(self) -> Iterator[SiteDescriptor]:
        if self.n_diffusion_diag > 0:
            yield make_site(
                "diffusion_diag_free",
                (self.n_diffusion_diag,),
                SupportClass.POSITIVE,
                "diffusion",
                SiteKind.DIFFUSION_DIAG,
                positions=tuple(self.diffusion_diag_positions),
                deterministic_name="diffusion",
                fixed_spec_field="diffusion_chol",
                priors_field="diffusion_diag",
            )
        if self.n_diffusion_lower > 0:
            yield make_site(
                "diffusion_lower_free",
                (self.n_diffusion_lower,),
                SupportClass.REAL,
                "diffusion",
                SiteKind.DIFFUSION_LOWER,
                positions=tuple(self.diffusion_lower_positions),
                deterministic_name="diffusion",
                fixed_spec_field="diffusion_chol",
                priors_field="diffusion_offdiag",
            )

    def assemble(
        self,
        diag_free: jnp.ndarray | None = None,
        lower_free: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        from nof1_causal_lab.models.ssm.structure.assembly import assemble_diffusion_chol

        return assemble_diffusion_chol(
            diffusion_chol_template=self.diffusion_chol_template,
            diag_positions=self.diffusion_diag_positions,
            lower_positions=self.diffusion_lower_positions,
            diag_free=diag_free,
            lower_free=lower_free,
            time_invariant_mask=self.time_invariant_mask,
        )


# ---------------------------------------------------------------------------
# Sparse-vector block: a vector-shape parameter (means, intercepts)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SparseVectorBlockSpec:
    """Generic sparse-vector block: free entries on a 1-D support substituted
    into a template. Used for ``t0_means``, ``manifest_means``, and
    any other length-``n`` parameter sampled element-wise.

    The role-identifying fields (``support``, ``site_kind``,
    ``assembly_group``, ``fixed_spec_field``, ``priors_field``)
    let the block declare its own SiteDescriptor without consulting
    an external table.
    """

    n: int
    free_support: np.ndarray
    template: jnp.ndarray
    free_site_name: str
    det_site_name: str
    support: SupportClass
    site_kind: SiteKind
    assembly_group: str
    fixed_spec_field: str
    priors_field: str

    @property
    def free_positions(self) -> list[int]:
        from nof1_causal_lab.models.ssm.structure.assembly import dense_vector_positions

        return dense_vector_positions(self.free_support, self.n)

    @property
    def n_free(self) -> int:
        return len(self.free_positions)

    def iter_sites(self) -> Iterator[SiteDescriptor]:
        if self.n_free > 0:
            yield make_site(
                self.free_site_name,
                (self.n_free,),
                self.support,
                self.assembly_group,
                self.site_kind,
                positions=tuple(self.free_positions),
                deterministic_name=self.det_site_name,
                fixed_spec_field=self.fixed_spec_field,
                priors_field=self.priors_field,
            )

    def assemble(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        from nof1_causal_lab.models.ssm.structure.assembly import assemble_sparse_vector

        return assemble_sparse_vector(
            template=self.template,
            free_positions=self.free_positions,
            free=free,
        )


# ---------------------------------------------------------------------------
# Sparse-matrix block: a rectangular-shape parameter (loadings, input_effect)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SparseMatrixBlockSpec:
    """Generic sparse-matrix block: free entries on a 2-D rectangular support
    substituted into a template. Used for ``lambda_mat`` (loading
    matrix) and ``input_effect``.
    """

    n_rows: int
    n_cols: int
    free_support: np.ndarray
    template: jnp.ndarray
    free_site_name: str
    det_site_name: str
    support: SupportClass
    site_kind: SiteKind
    assembly_group: str
    fixed_spec_field: str
    priors_field: str

    @property
    def free_positions(self) -> list[tuple[int, int]]:
        from nof1_causal_lab.models.ssm.structure.assembly import rect_matrix_positions

        return rect_matrix_positions(self.free_support, self.n_rows, self.n_cols)

    @property
    def n_free(self) -> int:
        return len(self.free_positions)

    def iter_sites(self) -> Iterator[SiteDescriptor]:
        if self.n_free > 0:
            yield make_site(
                self.free_site_name,
                (self.n_free,),
                self.support,
                self.assembly_group,
                self.site_kind,
                positions=tuple(self.free_positions),
                deterministic_name=self.det_site_name,
                fixed_spec_field=self.fixed_spec_field,
                priors_field=self.priors_field,
            )

    def assemble(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        from nof1_causal_lab.models.ssm.structure.assembly import assemble_sparse_matrix

        return assemble_sparse_matrix(
            template=self.template,
            free_positions=self.free_positions,
            free=free,
        )


# ---------------------------------------------------------------------------
# Manifest-Cholesky block: diagonal Cholesky factor for observation noise
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ManifestCholBlockSpec:
    """Manifest-noise Cholesky factor block. Diagonal Cholesky
    (per-channel variance); off-diagonal correlation is not modelled at
    this layer.
    """

    n_manifest: int
    diag_support: np.ndarray
    template: jnp.ndarray

    @property
    def free_positions(self) -> list[int]:
        from nof1_causal_lab.models.ssm.structure.assembly import dense_vector_positions

        return dense_vector_positions(self.diag_support, self.n_manifest)

    @property
    def n_free(self) -> int:
        return len(self.free_positions)

    def iter_sites(self) -> Iterator[SiteDescriptor]:
        if self.n_free > 0:
            yield make_site(
                "manifest_var_diag_free",
                (self.n_free,),
                SupportClass.POSITIVE,
                "manifest",
                SiteKind.MANIFEST_VAR_DIAG,
                positions=tuple(self.free_positions),
                deterministic_name="manifest_cov",
                fixed_spec_field="manifest_chol",
                priors_field="manifest_var_diag",
            )

    def assemble(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        from nof1_causal_lab.models.ssm.structure.assembly import assemble_manifest_chol

        return assemble_manifest_chol(
            template=self.template,
            free_positions=self.free_positions,
            free=free,
        )


# ---------------------------------------------------------------------------
# Initial-state covariance block: diagonal SDs + off-diagonal correlations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class T0CholBlockSpec:
    """Initial-state covariance block.

    The template is a Cholesky factor ``L`` such that ``L L^T`` is the
    base covariance. The factor is internally decomposed into base SDs
    (``sqrt(diag(LL^T))``) and base correlations
    (``LL^T / (std outer std)``). Free entries on ``diag_support`` replace
    base SDs; free entries on ``correlation_support`` (strict lower) replace
    base correlations symmetrically.

    The block assembles the LATENT-only covariance. Any static-factor
    contribution is added by the caller (see
    ``execution.parameters.assemble_model_matrices``).
    """

    n_latent: int
    diag_support: np.ndarray  # (n_latent,) bool
    correlation_support: np.ndarray  # (n_latent, n_latent) strict lower bool
    template: jnp.ndarray  # (n_latent, n_latent) lower-Cholesky factor

    @property
    def diag_positions(self) -> list[int]:
        from nof1_causal_lab.models.ssm.structure.assembly import dense_vector_positions

        return dense_vector_positions(self.diag_support, self.n_latent)

    @property
    def correlation_positions(self) -> list[tuple[int, int]]:
        from nof1_causal_lab.models.ssm.structure.assembly import strict_lower_positions

        return strict_lower_positions(self.correlation_support, self.n_latent)

    @property
    def n_diag_free(self) -> int:
        return len(self.diag_positions)

    @property
    def n_correlation_free(self) -> int:
        return len(self.correlation_positions)

    def iter_sites(self) -> Iterator[SiteDescriptor]:
        if self.n_diag_free > 0:
            yield make_site(
                "t0_var_diag_free",
                (self.n_diag_free,),
                SupportClass.POSITIVE,
                "t0",
                SiteKind.T0_VAR_DIAG,
                positions=tuple(self.diag_positions),
                deterministic_name="t0_cov",
                fixed_spec_field="t0_chol",
                priors_field="t0_var_diag",
            )
        if self.n_correlation_free > 0:
            yield make_site(
                "t0_var_lower_free",
                (self.n_correlation_free,),
                SupportClass.CORRELATION,
                "t0",
                SiteKind.T0_VAR_LOWER,
                positions=tuple(self.correlation_positions),
                deterministic_name="t0_cov",
                fixed_spec_field="t0_chol",
                priors_field="t0_var_offdiag",
            )

    @property
    def base_cov(self) -> jnp.ndarray:
        import jax.numpy as jnp_local

        L = jnp_local.asarray(self.template)
        return L @ L.T

    @property
    def base_std(self) -> jnp.ndarray:
        import jax.numpy as jnp_local

        return jnp_local.sqrt(jnp_local.clip(jnp_local.diag(self.base_cov), min=0.0))

    @property
    def base_corr(self) -> jnp.ndarray:
        import jax.numpy as jnp_local

        std = self.base_std
        cov = self.base_cov
        denom = std[:, None] * std[None, :]
        corr = jnp_local.where(
            denom > 0,
            cov / denom,
            jnp_local.eye(self.n_latent, dtype=cov.dtype),
        )
        corr = 0.5 * (corr + corr.T)
        return corr.at[jnp_local.diag_indices(self.n_latent)].set(1.0)

    def assemble_cov(
        self,
        diag_free: jnp.ndarray | None = None,
        correlation_free: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        import jax.numpy as jnp_local

        std = self.base_std
        if diag_free is not None:
            diag_free = jnp_local.asarray(diag_free, dtype=std.dtype)
            for idx, latent_idx in enumerate(self.diag_positions):
                std = std.at[latent_idx].set(diag_free[idx])
        corr = self.base_corr
        if correlation_free is not None:
            correlation_free = jnp_local.asarray(correlation_free, dtype=corr.dtype)
            for idx, (row, col) in enumerate(self.correlation_positions):
                corr = corr.at[row, col].set(correlation_free[idx])
                corr = corr.at[col, row].set(correlation_free[idx])
        cov = corr * (std[:, None] * std[None, :])
        return 0.5 * (cov + cov.T)
