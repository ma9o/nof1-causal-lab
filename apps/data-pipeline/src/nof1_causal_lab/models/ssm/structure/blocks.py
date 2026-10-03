"""Block-spec abstraction for every param-bearing concern of an SSM.

Each ``*BlockSpec`` is the canonical declarative representation of one
SSM concern (process-noise Cholesky, manifest mean, initial-state
covariance, …). ``ModelSpec`` stores these blocks as its only
param-bearing fields: there are no flat-field duplicates.

Each block is a frozen dataclass with:

- Its structural data (free supports + templates) — direct fields
- An ``iter_sites()`` method declaring names, shapes, supports, and prior bindings
  interpreted by the shared NumPyro site sampler
- Assembly delegated to the shared algorithms in ``structure.assembly``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor
from nof1_causal_lab.utils.immutability import freeze_fields

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import jax.numpy as jnp
    import numpy as np
    import numpyro.distributions as dist

    PriorFn = Callable[[str], dist.Distribution]


# Block specs share the position extractors in ``structure.assembly``.


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

    def __post_init__(self) -> None:
        freeze_fields(self)

    @property
    def diffusion_diag_positions(self) -> tuple[int, ...]:
        from nof1_causal_lab.models.ssm.structure.assembly import chol_diag_positions

        return tuple(chol_diag_positions(self.diffusion_chol_support, self.n_latent))

    @property
    def diffusion_lower_positions(self) -> tuple[tuple[int, int], ...]:
        from nof1_causal_lab.models.ssm.structure.assembly import strict_lower_positions

        return tuple(strict_lower_positions(self.diffusion_chol_support, self.n_latent))

    @property
    def n_diffusion_diag(self) -> int:
        return len(self.diffusion_diag_positions)

    @property
    def n_diffusion_lower(self) -> int:
        return len(self.diffusion_lower_positions)

    def iter_sites(self) -> Iterator[SiteDescriptor]:
        if self.n_diffusion_diag > 0:
            yield SiteDescriptor(
                name="diffusion_diag_free",
                shape=(self.n_diffusion_diag,),
                support=SupportClass.POSITIVE,
                assembly_group="diffusion",
                site_kind=SiteKind.DIFFUSION_DIAG,
                positions=tuple(self.diffusion_diag_positions),
                prior_field="diffusion_diag",
            )
        if self.n_diffusion_lower > 0:
            yield SiteDescriptor(
                name="diffusion_lower_free",
                shape=(self.n_diffusion_lower,),
                support=SupportClass.REAL,
                assembly_group="diffusion",
                site_kind=SiteKind.DIFFUSION_LOWER,
                positions=tuple(self.diffusion_lower_positions),
                prior_field="diffusion_offdiag",
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
# Sparse substitution for vector and matrix parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class SparseBlockSpec[Position: int | tuple[int, int]]:
    """Element-wise substitution at compiler-owned immutable coordinates."""

    free_support: np.ndarray
    template: jnp.ndarray
    free_positions: tuple[Position, ...]
    free_site_name: str
    support: SupportClass
    site_kind: SiteKind
    assembly_group: str
    prior_field: str

    def __post_init__(self) -> None:
        freeze_fields(self)

    @property
    def n_free(self) -> int:
        return len(self.free_positions)

    def iter_sites(self) -> Iterator[SiteDescriptor]:
        if self.n_free > 0:
            yield SiteDescriptor(
                name=self.free_site_name,
                shape=(self.n_free,),
                support=self.support,
                assembly_group=self.assembly_group,
                site_kind=self.site_kind,
                positions=self.free_positions,
                prior_field=self.prior_field,
            )

    def assemble(self, free: jnp.ndarray | None = None) -> jnp.ndarray:
        from nof1_causal_lab.models.ssm.structure.assembly import assemble_sparse

        return assemble_sparse(
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

    def __post_init__(self) -> None:
        freeze_fields(self)

    @property
    def free_positions(self) -> tuple[int, ...]:
        from nof1_causal_lab.models.ssm.structure.assembly import dense_vector_positions

        return tuple(dense_vector_positions(self.diag_support, self.n_manifest))

    @property
    def n_free(self) -> int:
        return len(self.free_positions)

    def iter_sites(self) -> Iterator[SiteDescriptor]:
        if self.n_free > 0:
            yield SiteDescriptor(
                name="manifest_var_diag_free",
                shape=(self.n_free,),
                support=SupportClass.POSITIVE,
                assembly_group="manifest",
                site_kind=SiteKind.MANIFEST_VAR_DIAG,
                positions=tuple(self.free_positions),
                prior_field="manifest_var_diag",
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

    def __post_init__(self) -> None:
        freeze_fields(self)

    @property
    def diag_positions(self) -> tuple[int, ...]:
        from nof1_causal_lab.models.ssm.structure.assembly import dense_vector_positions

        return tuple(dense_vector_positions(self.diag_support, self.n_latent))

    @property
    def correlation_positions(self) -> tuple[tuple[int, int], ...]:
        from nof1_causal_lab.models.ssm.structure.assembly import strict_lower_positions

        return tuple(strict_lower_positions(self.correlation_support, self.n_latent))

    @property
    def n_diag_free(self) -> int:
        return len(self.diag_positions)

    @property
    def n_correlation_free(self) -> int:
        return len(self.correlation_positions)

    def iter_sites(self) -> Iterator[SiteDescriptor]:
        if self.n_diag_free > 0:
            yield SiteDescriptor(
                name="t0_var_diag_free",
                shape=(self.n_diag_free,),
                support=SupportClass.POSITIVE,
                assembly_group="t0",
                site_kind=SiteKind.T0_VAR_DIAG,
                positions=tuple(self.diag_positions),
                prior_field="t0_var_diag",
            )
        if self.n_correlation_free > 0:
            yield SiteDescriptor(
                name="t0_var_lower_free",
                shape=(self.n_correlation_free,),
                support=SupportClass.CORRELATION,
                assembly_group="t0",
                site_kind=SiteKind.T0_VAR_LOWER,
                positions=tuple(self.correlation_positions),
                prior_field="t0_var_offdiag",
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
