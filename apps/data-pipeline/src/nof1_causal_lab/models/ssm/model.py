"""NumPyro State-Space Model.

Bayesian State-Space Model definition using NumPyro.
This module defines the probabilistic model only — inference is in inference.py.

Supports:
- Time-series trajectories
- Any noise family (Gaussian, Poisson, Student-t, Gamma)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import MultivariateNormal

from nof1_causal_lab.models.ssm.execution.dynamical_model import continuous_state_evolution
from nof1_causal_lab.models.ssm.execution.parameters import assemble_model_matrices, sample_sites

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx import StochasticContinuousTimeStateEvolution

    from nof1_causal_lab.artifacts.compiled_ssm import CompiledParameterBinding
    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId, ParameterId
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime
    from nof1_causal_lab.models.ssm.structure import (
        DiffusionBlockSpec,
        ManifestCholBlockSpec,
        SparseMatrixBlockSpec,
        SparseVectorBlockSpec,
        T0CholBlockSpec,
    )

from nof1_causal_lab.artifacts.statistical_model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.constants import MIN_DT
from nof1_causal_lab.models.ssm.covariance_utils import (
    INITIAL_STATE_COV_MIN_EIGENVALUE,
)
from nof1_causal_lab.models.ssm.execution.contracts import (
    LikelihoodExtraParams,
    MeasurementParams,
)
from nof1_causal_lab.models.ssm.likelihood_extra_params import (
    assemble_sampled_extra_params,
)
from nof1_causal_lab.models.ssm.parameter_layout import SSMParameterLayout
from nof1_causal_lab.models.ssm.parameterization import (
    PriorRuntimeBundle,
    build_prior_runtime_bundle,
    likelihood_sites,
)


@jax.custom_vjp
def _nan_safe_ll(ll):
    """Return ll if finite, else -1e30. Gradient is zeroed when ll is non-finite."""
    return jnp.where(jnp.isfinite(ll), ll, -1e30)


def _nan_safe_ll_fwd(ll):
    y = _nan_safe_ll(ll)
    return y, jnp.isfinite(ll)


def _nan_safe_ll_bwd(is_finite, g):
    return (jnp.where(is_finite, jnp.nan_to_num(g, nan=0.0), 0.0),)


_nan_safe_ll.defvjp(_nan_safe_ll_fwd, _nan_safe_ll_bwd)


@dataclass
class SSMSpec:
    """Specification for a state-space model.

    Blocks:
    - dynamics_spec: component-owned continuous-time latent vector field
    - diffusion_chol (L_Q): n_latent x n_latent process noise Cholesky factor
    - cint (c): n_latent x 1 continuous intercept
    - lambda_mat (Λ): n_manifest x n_latent factor loadings
    - manifest_means (μ): n_manifest x 1 manifest intercepts
    - manifest_chol (L_R): n_manifest x n_manifest measurement-error Cholesky factor
    - t0_means (η₀): n_latent x 1 initial state means
    - t0_chol (L_0): n_latent x n_latent initial-state Cholesky factor

    Distributions:
    - diffusion_dists (Nₛ): per-latent process noise family
    - manifest_dists (Fᵢ): per-channel observation family
    - manifest_links (gᵢ): per-channel link
    - hᵢ: extra observation parameters required by Fᵢ (for example df, shape, r, concentration, cutpoints, or categorical logits)

    State:       dη(t) = f(t, η(t), θ) dt + L_Q dNₛ(t)
                 η(0) ~ N(t0_means, L_0 L_0ᵀ)

    Linear pred: ξᵢ(t) = (Λ η(t) + μ)ᵢ
    Mean param:  mᵢ(t) = gᵢ⁻¹(ξᵢ(t))
    Emission:    yᵢ(t) ~ Fᵢ(mᵢ(t); hᵢ, (L_R L_Rᵀ)ᵢᵢ)
    """

    # Structural shape metadata (required)
    n_latent: int
    n_manifest: int

    # Canonical block-spec params (required). Each block owns its
    # structural support, template, and per-prior settings; the SSMSpec
    # itself stores no flat-field duplicates. Priors are typically
    # left None at construction time and attached at sample time from
    # the runtime PriorRuntimeBundle.
    dynamics_spec: DynamicsSpec
    diffusion_block: DiffusionBlockSpec
    lambda_block: SparseMatrixBlockSpec
    manifest_means_block: SparseVectorBlockSpec
    manifest_chol_block: ManifestCholBlockSpec
    t0_means_block: SparseVectorBlockSpec
    t0_chol_block: T0CholBlockSpec
    input_effect_block: SparseMatrixBlockSpec
    static_state_sd_block: SparseVectorBlockSpec

    # Pure structural metadata (no sampled params).
    static_factor_loadings: jnp.ndarray = field(
        default_factory=lambda: jnp.zeros((0, 0), dtype=jnp.float32)
    )
    diffusion_dists: list[DistributionFamily] = field(default_factory=list)
    manifest_dists: list[DistributionFamily] = field(default_factory=list)
    manifest_level_counts: list[int] | None = None
    manifest_links: list[LinkFunction] | None = None
    manifest_standardized: list[bool] | None = None
    # Categorical channels acting as their construct's scale/sign anchor: the
    # first non-baseline slope of these channels is pinned to +1 at assembly.
    manifest_cat_anchor: list[bool] | None = None
    latent_ids: list[ConstructId] | None = None
    manifest_ids: list[IndicatorId] | None = None
    input_ids: list[ConstructId] | None = None
    static_factor_ids: list[ParameterId] | None = None
    latent_names: list[str] | None = None
    manifest_names: list[str] | None = None
    input_names: list[str] | None = None
    input_source_indicators: list[str] | None = None
    input_scales: list[float] | None = None
    input_missing_policies: list[str] | None = None
    input_lagged: list[bool] = field(default_factory=list)
    static_factor_names: list[str] | None = None

    def __post_init__(self) -> None:
        """Validate block-spec shape agreement and canonicalize metadata."""

        def _shape_tuple(shape: tuple[int, ...]) -> str:
            return "(" + ", ".join(str(dim) for dim in shape) + ")"

        def _require_shape(name: str, value: Any, shape: tuple[int, ...]) -> None:
            if value is None:
                raise ValueError(f"{name} must have shape {_shape_tuple(shape)}, got None.")
            actual = np.asarray(value).shape
            if actual != shape:
                raise ValueError(f"{name} must have shape {_shape_tuple(shape)}, got {actual}.")

        def _require_vector(name: str, value: Any, n: int) -> None:
            _require_shape(name, value, (n,))

        def _require_matrix(name: str, value: Any, rows: int, cols: int) -> None:
            _require_shape(name, value, (rows, cols))

        # Static factor loadings shape: (n_latent, n_factor)
        loadings = jnp.asarray(self.static_factor_loadings)
        if loadings.ndim != 2:
            raise ValueError("static_factor_loadings must be a rank-2 array.")
        if loadings.shape[0] == 0 and loadings.shape[1] == 0:
            loadings = jnp.zeros((self.n_latent, 0), dtype=jnp.float32)
        elif loadings.shape[0] != self.n_latent:
            raise ValueError(
                "static_factor_loadings must have shape "
                f"({self.n_latent}, n_factor), got {loadings.shape}"
            )
        self.static_factor_loadings = loadings
        n_static_factor = int(loadings.shape[1])

        # Cross-check block shapes against n_latent / n_manifest.
        if self.dynamics_spec.n_latent != self.n_latent:
            raise ValueError(
                f"dynamics_spec.n_latent ({self.dynamics_spec.n_latent}) "
                f"!= SSMSpec.n_latent ({self.n_latent})"
            )
        if self.diffusion_block.n_latent != self.n_latent:
            raise ValueError(
                f"diffusion_block.n_latent ({self.diffusion_block.n_latent}) "
                f"!= SSMSpec.n_latent ({self.n_latent})"
            )
        _require_matrix(
            "diffusion_chol_support",
            self.diffusion_block.diffusion_chol_support,
            self.n_latent,
            self.n_latent,
        )
        _require_matrix(
            "diffusion_chol",
            self.diffusion_block.diffusion_chol_template,
            self.n_latent,
            self.n_latent,
        )
        if self.lambda_block.n_rows != self.n_manifest or self.lambda_block.n_cols != self.n_latent:
            raise ValueError(
                f"lambda_block shape ({self.lambda_block.n_rows}, "
                f"{self.lambda_block.n_cols}) != "
                f"({self.n_manifest}, {self.n_latent})"
            )
        _require_matrix(
            "lambda_support",
            self.lambda_block.free_support,
            self.n_manifest,
            self.n_latent,
        )
        _require_matrix("lambda_mat", self.lambda_block.template, self.n_manifest, self.n_latent)
        if self.manifest_means_block.n != self.n_manifest:
            raise ValueError(
                f"manifest_means_block.n ({self.manifest_means_block.n}) "
                f"!= n_manifest ({self.n_manifest})"
            )
        _require_vector(
            "manifest_means_support",
            self.manifest_means_block.free_support,
            self.n_manifest,
        )
        _require_vector("manifest_means", self.manifest_means_block.template, self.n_manifest)
        if self.manifest_chol_block.n_manifest != self.n_manifest:
            raise ValueError(
                f"manifest_chol_block.n_manifest "
                f"({self.manifest_chol_block.n_manifest}) "
                f"!= n_manifest ({self.n_manifest})"
            )
        _require_vector(
            "manifest_chol_diag_support",
            self.manifest_chol_block.diag_support,
            self.n_manifest,
        )
        _require_matrix(
            "manifest_chol",
            self.manifest_chol_block.template,
            self.n_manifest,
            self.n_manifest,
        )
        if self.t0_means_block.n != self.n_latent:
            raise ValueError(
                f"t0_means_block.n ({self.t0_means_block.n}) != n_latent ({self.n_latent})"
            )
        _require_vector("t0_means_support", self.t0_means_block.free_support, self.n_latent)
        _require_vector("t0_means", self.t0_means_block.template, self.n_latent)
        if self.t0_chol_block.n_latent != self.n_latent:
            raise ValueError(
                f"t0_chol_block.n_latent ({self.t0_chol_block.n_latent}) "
                f"!= n_latent ({self.n_latent})"
            )
        _require_vector("t0_chol_diag_support", self.t0_chol_block.diag_support, self.n_latent)
        _require_matrix(
            "t0_correlation_support",
            self.t0_chol_block.correlation_support,
            self.n_latent,
            self.n_latent,
        )
        _require_matrix(
            "t0_chol",
            self.t0_chol_block.template,
            self.n_latent,
            self.n_latent,
        )
        if self.input_effect_block.n_rows not in {0, self.n_latent}:
            raise ValueError(
                f"input_effect_block.n_rows ({self.input_effect_block.n_rows}) "
                f"!= n_latent ({self.n_latent}) or 0"
            )
        _require_matrix(
            "input_effect_support",
            self.input_effect_block.free_support,
            self.input_effect_block.n_rows,
            self.input_effect_block.n_cols,
        )
        _require_matrix(
            "input_effect",
            self.input_effect_block.template,
            self.input_effect_block.n_rows,
            self.input_effect_block.n_cols,
        )
        if self.static_state_sd_block.n != n_static_factor:
            raise ValueError(
                f"static_state_sd_block.n ({self.static_state_sd_block.n}) "
                f"!= n_static_factor ({n_static_factor})"
            )
        _require_vector(
            "static_state_sd_support",
            self.static_state_sd_block.free_support,
            n_static_factor,
        )
        _require_vector("static_state_sds", self.static_state_sd_block.template, n_static_factor)

        # Resolve n_input from input_effect_block, then canonicalize names.
        n_input = self.input_effect_block.n_cols
        if self.input_names is None:
            self.input_names = [f"input_{idx}" for idx in range(n_input)]
        elif len(self.input_names) != n_input:
            raise ValueError(
                f"input_names length must match n_input: {len(self.input_names)} vs {n_input}"
            )
        if self.input_source_indicators is None:
            self.input_source_indicators = list(self.input_names)
        elif len(self.input_source_indicators) != n_input:
            raise ValueError(
                "input_source_indicators length must match n_input: "
                f"{len(self.input_source_indicators)} vs {n_input}"
            )
        if self.input_scales is None:
            self.input_scales = [1.0] * n_input
        elif len(self.input_scales) != n_input:
            raise ValueError(
                f"input_scales length must match n_input: {len(self.input_scales)} vs {n_input}"
            )
        if any(float(scale) <= 0.0 for scale in self.input_scales):
            raise ValueError("input_scales must be strictly positive")
        if self.input_missing_policies is None:
            self.input_missing_policies = ["zero"] * n_input
        elif len(self.input_missing_policies) != n_input:
            raise ValueError(
                "input_missing_policies length must match n_input: "
                f"{len(self.input_missing_policies)} vs {n_input}"
            )
        invalid_policies = sorted(
            {
                str(policy)
                for policy in self.input_missing_policies
                if policy not in {"zero", "forward_fill"}
            }
        )
        if invalid_policies:
            raise ValueError(f"Unsupported input_missing_policies: {invalid_policies}")
        if len(self.input_lagged) != n_input:
            raise ValueError(
                f"input_lagged length must match n_input: {len(self.input_lagged)} vs {n_input}"
            )
        self.input_lagged = [bool(lagged) for lagged in self.input_lagged]

        # Canonicalize per-channel family + link enums.
        if self.diffusion_dists:
            self.diffusion_dists = [DistributionFamily(d) for d in self.diffusion_dists]
        else:
            self.diffusion_dists = [DistributionFamily.GAUSSIAN] * self.n_latent
        if len(self.diffusion_dists) != self.n_latent:
            raise ValueError(
                "diffusion_dists length must match n_latent: "
                f"{len(self.diffusion_dists)} vs {self.n_latent}"
            )
        if self.manifest_dists:
            self.manifest_dists = [DistributionFamily(d) for d in self.manifest_dists]
        else:
            self.manifest_dists = [DistributionFamily.GAUSSIAN] * self.n_manifest
        if len(self.manifest_dists) != self.n_manifest:
            raise ValueError(
                "manifest_dists length must match n_manifest: "
                f"{len(self.manifest_dists)} vs {self.n_manifest}"
            )
        if self.manifest_links is not None and len(self.manifest_links) != self.n_manifest:
            raise ValueError(
                "manifest_links length must match n_manifest: "
                f"{len(self.manifest_links)} vs {self.n_manifest}"
            )
        from nof1_causal_lab.models.ssm.execution.observation_families import (
            resolve_manifest_families_and_links,
        )

        _, self.manifest_links = resolve_manifest_families_and_links(
            self.manifest_dists,
            manifest_links=self.manifest_links,
        )
        if (
            self.manifest_level_counts is not None
            and len(self.manifest_level_counts) != self.n_manifest
        ):
            raise ValueError(
                "manifest_level_counts length must match n_manifest: "
                f"{len(self.manifest_level_counts)} vs {self.n_manifest}"
            )
        if self.manifest_standardized is None:
            self.manifest_standardized = [False] * self.n_manifest
        elif len(self.manifest_standardized) != self.n_manifest:
            raise ValueError(
                "manifest_standardized length must match n_manifest: "
                f"{len(self.manifest_standardized)} vs {self.n_manifest}"
            )
        if self.manifest_cat_anchor is None:
            self.manifest_cat_anchor = [False] * self.n_manifest
        elif len(self.manifest_cat_anchor) != self.n_manifest:
            raise ValueError(
                "manifest_cat_anchor length must match n_manifest: "
                f"{len(self.manifest_cat_anchor)} vs {self.n_manifest}"
            )
        if self.static_factor_names is None:
            self.static_factor_names = [f"tau_{idx}" for idx in range(n_static_factor)]
        elif len(self.static_factor_names) != n_static_factor:
            raise ValueError(
                "static_factor_names length must match number of static factors: "
                f"{len(self.static_factor_names)} vs {n_static_factor}"
            )

    def iter_sample_sites(self):
        """Flat iteration over every sample-site descriptor on this spec.

        Drift components receive their canonical vector-field component prefix
        so their sample sites match ``compile_dynamics(prefix="vf")``.
        Blocks with no free parameters yield nothing.
        """
        for idx, component in enumerate(self.dynamics_spec.components):
            yield from component.iter_sites(
                prefix=f"vf_{idx}",
                n_latent=self.n_latent,
            )
        for block in self.parameter_blocks:
            yield from block.iter_sites()

    @property
    def parameter_blocks(self):
        """Non-dynamics blocks in their declared NumPyro sampling order."""
        return (
            self.diffusion_block,
            self.lambda_block,
            self.manifest_means_block,
            self.manifest_chol_block,
            self.t0_means_block,
            self.t0_chol_block,
            self.input_effect_block,
            self.static_state_sd_block,
        )


class SSMModel:
    """NumPyro state-space model definition.

    Defines the probabilistic model for Bayesian state-space models.
    Inference is handled externally by ssm.inference.fit().

    Features:
    - Continuous-time dynamics via stochastic differential equations
    - IEKS/Laplace marginal likelihood backend
    """

    def __init__(
        self,
        spec: SSMSpec,
        priors: dict[str, dist.Distribution] | None = None,
        prior_runtime_bundle: PriorRuntimeBundle | None = None,
    ):
        """Initialize state-space model.

        Args:
            spec: Statistical model specification
            priors: Prior registry (uses compiler defaults if None)
        """
        self.spec = spec
        self.priors = priors
        self._parameter_layout = SSMParameterLayout.from_spec(spec)
        self._artifact_cache: dict[tuple[object, ...], object] = {}
        self.observation_support: ObservationSupportRuntime | None = None
        self.transition_inputs: jnp.ndarray | None = None
        self.parameter_bindings: list[CompiledParameterBinding] = []
        self._prior_runtime_bundle = prior_runtime_bundle
        self._prior_site_index = (
            {site.name: site for site in prior_runtime_bundle.registry}
            if prior_runtime_bundle is not None
            else None
        )

    def get_cached_artifact[T](
        self,
        cache_key: tuple[object, ...],
        factory: Callable[[], T],
    ) -> T:
        """Construct an artifact once per model instance and reuse it afterwards."""
        if cache_key not in self._artifact_cache:
            self._artifact_cache[cache_key] = factory()
        return cast("T", self._artifact_cache[cache_key])

    def set_observation_support(
        self, observation_support: ObservationSupportRuntime | None
    ) -> None:
        """Attach prepared observation-support metadata and invalidate backend caches."""
        self.observation_support = observation_support
        self._artifact_cache = {
            key: value
            for key, value in self._artifact_cache.items()
            if not (isinstance(key, tuple) and key and key[0] == "backend")
        }

    def set_transition_inputs(self, transition_inputs: jnp.ndarray | None) -> None:
        """Attach prepared known-input trajectories aligned to transition intervals."""
        self.transition_inputs = transition_inputs

    @property
    def vector_field(self):
        """Unified dynamics representation as a :class:`VectorField`.

        Every spec carries a populated ``dynamics_spec``. The compiled vector
        field is what downstream consumers
        (``compute_steady_state``, ``simulate``, the per-step linearisation in
        the IEKS/Laplace warmup backend, …) all consume uniformly.
        """

        def _build():
            from nof1_causal_lab.models.ssm.dynamics.spec import compile_dynamics

            return compile_dynamics(self.spec.dynamics_spec).vector_field

        return self.get_cached_artifact(("vector_field",), _build)

    @property
    def parameter_layout(self) -> SSMParameterLayout:
        """Return the derived parameter layout for this model."""
        return self._parameter_layout

    def get_prior_runtime_bundle(self) -> PriorRuntimeBundle:
        """Return canonical prior runtime state for this model instance."""
        if self._prior_runtime_bundle is None:
            self._prior_runtime_bundle = build_prior_runtime_bundle(self.spec, self.priors)
            self._prior_site_index = {
                site.name: site for site in self._prior_runtime_bundle.registry
            }
        return self._prior_runtime_bundle

    def _prior_distribution(self, site_name: str) -> dist.Distribution:
        """Resolve a sample-site prior from canonical runtime semantics."""
        runtime = self.get_prior_runtime_bundle()
        assert self._prior_site_index is not None
        site = self._prior_site_index.get(site_name)
        if site is None:
            raise ValueError(f"Prior runtime bundle has no site named {site_name!r}")
        return runtime.priors[site_name]

    def _sample_likelihood_extra_params(self, spec: SSMSpec) -> LikelihoodExtraParams:
        """Sample the shared likelihood-site catalog and assemble its semantics."""
        return assemble_sampled_extra_params(
            spec, sample_sites(likelihood_sites(spec), self._prior_distribution)
        )

    def _sample_parameters(self) -> dict[str, jnp.ndarray]:
        """Sample declared sites and emit the canonical scientific matrices."""
        sites = chain.from_iterable(block.iter_sites() for block in self.spec.parameter_blocks)
        matrices, min_eigenvalue = assemble_model_matrices(
            self.spec, sample_sites(sites, self._prior_distribution)
        )
        for name, value in matrices.items():
            # Empty input/static-factor blocks have no public deterministic site.
            if name not in {"input_effect", "static_state_sds"} or value.size:
                numpyro.deterministic(name, value)
        numpyro.factor(
            "t0_correlation_positive_definite",
            jnp.where(
                min_eigenvalue > INITIAL_STATE_COV_MIN_EIGENVALUE,
                0.0,
                -1e6 * (INITIAL_STATE_COV_MIN_EIGENVALUE - min_eigenvalue),
            ),
        )
        return matrices

    def _sample_runtime_dynamics(
        self,
        diffusion_cov: jnp.ndarray,
        input_effect: jnp.ndarray,
    ) -> StochasticContinuousTimeStateEvolution:
        """Sample vector-field parameters inside the NumPyro trace."""
        from nof1_causal_lab.models.ssm.dynamics.spec import compile_dynamics

        compiled = compile_dynamics(self.spec.dynamics_spec)
        return continuous_state_evolution(
            vector_field=compiled.vector_field,
            vf_params=compiled.sample_params(self._prior_distribution),
            diffusion_cov=diffusion_cov,
            input_effect=input_effect,
        )

    def model(
        self,
        observations: jnp.ndarray,
        times: jnp.ndarray,
        likelihood_backend=None,
    ) -> None:
        """NumPyro model function.

        Args:
            observations: (N, n_manifest) observed data
            times: (N,) observation times
            likelihood_backend: Laplace likelihood backend instance. Required —
                construct it in the inference warmup layer.
        """
        if likelihood_backend is None:
            raise ValueError(
                "likelihood_backend is required. Construct it in the inference warmup layer."
            )

        spec = self.spec
        sampled = self._sample_parameters()

        diffusion_chol = sampled["diffusion"]
        input_effect = sampled["input_effect"]
        lambda_mat = sampled["lambda"]
        manifest_means = sampled["manifest_means"]
        t0_means = sampled["t0_means"]

        diffusion_cov = diffusion_chol @ diffusion_chol.T
        manifest_cov = sampled["manifest_cov"]
        t0_cov = sampled["t0_cov"]
        extra_params = self._sample_likelihood_extra_params(spec)
        dynamics = self._sample_runtime_dynamics(diffusion_cov, input_effect)

        meas_params = MeasurementParams(
            lambda_mat=lambda_mat,
            manifest_means=manifest_means,
            manifest_cov=manifest_cov,
        )

        time_intervals = jnp.diff(times, prepend=times[0])
        time_intervals = time_intervals.at[0].set(MIN_DT)

        init = MultivariateNormal(loc=t0_means, covariance_matrix=t0_cov)
        lnc = likelihood_backend.compute_log_likelihood(
            dynamics,
            meas_params,
            init,
            observations,
            time_intervals,
            extra_params=extra_params or None,
            transition_inputs=self.transition_inputs,
        )

        # lnc is (T,) cumulative log-normalizing constants from the filter.
        # lnc[-1] = total log p(y|θ).
        # diff(lnc) exposes per-timestep contributions to the initialization
        # objective. Reported LOO uses emission factors on joint particle draws.
        if lnc.ndim == 0:
            total_ll = _nan_safe_ll(lnc)
            numpyro.factor("log_likelihood", total_ll)
        else:
            total_ll = _nan_safe_ll(lnc[-1])
            numpyro.factor("log_likelihood", total_ll)
            ll_per_timestep = jnp.diff(lnc, prepend=0.0)
            numpyro.deterministic("ll_per_timestep", ll_per_timestep)
