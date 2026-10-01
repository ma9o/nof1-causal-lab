"""Test-owned ModelSpec construction helpers."""

from __future__ import annotations

from pathlib import Path

from typing import TYPE_CHECKING, Any, cast, override

import dynestyx as dsx
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
from dynestyx.inference.configs.discretizer import ExactAffineConfig

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import CoefficientExpression, StateExpression
from nof1_causal_lab.artifacts.expressions import (
    coefficient as expr_coefficient,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.autoreparam import Strategy, _minimal_reparam
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.execution.observation_families import (
    resolve_manifest_families_and_links,
)
from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from tests.dynamics_fixtures import decay_term, intercept_term, linear_term
from tests.helpers import fixture_entity_id, native_axis_metadata

if TYPE_CHECKING:
    from nof1_causal_lab.measurement_types import MeasurementDtype
    from nof1_causal_lab.models.ssm.autoreparam import ReparamSite


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
    def configure(self, msg: ReparamSite):
        return _minimal_reparam(msg["fn"], msg.get("is_observed", False))


def zero_loading_support(n_manifest: int, n_latent: int) -> np.ndarray:
    return np.zeros((n_manifest, n_latent), dtype=bool)


def full_vector_support(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def full_diagonal_support(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def full_cholesky_support(n: int) -> np.ndarray:
    return np.tri(n, dtype=bool)


def default_diffusion_block(n_latent: int) -> DiffusionBlockSpec:
    return DiffusionBlockSpec(
        n_latent=n_latent,
        diffusion_chol_support=np.tri(n_latent, dtype=bool),
        diffusion_chol_template=jnp.eye(n_latent),
    )


def default_lambda_block(n_manifest: int, n_latent: int) -> SparseMatrixBlockSpec:
    return SparseMatrixBlockSpec(
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
    )


def default_manifest_means_block(n_manifest: int) -> SparseVectorBlockSpec:
    return SparseVectorBlockSpec(
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
    )


def default_manifest_chol_block(n_manifest: int) -> ManifestCholBlockSpec:
    return ManifestCholBlockSpec(
        n_manifest=n_manifest,
        diag_support=np.ones(n_manifest, dtype=bool),
        template=jnp.zeros((n_manifest, n_manifest)),
    )


def default_t0_means_block(n_latent: int) -> SparseVectorBlockSpec:
    return SparseVectorBlockSpec(
        n=n_latent,
        free_support=np.ones(n_latent, dtype=bool),
        template=jnp.zeros(n_latent),
        free_site_name="t0_means_free",
        det_site_name="t0_means",
        support=SupportClass.REAL,
        site_kind=SiteKind.T0_MEANS,
        assembly_group="t0",
        fixed_spec_field="t0_means",
        priors_field="t0_means",
    )


def default_t0_chol_block(n_latent: int) -> T0CholBlockSpec:
    return T0CholBlockSpec(
        n_latent=n_latent,
        diag_support=np.ones(n_latent, dtype=bool),
        correlation_support=np.tri(n_latent, k=-1, dtype=bool),
        template=jnp.eye(n_latent),
    )


def default_static_state_sd_block() -> SparseVectorBlockSpec:
    return SparseVectorBlockSpec(
        n=0,
        free_support=np.zeros(0, dtype=bool),
        template=jnp.zeros(0),
        free_site_name="static_state_sd_free",
        det_site_name="static_state_sds",
        support=SupportClass.POSITIVE,
        site_kind=SiteKind.STATIC_STATE_SD,
        assembly_group="t0",
        fixed_spec_field="static_state_sds",
        priors_field="static_state_sd",
    )


def dense_matrix_dynamics_spec(
    *,
    n_latent: int,
    decay_support: np.ndarray,
    edge_support: np.ndarray,
    coupling_template: jnp.ndarray,
    intercept_support: np.ndarray,
    cint_template: jnp.ndarray,
    time_invariant_mask: np.ndarray | None = None,
    stability_margin: float = 0.05,
) -> DynamicsSpec:
    """Build a component-native dense-matrix dynamics fixture for tests."""
    del stability_margin

    components: list[Any] = []
    diag_support = np.asarray(decay_support, dtype=bool)
    edge_support = np.asarray(edge_support, dtype=bool)
    coupling_template_array = np.asarray(coupling_template, dtype=float)
    ti_mask = (
        np.asarray(time_invariant_mask, dtype=bool)
        if time_invariant_mask is not None
        else np.zeros(n_latent, dtype=bool)
    )

    for target in range(n_latent):
        if bool(ti_mask[target]):
            continue
        fixed_diag = float(coupling_template_array[target, target])
        if bool(diag_support[target]) or fixed_diag < 0.0:
            components.append(decay_term(target=target))
        elif fixed_diag > 0.0:
            components.append(linear_term(source=target, target=target))

    for effect in range(n_latent):
        for cause in range(n_latent):
            if effect == cause:
                continue
            if bool(edge_support[effect, cause]):
                components.append(
                    linear_term(
                        source=cause,
                        target=effect,
                    )
                )
                continue
            fixed_weight = float(coupling_template_array[effect, cause])
            if fixed_weight != 0.0:
                components.append(
                    linear_term(
                        source=cause,
                        target=effect,
                    )
                )

    intercept_support_array = np.asarray(intercept_support, dtype=bool)
    cint_template_array = np.asarray(cint_template, dtype=float)
    for target in range(n_latent):
        fixed_cint = float(cint_template_array[target])
        if bool(intercept_support_array[target]) or fixed_cint != 0.0:
            components.append(intercept_term(target=target))

    return DynamicsSpec(n_latent=n_latent, components=tuple(components))


def full_dense_matrix_dynamics_spec(n_latent: int) -> DynamicsSpec:
    """Build a full-free structural dense dynamics fixture for tests."""
    return dense_matrix_dynamics_spec(
        n_latent=n_latent,
        decay_support=np.ones(n_latent, dtype=bool),
        edge_support=np.ones((n_latent, n_latent), dtype=bool) & ~np.eye(n_latent, dtype=bool),
        coupling_template=jnp.zeros((n_latent, n_latent)),
        intercept_support=np.zeros(n_latent, dtype=bool),
        cint_template=jnp.zeros(n_latent),
    )


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

    spec = ModelSpec.model_validate_json((Path(__file__).resolve().parents[0] / "fixtures/models" / 'model_fixtures/make_lgss_data_model_fixture.json').read_text())

    return {
        "observations": observations,
        "times": times,
        "spec": spec,
        "true_decay_diag": decay_diag,
        "true_diff_diag": diff_sd,
        "true_obs_sd": obs_sd,
        "n_latent": n_latent,
    }




def diagonal_diffusion_block(n_latent: int) -> DiffusionBlockSpec:
    """Diagonal-only diffusion: only diagonal entries free, identity template."""
    return DiffusionBlockSpec(
        n_latent=n_latent,
        diffusion_chol_support=np.diag(np.ones(n_latent, dtype=bool)),
        diffusion_chol_template=jnp.eye(n_latent),
    )


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
    return ObservationSupportRuntime(**kwargs)


def parameter_draws(model: ModelSpec, n_draws: int) -> dict[str, jnp.ndarray]:
    """Repeat the authored prior reference point without invoking inference."""
    from nof1_causal_lab.models.ssm.compile.inputs import compile_priors
    from nof1_causal_lab.prior_distributions import prior_reference_value

    priors, _, _ = compile_priors(model)
    return {
        name: jnp.broadcast_to(value, (n_draws, *value.shape))
        for name, law in priors.items()
        for value in [jnp.asarray(prior_reference_value(law))]
    }


def compile_fit_fixture(spec: ModelSpec):
    """Require real compilation in fixtures instead of forging fit evidence."""
    from nof1_causal_lab.models.ssm.compile.inputs import (
        CompiledFitInputs,
        compile_ssm_inputs_from_model,
    )

    inputs = compile_ssm_inputs_from_model(spec)
    assert isinstance(inputs, CompiledFitInputs), inputs
    return inputs
