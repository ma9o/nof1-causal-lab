"""Tests for the canonical site registry and compile-stable prior evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.parameter import SupportClass
from nof1_causal_lab.artifacts.statistical_model_spec import LinkFunction, StatisticalModelSpec
from nof1_causal_lab.distributions import (
    DistributionFamily,
    PriorDistributionFamily,
)
from nof1_causal_lab.models.ssm.inference.backend_factory import get_laplace_backend
from nof1_causal_lab.models.ssm.inference.utils import _discover_sites
from nof1_causal_lab.models.ssm.model import (
    SSMModel,
    SSMSpec,
)
from nof1_causal_lab.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
    compile_prior_semantics,
    deserialize_site_registry,
    load_prior_runtime_bundle,
    sample_prior_parameters,
    serialize_site_registry,
)
from nof1_causal_lab.models.ssm.priors import resolve_site_priors
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from nof1_causal_lab.prior_distributions import distribution_from_params
from tests.helpers import model_with_prior_payloads, named_prior_payloads

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.sampler_config import SamplerConfigOverride
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform, SiteKind
from tests.helpers import make_prior_model, native_axis_metadata
from tests.ssm_spec_fixtures import (
    default_diffusion_block,
    default_input_effect_block,
    default_lambda_block,
    default_manifest_chol_block,
    default_manifest_means_block,
    default_static_state_sd_block,
    default_t0_chol_block,
    default_t0_means_block,
    dense_matrix_dynamics_spec,
    full_cholesky_support,
    full_dense_matrix_dynamics_spec,
    full_diagonal_support,
    full_vector_support,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spec(
    *,
    n_latent: int = 2,
    n_manifest: int = 2,
    dynamics_spec=None,
    diffusion_block=None,
    lambda_block=None,
    manifest_means_block=None,
    manifest_chol_block=None,
    t0_means_block=None,
    t0_chol_block=None,
    input_effect_block=None,
    static_state_sd_block=None,
    **kwargs,
) -> SSMSpec:
    """Build an SSMSpec from explicit block specs for tests."""
    if dynamics_spec is None:
        dynamics_spec = full_dense_matrix_dynamics_spec(n_latent)
    return SSMSpec(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dynamics_spec,
        diffusion_block=diffusion_block or default_diffusion_block(n_latent),
        lambda_block=lambda_block or default_lambda_block(n_manifest, n_latent),
        manifest_means_block=manifest_means_block or default_manifest_means_block(n_manifest),
        manifest_chol_block=manifest_chol_block or default_manifest_chol_block(n_manifest),
        t0_means_block=t0_means_block or default_t0_means_block(n_latent),
        t0_chol_block=t0_chol_block or default_t0_chol_block(n_latent),
        input_effect_block=input_effect_block or default_input_effect_block(n_latent),
        static_state_sd_block=static_state_sd_block or default_static_state_sd_block(),
        **native_axis_metadata(n_latent, n_manifest, kwargs),
    )


def _linear_dynamics(
    n_latent: int,
    *,
    decay_support: np.ndarray | None = None,
    edge_support: np.ndarray | None = None,
    coupling_template=None,
    intercept_support: np.ndarray | None = None,
    cint_template=None,
):
    if decay_support is None:
        decay_support = full_diagonal_support(n_latent)
    if edge_support is None:
        edge_support = np.zeros((n_latent, n_latent), dtype=bool)
    if coupling_template is None:
        coupling_template = jnp.zeros((n_latent, n_latent), dtype=jnp.float32)
    if intercept_support is None:
        intercept_support = np.zeros(n_latent, dtype=bool)
    if cint_template is None:
        cint_template = jnp.zeros(n_latent, dtype=jnp.float32)
    return dense_matrix_dynamics_spec(
        n_latent=n_latent,
        decay_support=decay_support,
        edge_support=edge_support,
        coupling_template=jnp.asarray(coupling_template),
        intercept_support=intercept_support,
        cint_template=jnp.asarray(cint_template),
    )


def _fixed_matrix_dynamics(coupling_template):
    coupling_template = jnp.asarray(coupling_template)
    n_latent = int(coupling_template.shape[0])
    return _linear_dynamics(
        n_latent,
        decay_support=np.zeros(n_latent, dtype=bool),
        edge_support=np.zeros((n_latent, n_latent), dtype=bool),
        coupling_template=coupling_template,
    )


def _diffusion_block(n_latent: int, *, free_support=None, template=None) -> DiffusionBlockSpec:
    if free_support is None:
        free_support = np.tri(n_latent, dtype=bool)
    if template is None:
        template = jnp.eye(n_latent)
    return DiffusionBlockSpec(
        n_latent=n_latent,
        diffusion_chol_support=free_support,
        diffusion_chol_template=jnp.asarray(template),
    )


def _lambda_block(n_manifest: int, n_latent: int, *, free_support=None, template=None):
    if free_support is None:
        free_support = np.zeros((n_manifest, n_latent), dtype=bool)
    if template is None:
        template = jnp.eye(n_manifest, n_latent)
    return SparseMatrixBlockSpec(
        n_rows=n_manifest,
        n_cols=n_latent,
        free_support=free_support,
        template=jnp.asarray(template),
        free_site_name="lambda_free",
        det_site_name="lambda",
        support=SupportClass.REAL,
        site_kind=SiteKind.LOADING,
        assembly_group="lambda",
        fixed_spec_field="lambda_mat",
        priors_field="lambda_free",
    )


def _manifest_chol_block(n_manifest: int, *, diag_support=None, template=None):
    if diag_support is None:
        diag_support = full_diagonal_support(n_manifest)
    if template is None:
        template = jnp.zeros((n_manifest, n_manifest))
    return ManifestCholBlockSpec(
        n_manifest=n_manifest,
        diag_support=diag_support,
        template=jnp.asarray(template),
    )


def _t0_means_block(n_latent: int, *, free_support=None, template=None):
    if free_support is None:
        free_support = np.ones(n_latent, dtype=bool)
    if template is None:
        template = jnp.zeros(n_latent)
    return SparseVectorBlockSpec(
        n=n_latent,
        free_support=free_support,
        template=jnp.asarray(template),
        free_site_name="t0_means_free",
        det_site_name="t0_means",
        support=SupportClass.REAL,
        site_kind=SiteKind.T0_MEANS,
        assembly_group="t0",
        fixed_spec_field="t0_means",
        priors_field="t0_means",
    )


def _t0_chol_block(n_latent: int, *, diag_support=None, correlation_support=None, template=None):
    if diag_support is None:
        diag_support = full_diagonal_support(n_latent)
    if correlation_support is None:
        correlation_support = np.tri(n_latent, k=-1, dtype=bool)
    if template is None:
        template = jnp.eye(n_latent)
    return T0CholBlockSpec(
        n_latent=n_latent,
        diag_support=diag_support,
        correlation_support=correlation_support,
        template=jnp.asarray(template),
    )


def _static_state_sd_block(mask, template):
    return SparseVectorBlockSpec(
        n=int(np.asarray(mask).shape[0]),
        free_support=np.asarray(mask, dtype=bool),
        template=jnp.asarray(template),
        free_site_name="static_state_sd_free",
        det_site_name="static_state_sds",
        support=SupportClass.POSITIVE,
        site_kind=SiteKind.STATIC_STATE_SD,
        assembly_group="t0",
        fixed_spec_field="static_state_sds",
        priors_field="static_state_sd",
    )


@pytest.fixture
def simple_spec():
    """Minimal 2-latent, 2-manifest Gaussian SSM."""
    return _make_spec(n_latent=2, n_manifest=2)


@pytest.fixture
def simple_model(simple_spec):
    return SSMModel(simple_spec)


@pytest.fixture
def dag_spec():
    """DAG-constrained spec with dynamics mask and lambda mask."""
    import numpy as np

    edge_support = np.array([[False, False], [True, False]])
    lambda_support = np.array([[True, False], [False, True]])
    lambda_template = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    return _make_spec(
        n_latent=2,
        n_manifest=2,
        dynamics_spec=_linear_dynamics(
            2,
            edge_support=edge_support,
            intercept_support=full_vector_support(2),
        ),
        lambda_block=_lambda_block(
            2,
            2,
            free_support=lambda_support,
            template=lambda_template,
        ),
    )


@pytest.fixture
def dag_model(dag_spec):
    return SSMModel(dag_spec)


@pytest.fixture
def statistical_model_spec_and_priors():
    return (
        StatisticalModelSpec.model_validate(
            {
                "mechanisms": [
                    {
                        "kind": "node_potential",
                        "target_id": "construct:bbc87212909e45b9e6c3",
                        "center": {"kind": "fixed", "value": 0},
                        "stiffness": {
                            "kind": "estimated",
                            "parameter_id": "parameter:fb33dbedf43eb15e324c86fa97201278e306cb48aa9752104361309d61215122",
                        },
                        "quartic": {"kind": "fixed", "value": 0},
                    }
                ],
                "likelihoods": [
                    {
                        "indicator_id": "indicator:45f78731e3e0c6f3efe1",
                        "distribution": "gaussian",
                        "link": "identity",
                        "reasoning": "test",
                    }
                ],
                "parameters": [
                    {
                        "prior_transform": "dt_persistence_to_ct_decay",
                        "id": "parameter:fb33dbedf43eb15e324c86fa97201278e306cb48aa9752104361309d61215122",
                        "owners": [{"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"}],
                        "quantity": "dynamics_decay",
                        "name": "rho_mood",
                        "role": "ar_coefficient",
                        "constraint": "unit_interval",
                        "description": "AR mood",
                    },
                    {
                        "id": "parameter:146688c9f8e2c980c9e7963be61deb23225a81f828c204339c2164d1f51d441e",
                        "owners": [{"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"}],
                        "quantity": "diffusion_diag",
                        "name": "sigma_mood",
                        "role": "residual_sd",
                        "constraint": "positive",
                        "description": "SD mood",
                    },
                ],
            }
        ),
        {
            "rho_mood": {
                "parameter": "rho_mood",
                "distribution": "Beta",
                "params": {"alpha": 2.0, "beta": 2.0},
                "sources": [],
                "reasoning": "r",
            },
            "sigma_mood": {
                "parameter": "sigma_mood",
                "distribution": "HalfNormal",
                "params": {"sigma": 1.0},
                "sources": [],
                "reasoning": "r",
            },
        },
    )


def _mood_structural_plan() -> StructuralPlan:
    from nof1_causal_lab.models.structural import build_structural_plan

    return build_structural_plan(
        CausalDesign.model_validate(
            {
                "latent": {
                    "constructs": [
                        {
                            "id": "construct:bbc87212909e45b9e6c3",
                            "name": "mood",
                            "description": "Mood",
                            "role": "exogenous",
                            "temporal_status": "time_varying",
                        }
                    ],
                    "edges": [],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "id": "indicator:45f78731e3e0c6f3efe1",
                            "construct_id": "construct:bbc87212909e45b9e6c3",
                            "name": "mood_score",
                            "how_to_measure": "Mood score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "construct_polarity": "positive",
                        }
                    ],
                },
                "default_outcome": None,
            }
        )
    )


# ---------------------------------------------------------------------------
# Site registry tests
# ---------------------------------------------------------------------------


def _assert_registry_matches_trace(registry, site_info):
    """Keep the registry/trace integration assertion owned by this test module."""
    assert {site.name for site in registry} == set(site_info)
    for site in registry:
        assert site.shape == site_info[site.name]["shape"]


class TestSiteRegistry:
    @pytest.mark.cpu_expensive
    def test_registry_names_match_trace(self, simple_model):
        """Registry produces the same site names as model tracing."""
        spec = simple_model.spec
        registry = build_site_registry(spec)
        backend = get_laplace_backend(simple_model, 6)
        T = 10
        obs = jnp.zeros((T, spec.n_manifest))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(simple_model, obs, times, random.PRNGKey(0), backend)
        _assert_registry_matches_trace(registry, site_info)

    @pytest.mark.cpu_expensive
    def test_registry_names_match_trace_dag(self, dag_model):
        """Registry matches trace for DAG-constrained model with cint."""
        spec = dag_model.spec
        registry = build_site_registry(spec)
        backend = get_laplace_backend(dag_model, 6)
        T = 10
        obs = jnp.zeros((T, spec.n_manifest))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(dag_model, obs, times, random.PRNGKey(0), backend)
        _assert_registry_matches_trace(registry, site_info)

    @pytest.mark.cpu_expensive
    def test_registry_shapes_match_trace(self, simple_model):
        """Registry shapes match traced shapes."""
        spec = simple_model.spec
        registry = build_site_registry(spec)
        backend = get_laplace_backend(simple_model, 6)
        T = 10
        obs = jnp.zeros((T, spec.n_manifest))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(simple_model, obs, times, random.PRNGKey(0), backend)
        for site in registry:
            assert site.shape == site_info[site.name]["shape"], (
                f"Shape mismatch for {site.name}: "
                f"registry={site.shape}, trace={site_info[site.name]['shape']}"
            )

    @pytest.mark.cpu_expensive
    def test_registry_shapes_match_trace_partial_manifest_variance_mask(self):
        """Masked manifest variance exposes only free diagonal entries as a site."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            manifest_chol_block=_manifest_chol_block(
                2,
                diag_support=np.array([False, True]),
                template=jnp.diag(jnp.array([0.4, 0.0], dtype=jnp.float32)),
            ),
        )
        model = SSMModel(spec)
        registry = build_site_registry(spec)
        backend = get_laplace_backend(model, 6)
        T = 10
        obs = jnp.zeros((T, spec.n_manifest))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(model, obs, times, random.PRNGKey(0), backend)

        _assert_registry_matches_trace(registry, site_info)
        manifest_site = next(site for site in registry if site.name == "manifest_var_diag_free")
        assert manifest_site.shape == (1,)

    def test_fixed_dynamics_excludes_dynamics_sites(self):
        """When dynamics is a fixed array, no dynamics sites appear."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            dynamics_spec=_fixed_matrix_dynamics(
                jnp.array([[-0.5, 0.0], [0.0, -0.5]], dtype=jnp.float32)
            ),
        )
        registry = build_site_registry(spec)
        names = {s.name for s in registry}
        assert {"vf_0_decay", "vf_1_decay"}.issubset(names)

    def test_diag_diffusion_excludes_lower(self):
        """Diagonal diffusion has no lower-triangle sites."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            diffusion_block=_diffusion_block(
                2,
                free_support=np.diag(full_diagonal_support(2)),
                template=jnp.eye(2),
            ),
        )
        registry = build_site_registry(spec)
        names = {s.name for s in registry}
        assert "diffusion_diag_free" in names
        assert "diffusion_lower_free" not in names

    def test_free_diffusion_includes_lower(self):
        """Free diffusion includes lower-triangle sites."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            diffusion_block=_diffusion_block(
                2,
                free_support=full_cholesky_support(2),
                template=jnp.eye(2),
            ),
        )
        registry = build_site_registry(spec)
        names = {s.name for s in registry}
        assert "diffusion_diag_free" in names
        assert "diffusion_lower_free" in names

    def test_sparse_initial_state_correlations_only_include_authored_pairs(self):
        """Initial-state correlation sites should only exist for authored pairs."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 0] = True
        spec = _make_spec(
            n_latent=3,
            n_manifest=2,
            t0_chol_block=_t0_chol_block(
                3,
                diag_support=full_diagonal_support(3),
                correlation_support=mask,
                template=jnp.eye(3),
            ),
        )
        registry = build_site_registry(spec)
        site_map = {site.name: site for site in registry}
        assert site_map["t0_var_lower_free"].shape == (1,)

    def test_support_classes(self, simple_spec):
        """Check that support classes are correctly assigned."""
        registry = build_site_registry(simple_spec)
        support_map = {s.name: s.support for s in registry}
        # POSITIVE support sites
        assert support_map["vf_0_decay"] == SupportClass.POSITIVE
        # POSITIVE support sites (HalfNormal priors)
        assert support_map["diffusion_diag_free"] == SupportClass.POSITIVE
        assert support_map["manifest_var_diag_free"] == SupportClass.POSITIVE
        assert support_map["t0_var_diag_free"] == SupportClass.POSITIVE

    def test_mixed_diffusion_includes_proc_df_site(self):
        """Any student-t latent in diffusion_dists should expose proc_df."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=1,
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        registry = build_site_registry(spec)
        assert "proc_df" in {site.name for site in registry}

    def test_mixed_diffusion_sampling_emits_proc_df(self):
        """The traced model should sample proc_df when diffusion_dists include student_t."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=1,
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        model = SSMModel(spec)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(lambda: model._sample_likelihood_extra_params(spec)).get_trace()

        assert "proc_df" in trace

    @pytest.mark.cpu_expensive
    def test_static_state_sd_site_is_registered_and_traced(self):
        """Compiled baseline factors should expose a positive static-state SD site."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=1,
            static_state_sd_block=_static_state_sd_block(
                np.array([True]),
                jnp.zeros(1),
            ),
            static_factor_loadings=jnp.array([[1.0], [1.0]]),
            t0_chol_block=_t0_chol_block(
                2,
                diag_support=np.zeros(2, dtype=bool),
                correlation_support=np.zeros((2, 2), dtype=bool),
                template=jnp.eye(2),
            ),
        )
        model = SSMModel(spec)

        registry = build_site_registry(spec)
        site_map = {site.name: site for site in registry}
        assert site_map["static_state_sd_free"].shape == (1,)
        assert site_map["static_state_sd_free"].support == SupportClass.POSITIVE

        backend = get_laplace_backend(model, 6)
        obs = jnp.zeros((5, spec.n_manifest))
        times = jnp.arange(5, dtype=jnp.float32)
        site_info = _discover_sites(model, obs, times, random.PRNGKey(0), backend)
        _assert_registry_matches_trace(registry, site_info)
        assert site_info["static_state_sd_free"]["shape"] == (1,)


class TestSpecBlockAssembly:
    def test_assemble_t0_cov_adds_low_rank_baseline_factor_covariance(self):
        """Static baseline factors should add `B diag(tau^2) B^T` to the t0 covariance."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=1,
            static_state_sd_block=_static_state_sd_block(
                np.array([True]),
                jnp.zeros(1),
            ),
            static_factor_loadings=jnp.array([[1.0], [1.0]]),
            t0_chol_block=_t0_chol_block(
                2,
                diag_support=np.zeros(2, dtype=bool),
                correlation_support=np.zeros((2, 2), dtype=bool),
                template=jnp.eye(2),
            ),
        )
        model = SSMModel(spec)
        values = {
            site.name: jnp.ones(site.shape)
            for block in spec.parameter_blocks
            for site in block.iter_sites()
        }
        values["static_state_sd_free"] = jnp.array([2.0])
        with handlers.substitute(data=values), handlers.trace() as trace:
            cov = model._sample_parameters()["t0_cov"]
        assert "t0_correlation_positive_definite" in trace

        np.testing.assert_allclose(
            np.asarray(cov),
            np.array([[5.0, 4.0], [4.0, 5.0]]),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Deterministic assembly
# ---------------------------------------------------------------------------


class TestDeterministicAssembly:
    def test_assemble_deterministics_from_registry_free_spec(self, simple_spec):
        """Registry-driven assembly builds the expected matrices."""
        samples = {
            "vf_0_decay": jnp.array([[0.5, 0.3]], dtype=jnp.float32),
            "vf_1_weight": jnp.array([0.1], dtype=jnp.float32),
            "vf_2_weight": jnp.array([-0.2], dtype=jnp.float32),
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[0.9, 1.1]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.zeros((1, 1), dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, simple_spec)
        assert jnp.allclose(det["diffusion"][0], jnp.array([[0.4, 0.0], [0.25, 0.6]]))
        assert det["lambda"].shape == (1, 2, 2)
        assert jnp.allclose(det["manifest_cov"][0], jnp.diag(jnp.array([0.49, 0.64])))
        assert jnp.allclose(det["t0_means"][0], jnp.array([1.0, -1.0]))
        assert jnp.allclose(det["t0_cov"][0], jnp.diag(jnp.array([0.81, 1.21])))

    def test_missing_declared_free_value_is_rejected(self, simple_spec):
        with pytest.raises(KeyError, match="diffusion_diag_free"):
            assemble_deterministics_from_registry({}, simple_spec, n_draws=2)

    def test_assemble_deterministics_from_registry_fixed_blocks(self):
        """Fixed spec matrices are broadcast without any sampled sites."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            dynamics_spec=_fixed_matrix_dynamics(
                jnp.array([[-0.4, 0.1], [0.0, -0.2]], dtype=jnp.float32)
            ),
            diffusion_block=_diffusion_block(
                2,
                free_support=np.zeros((2, 2), dtype=bool),
                template=jnp.array([[0.3, 0.0], [0.1, 0.5]], dtype=jnp.float32),
            ),
            lambda_block=_lambda_block(
                2,
                2,
                template=jnp.array([[1.0, 0.0], [0.2, 1.0]], dtype=jnp.float32),
            ),
            manifest_chol_block=_manifest_chol_block(
                2,
                diag_support=np.zeros(2, dtype=bool),
                template=jnp.array([[0.4, 0.0], [0.0, 0.6]], dtype=jnp.float32),
            ),
            t0_means_block=_t0_means_block(
                2,
                free_support=np.zeros(2, dtype=bool),
                template=jnp.array([0.5, -0.5], dtype=jnp.float32),
            ),
            t0_chol_block=_t0_chol_block(
                2,
                diag_support=np.zeros(2, dtype=bool),
                correlation_support=np.zeros((2, 2), dtype=bool),
                template=jnp.array([[0.7, 0.0], [0.0, 0.8]], dtype=jnp.float32),
            ),
        )
        det = assemble_deterministics_from_registry({}, spec, n_draws=3)
        assert jnp.allclose(
            det["diffusion"],
            jnp.broadcast_to(spec.diffusion_block.assemble(), (3, 2, 2)),
        )
        assert jnp.allclose(
            det["lambda"],
            jnp.broadcast_to(spec.lambda_block.assemble(), (3, 2, 2)),
        )
        manifest_chol = spec.manifest_chol_block.assemble()
        expected_manifest_cov = manifest_chol @ manifest_chol.T
        assert jnp.allclose(det["manifest_cov"], jnp.broadcast_to(expected_manifest_cov, (3, 2, 2)))
        assert isinstance(spec.t0_means_block.assemble(), jnp.ndarray)
        assert jnp.allclose(
            det["t0_means"],
            jnp.broadcast_to(spec.t0_means_block.assemble(), (3, 2)),
        )
        expected_t0_cov = spec.t0_chol_block.assemble_cov()
        assert jnp.allclose(det["t0_cov"], jnp.broadcast_to(expected_t0_cov, (3, 2, 2)))

    def test_assemble_deterministics_from_registry_partial_manifest_variance_mask(self):
        """Registry assembly respects mixed fixed/free manifest-noise diagonals."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            manifest_chol_block=_manifest_chol_block(
                2,
                diag_support=np.array([False, True]),
                template=jnp.diag(jnp.array([0.4, 0.0], dtype=jnp.float32)),
            ),
        )
        samples = {
            "vf_0_decay": jnp.array([[0.5, 0.3]], dtype=jnp.float32),
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.9]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[0.9, 1.1]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.zeros((1, 1), dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, spec)
        assert jnp.allclose(det["manifest_cov"][0], jnp.diag(jnp.array([0.16, 0.81])))

    def test_assemble_deterministics_from_registry_initial_state_correlations(self):
        """Initial-state off-diagonal samples are interpreted as correlations."""
        mask = np.zeros((2, 2), dtype=bool)
        mask[1, 0] = True
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            t0_chol_block=_t0_chol_block(
                2,
                diag_support=full_diagonal_support(2),
                correlation_support=mask,
                template=jnp.eye(2),
            ),
        )
        samples = {
            "vf_0_decay": jnp.array([[0.5, 0.3]], dtype=jnp.float32),
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[2.0, 3.0]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, spec)

        assert jnp.allclose(
            det["t0_cov"][0],
            jnp.array([[4.0, 1.5], [1.5, 9.0]], dtype=jnp.float32),
        )

    def test_assemble_deterministics_repairs_invalid_initial_correlation_matrix(self):
        """Impossible authored initial correlations are repaired to a PSD covariance."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 0] = True
        mask[2, 0] = True
        mask[2, 1] = True
        spec = _make_spec(
            n_latent=3,
            n_manifest=3,
            t0_chol_block=_t0_chol_block(
                3,
                diag_support=full_diagonal_support(3),
                correlation_support=mask,
                template=jnp.eye(3),
            ),
        )
        samples = {
            "vf_0_decay": jnp.array([[0.5, 0.3, 0.4]], dtype=jnp.float32),
            "diffusion_diag_free": jnp.array([[0.4, 0.6, 0.5]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25, 0.1, -0.15]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8, 0.9]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0, 0.5]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[1.0, 1.0, 1.0]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.array([[0.9, 0.9, -0.9]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, spec)
        min_eig = jnp.min(jnp.linalg.eigvalsh(det["t0_cov"][0]))

        assert bool(jnp.isfinite(det["t0_cov"]).all())
        assert float(min_eig) > -1e-6

        model = SSMModel(spec)
        with handlers.substitute(data={name: value[0] for name, value in samples.items()}):
            trace = handlers.trace(model._sample_parameters).get_trace()
        np.testing.assert_allclose(trace["t0_cov"]["value"], det["t0_cov"][0], atol=1e-6)
        factor = trace["t0_correlation_positive_definite"]
        assert float(factor["fn"].log_prob(factor["value"])) == pytest.approx(-800000.01, rel=1e-5)


# ---------------------------------------------------------------------------
# Prior runtime state
# ---------------------------------------------------------------------------


class TestNativeRuntimePriors:
    def test_sites_have_native_distributions_with_the_declared_shapes(self, simple_spec):
        registry = build_site_registry(simple_spec)
        priors = resolve_site_priors(registry)
        assert set(priors) == {site.name for site in registry}
        for site in registry:
            assert isinstance(priors[site.name], dist.Distribution)
            assert priors[site.name].batch_shape == site.shape

    def test_partial_native_overrides_preserve_other_defaults(self, simple_spec):
        registry = build_site_registry(simple_spec)
        priors = resolve_site_priors(registry, {"vf_0_decay": dist.Gamma(4.0, 2.0)})
        np.testing.assert_allclose(priors["vf_0_decay"].mean, [2.0, 2.0])
        assert priors["diffusion_diag_free"].batch_shape == (2,)

    def test_native_laws_are_jax_pytrees(self, simple_spec):
        priors = resolve_site_priors(build_site_registry(simple_spec))
        leaves, tree = jax.tree.flatten(priors)
        restored = jax.tree.unflatten(tree, leaves)
        for name, prior in priors.items():
            np.testing.assert_array_equal(restored[name].mean, prior.mean)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


class TestSampling:
    def test_sample_shapes_and_native_support(self, simple_spec):
        registry = build_site_registry(simple_spec)
        state = resolve_site_priors(registry)
        samples = sample_prior_parameters(random.PRNGKey(0), registry, state, n_samples=4)
        assert set(samples) == {site.name for site in registry}
        for site in registry:
            assert samples[site.name].shape == (4, *site.shape)
            assert jnp.all(jnp.isfinite(samples[site.name]))
            if site.support == SupportClass.POSITIVE:
                assert jnp.all(samples[site.name] > 0)

    def test_site_streams_are_stable_under_registry_reordering(self, simple_spec):
        registry = build_site_registry(simple_spec)
        state = resolve_site_priors(registry)
        samples = sample_prior_parameters(random.PRNGKey(7), registry, state, n_samples=4)
        reversed_samples = sample_prior_parameters(
            random.PRNGKey(7), list(reversed(registry)), state, n_samples=4
        )
        for site_name in samples:
            assert jnp.array_equal(samples[site_name], reversed_samples[site_name])

    def test_fixed_model_has_no_sampled_parameters(self):
        assert sample_prior_parameters(random.PRNGKey(0), [], {}, n_samples=3) == {}


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_registry_roundtrip(self, simple_spec):
        registry = build_site_registry(simple_spec)
        restored = deserialize_site_registry(serialize_site_registry(registry))
        assert [(site.name, site.shape, site.support, site.site_kind) for site in restored] == [
            (site.name, site.shape, site.support, site.site_kind) for site in registry
        ]

    def test_native_prior_laws_roundtrip(self, simple_spec):
        priors = {"vf_0_decay": dist.Gamma(jnp.array([3.0, 4.0]), jnp.array([2.0, 5.0]))}
        semantics = compile_prior_semantics(simple_spec, priors)
        assert semantics.schema_version == 7
        restored = load_prior_runtime_bundle(semantics)
        prior = restored.priors["vf_0_decay"]
        np.testing.assert_allclose(prior.concentration, [3.0, 4.0])
        np.testing.assert_allclose(prior.rate, [2.0, 5.0])
        np.testing.assert_allclose(
            prior.log_prob(jnp.array([0.5, 1.0])),
            priors["vf_0_decay"].log_prob(jnp.array([0.5, 1.0])),
        )


class TestCanonicalRuntimePriors:
    def test_loaded_runtime_preserves_per_element_priors(self):
        spec = _make_spec(n_latent=3, n_manifest=3)
        priors = {"vf_0_decay": dist.Gamma(jnp.array([2.0, 3.0, 4.0]), jnp.array([4.0, 5.0, 6.0]))}
        runtime = load_prior_runtime_bundle(compile_prior_semantics(spec, priors))
        np.testing.assert_allclose(runtime.priors["vf_0_decay"].concentration, [2.0, 3.0, 4.0])

    def test_vector_positive_prior_is_a_native_half_normal(self, simple_spec):
        priors = {"t0_var_diag_free": dist.HalfNormal(jnp.array([1.0, 2.0]))}
        runtime = load_prior_runtime_bundle(compile_prior_semantics(simple_spec, priors))
        law = runtime.priors["t0_var_diag_free"]
        assert isinstance(law, dist.HalfNormal)
        assert law.batch_shape == (2,)
        np.testing.assert_allclose(law.scale, [1.0, 2.0])

    def test_delta_roundtrip_preserves_fixed_coordinates(self, simple_spec):
        priors = {"vf_0_decay": dist.Delta(jnp.array([0.25, 0.5]))}
        runtime = load_prior_runtime_bundle(compile_prior_semantics(simple_spec, priors))
        assert isinstance(runtime.priors["vf_0_decay"], dist.Delta)
        np.testing.assert_array_equal(runtime.priors["vf_0_decay"].v, [0.25, 0.5])


# ---------------------------------------------------------------------------
# Compiled artifact integration (hard cutover)
# ---------------------------------------------------------------------------


class TestCompiledArtifactIntegration:
    """Test that compiled_prior_semantics is emitted and correctly consumed."""

    def test_global_ordered_threshold_priors_are_not_authorable(self):
        from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors
        from nof1_causal_lab.models.ssm.compile.prior_indexing import PriorIndexingError

        spec = _make_spec(
            n_latent=1,
            n_manifest=1,
            latent_names=["burden"],
            manifest_names=["scale"],
            manifest_dists=[DistributionFamily.ORDERED_LOGISTIC],
            manifest_links=[LinkFunction.CUMULATIVE_LOGIT],
            manifest_level_counts=[4],
        )
        statistical_model_spec = StatisticalModelSpec.model_validate(
            {
                "mechanisms": [],
                "likelihoods": [
                    {
                        "indicator_id": "indicator:629caa38759753c3b1e8",
                        "distribution": "ordered_logistic",
                        "link": "cumulative_logit",
                        "reasoning": "test",
                    }
                ],
                "parameters": [
                    {
                        "id": "parameter:70806f3c141a7241a93e94d412f6f7a2a356392c691a2a777c6ac9ed9e3e82f1",
                        "owners": [{"kind": "indicator", "id": "indicator:8467e96e70bbb56aa963"}],
                        "quantity": "obs_ordered_base",
                        "name": "obs_ordered_base",
                        "role": "observation_hyperparameter",
                        "constraint": "none",
                        "description": "global base",
                    }
                ],
            }
        )

        with pytest.raises(
            PriorIndexingError,
            match="must bind to one active site through its scientific owners",
        ):
            compile_priors(
                model_with_prior_payloads(
                    statistical_model_spec,
                    named_prior_payloads(
                        statistical_model_spec,
                        {
                            "obs_ordered_base": {
                                "distribution": "Normal",
                                "params": {"mu": 0.0, "sigma": 1.0},
                            }
                        },
                    ),
                ),
                spec,
            )

    def test_ordered_threshold_priors_bind_per_manifest_component_and_row(self):
        from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors

        spec = _make_spec(
            n_latent=1,
            n_manifest=2,
            latent_names=["burden"],
            manifest_names=["short_scale", "long_scale"],
            manifest_dists=[
                DistributionFamily.ORDERED_LOGISTIC,
                DistributionFamily.ORDERED_LOGISTIC,
            ],
            manifest_links=[
                LinkFunction.CUMULATIVE_LOGIT,
                LinkFunction.CUMULATIVE_LOGIT,
            ],
            manifest_level_counts=[4, 10],
        )
        statistical_model_spec = StatisticalModelSpec.model_validate(
            {
                "mechanisms": [],
                "likelihoods": [
                    {
                        "indicator_id": "indicator:d50004e46a2ff28d2af2",
                        "distribution": "ordered_logistic",
                        "link": "cumulative_logit",
                        "reasoning": "test",
                    },
                    {
                        "indicator_id": "indicator:153ac0200cf2403b6774",
                        "distribution": "ordered_logistic",
                        "link": "cumulative_logit",
                        "reasoning": "test",
                    },
                ],
                "parameters": [
                    {
                        "id": "parameter:cf25138fc98c90be59a0cea63209ce252f18f82d932f8ff25dd07241c77df05a",
                        "owners": [{"kind": "indicator", "id": "indicator:d50004e46a2ff28d2af2"}],
                        "quantity": "obs_ordered_base",
                        "name": "obs_ordered_base_short_scale",
                        "role": "observation_hyperparameter",
                        "constraint": "none",
                        "description": "short base",
                    },
                    {
                        "id": "parameter:6c198078516656e980169ef85a15afad18fc7e11b5f1cbe01a61bcb1ebb92df2",
                        "owners": [{"kind": "indicator", "id": "indicator:d50004e46a2ff28d2af2"}],
                        "quantity": "obs_ordered_gaps",
                        "name": "obs_ordered_gaps_short_scale",
                        "role": "observation_hyperparameter_positive",
                        "constraint": "positive",
                        "description": "short gaps",
                    },
                    {
                        "id": "parameter:3f6cad53a748ba6f1e5fcb3bba6af4855f26600fd47a11ed30c47ddbe027fe98",
                        "owners": [{"kind": "indicator", "id": "indicator:153ac0200cf2403b6774"}],
                        "quantity": "obs_ordered_base",
                        "name": "obs_ordered_base_long_scale",
                        "role": "observation_hyperparameter",
                        "constraint": "none",
                        "description": "long base",
                    },
                    {
                        "id": "parameter:d494c90161198c85be93c1aeb380c0a8f915f9934825cb2deb9b18e8af1ae056",
                        "owners": [{"kind": "indicator", "id": "indicator:153ac0200cf2403b6774"}],
                        "quantity": "obs_ordered_gaps",
                        "name": "obs_ordered_gaps_long_scale",
                        "role": "observation_hyperparameter_positive",
                        "constraint": "positive",
                        "description": "long gaps",
                    },
                ],
            }
        )
        priors, bindings, _diagnostics = compile_priors(
            model_with_prior_payloads(
                statistical_model_spec,
                named_prior_payloads(
                    statistical_model_spec,
                    {
                        "obs_ordered_base_short_scale": {
                            "distribution": "Normal",
                            "params": {"mu": -1.0, "sigma": 0.5},
                        },
                        "obs_ordered_gaps_short_scale": {
                            "distribution": "HalfNormal",
                            "params": {"sigma": 2.0},
                        },
                        "obs_ordered_base_long_scale": {
                            "distribution": "Normal",
                            "params": {"mu": -3.0, "sigma": 1.0},
                        },
                        "obs_ordered_gaps_long_scale": {
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                        },
                    },
                ),
            ),
            spec,
        )

        binding_by_parameter = {
            binding.parameter_name: binding for binding in bindings.by_parameter.values()
        }
        assert binding_by_parameter["obs_ordered_base_short_scale"].flat_index == 0
        assert binding_by_parameter["obs_ordered_base_long_scale"].flat_index == 1
        assert (
            binding_by_parameter["obs_ordered_gaps_short_scale"].transform
            is PriorAuthoringTransform.SITE_ROW
        )
        assert (
            binding_by_parameter["obs_ordered_gaps_long_scale"].transform
            is PriorAuthoringTransform.SITE_ROW
        )

        base_prior = priors["obs_ordered_base"]
        np.testing.assert_allclose(base_prior.loc, [-1.0, -3.0])
        np.testing.assert_allclose(base_prior.scale, [0.5, 1.0])

        gap_prior = priors["obs_ordered_gaps"]
        gap_scales = np.asarray(gap_prior.scale).reshape(2, 8)
        np.testing.assert_allclose(gap_scales[0], 2.0)
        np.testing.assert_allclose(gap_scales[1], 0.5)

    def test_artifact_contains_compiled_prior_semantics(self, statistical_model_spec_and_priors):
        """compile_ssm_artifact emits semantics and omits legacy priors."""
        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact

        statistical_model_spec, priors = statistical_model_spec_and_priors
        artifact = compile_ssm_artifact(
            make_prior_model(statistical_model_spec, priors),
            _mood_structural_plan(),
        )
        assert not hasattr(artifact, "priors")
        assert artifact.edge_lag_days == []
        sem = artifact.compiled_prior_semantics
        assert sem.schema_version == 7
        assert sem.site_registry
        assert sem.priors

    def test_known_input_beta_binds_to_input_effect_site(self):
        """A beta from a known input compiles to B, not the latent dynamics matrix."""
        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact

        causal_design = {
            "latent": {
                "default_outcome": {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
                "constructs": [
                    {
                        "id": "construct:16176a18c25802dee8a1",
                        "name": "dose",
                        "description": "Dose",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                    },
                    {
                        "id": "construct:bbc87212909e45b9e6c3",
                        "name": "mood",
                        "description": "Mood",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                    },
                ],
                "edges": [
                    {
                        "cause_id": "construct:16176a18c25802dee8a1",
                        "effect_id": "construct:bbc87212909e45b9e6c3",
                        "id": "edge:3d7176b256a26799c7c4",
                        "description": "Dose affects mood",
                        "lagged": True,
                    }
                ],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "id": "indicator:5806a6a8417abd85897f",
                        "construct_id": "construct:16176a18c25802dee8a1",
                        "name": "dose_mg",
                        "construct_polarity": "positive",
                        "how_to_measure": "Dose in mg",
                        "measurement_dtype": "continuous",
                        "aggregation": "sum",
                    },
                    {
                        "id": "indicator:45f78731e3e0c6f3efe1",
                        "construct_id": "construct:bbc87212909e45b9e6c3",
                        "name": "mood_score",
                        "construct_polarity": "positive",
                        "how_to_measure": "Mood score",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                ],
            },
            "known_inputs": [
                {
                    "construct_id": "construct:16176a18c25802dee8a1",
                    "source_indicator_id": "indicator:5806a6a8417abd85897f",
                    "scale": 10.0,
                    "missing_policy": "forward_fill",
                }
            ],
        }
        statistical_model_spec = {
            "mechanisms": [
                {
                    "kind": "node_potential",
                    "target_id": "construct:bbc87212909e45b9e6c3",
                    "center": {"kind": "fixed", "value": 0},
                    "stiffness": {
                        "kind": "estimated",
                        "parameter_id": "parameter:fb33dbedf43eb15e324c86fa97201278e306cb48aa9752104361309d61215122",
                    },
                    "quartic": {"kind": "fixed", "value": 0},
                },
                {
                    "kind": "linear",
                    "edge_id": "edge:3d7176b256a26799c7c4",
                    "weight": {
                        "kind": "estimated",
                        "parameter_id": "parameter:53affe7fe8301a130af878d322911999e653c62dd0104d07b233ba7cf69c69af",
                    },
                },
            ],
            "likelihoods": [
                {
                    "indicator_id": "indicator:45f78731e3e0c6f3efe1",
                    "distribution": "gaussian",
                    "link": "identity",
                    "reasoning": "",
                }
            ],
            "parameters": [
                {
                    "prior_transform": "dt_persistence_to_ct_decay",
                    "id": "parameter:fb33dbedf43eb15e324c86fa97201278e306cb48aa9752104361309d61215122",
                    "owners": [{"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"}],
                    "quantity": "dynamics_decay",
                    "name": "rho_mood",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "description": "",
                },
                {
                    "prior_transform": "dt_effect_to_ct_rate",
                    "id": "parameter:53affe7fe8301a130af878d322911999e653c62dd0104d07b233ba7cf69c69af",
                    "owners": [
                        {"kind": "construct", "id": "construct:16176a18c25802dee8a1"},
                        {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
                        {"kind": "edge", "id": "edge:3d7176b256a26799c7c4"},
                    ],
                    "quantity": "input_effect",
                    "name": "beta_dose_mood",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "description": "",
                },
                {
                    "id": "parameter:146688c9f8e2c980c9e7963be61deb23225a81f828c204339c2164d1f51d441e",
                    "owners": [{"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"}],
                    "quantity": "diffusion_diag",
                    "name": "sigma_mood",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "description": "",
                },
            ],
        }
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_dose_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.1}},
            "sigma_mood": {"distribution": "HalfNormal", "params": {"sigma": 1.0}},
        }

        from nof1_causal_lab.models.structural import build_structural_plan

        structural_plan = build_structural_plan(CausalDesign.model_validate(causal_design))
        typed_statistical_model_spec = StatisticalModelSpec.model_validate(statistical_model_spec)
        artifact = compile_ssm_artifact(
            make_prior_model(typed_statistical_model_spec, priors),
            structural_plan,
        )

        assert artifact.spec.manifest_names == ["mood_score"]
        assert artifact.spec.input_names == ["dose"]
        assert artifact.spec.input_source_indicators == ["dose_mg"]
        assert artifact.spec.input_lagged == [True]
        assert artifact.spec.input_effect_block["free_support"] == [[True]]
        beta_binding = next(
            binding
            for binding in artifact.parameter_bindings
            if binding.parameter_id
            == next(p.id for p in artifact.parameters if p.name == "beta_dose_mood")
        )
        assert {
            "parameter_id": beta_binding.parameter_id,
            "site_name": beta_binding.site_name,
            "flat_index": beta_binding.flat_index,
        } == {
            "parameter_id": next(p.id for p in artifact.parameters if p.name == "beta_dose_mood"),
            "site_name": "input_effect_free",
            "flat_index": 0,
        }
        assert beta_binding.site_kind is SiteKind.INPUT_EFFECT
        assert beta_binding.transform is PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE

    def test_model_from_artifact_uses_semantics(self, statistical_model_spec_and_priors):
        """hydrate_compiled_model reads compiled_prior_semantics."""
        import polars as pl

        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
        from nof1_causal_lab.models.ssm.runtime import hydrate_compiled_model

        statistical_model_spec, priors = statistical_model_spec_and_priors
        artifact = compile_ssm_artifact(
            make_prior_model(statistical_model_spec, priors),
            _mood_structural_plan(),
        )
        model = hydrate_compiled_model(
            artifact,
            pl.DataFrame({"time": [0.0], "mood_score": [5.0]}),
        )
        assert model.priors is not None
        assert model.get_prior_runtime_bundle() is not None

    def test_model_from_artifact_requires_compiled_prior_semantics(
        self, statistical_model_spec_and_priors
    ):
        """Model rebuild fails clearly when compiled semantics are missing."""
        from pydantic import ValidationError

        from nof1_causal_lab.artifacts.compiled_ssm import CompiledSSMArtifact
        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact

        statistical_model_spec, priors = statistical_model_spec_and_priors
        artifact = compile_ssm_artifact(
            make_prior_model(statistical_model_spec, priors),
            _mood_structural_plan(),
        )
        payload = artifact.model_dump(mode="json")
        del payload["compiled_prior_semantics"]

        with pytest.raises(ValidationError, match="compiled_prior_semantics"):
            CompiledSSMArtifact.model_validate(payload)

    def test_compiled_artifact_requires_input_alignment_metadata(
        self, statistical_model_spec_and_priors
    ):
        from pydantic import ValidationError

        from nof1_causal_lab.artifacts.compiled_ssm import CompiledSSMArtifact
        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact

        statistical_model_spec, priors = statistical_model_spec_and_priors
        artifact = compile_ssm_artifact(
            make_prior_model(statistical_model_spec, priors),
            _mood_structural_plan(),
        )
        payload = artifact.model_dump(mode="json")
        del payload["structure"]["spec"]["input_lagged"]

        with pytest.raises(ValidationError, match="input_lagged"):
            CompiledSSMArtifact.model_validate(payload)

    @pytest.mark.cpu_expensive
    def test_end_to_end_compile_rebuild_sample(self, statistical_model_spec_and_priors):
        """Full roundtrip: compile → rebuild → sample."""
        import numpy as np
        import polars as pl

        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
        from nof1_causal_lab.models.ssm.runtime import (
            prepare_wide_model_runtime,
            sample_prior_predictive,
        )
        from nof1_causal_lab.utils.data import pivot_to_wide

        statistical_model_spec, priors = statistical_model_spec_and_priors
        artifact = compile_ssm_artifact(
            make_prior_model(statistical_model_spec, priors),
            _mood_structural_plan(),
        )

        rng = np.random.default_rng(42)
        n = 30
        data_for_model = pl.DataFrame(
            {
                "indicator": ["mood_score"] * n,
                "value": (rng.standard_normal(n) * 1.5 + 5).tolist(),
                "anchor_time": list(range(n)),
            }
        )
        runtime = prepare_wide_model_runtime(
            pivot_to_wide(data_for_model),
            compiled_ssm=artifact,
            sampler_config=cast(
                "SamplerConfigOverride",
                {"method": "marginal_particle_gibbs"},
            ),
        )
        samples = sample_prior_predictive(
            runtime.model,
            samples=5,
            times=runtime.times,
            observation_support=runtime.observation_support,
            observation_mask=~jnp.isnan(runtime.observations),
            transition_inputs=runtime.transition_inputs,
        )
        assert samples is not None

    @pytest.mark.cpu_expensive
    def test_compiled_model_prior_predictive(self, statistical_model_spec_and_priors):
        """Compiled models can sample prior predictive from artifact semantics."""
        import polars as pl

        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
        from nof1_causal_lab.models.ssm.runtime import (
            hydrate_compiled_model,
            sample_prior_predictive,
        )

        statistical_model_spec, priors = statistical_model_spec_and_priors
        artifact = compile_ssm_artifact(
            make_prior_model(statistical_model_spec, priors),
            _mood_structural_plan(),
        )
        model = hydrate_compiled_model(
            artifact,
            pl.DataFrame({"time": [0.0], "mood_score": [5.0]}),
        )
        samples = sample_prior_predictive(model, samples=4)
        assert "vf_0_decay" in samples
        assert samples["vf_0_decay"].shape[0] == 4
        assert "observations" in samples

    @pytest.mark.cpu_expensive
    def test_compiled_model_traces_vector_t0_prior_without_reconstructing(self):
        """Compiled models execute vector-valued positive priors via runtime semantics."""
        import polars as pl

        from nof1_causal_lab.artifacts.compiled_ssm import CompiledSSMArtifact, CompiledStructure
        from nof1_causal_lab.models.ssm.compile.artifact import serialize_ssm_spec
        from nof1_causal_lab.models.ssm.runtime import (
            hydrate_compiled_model,
            prepare_fit_inputs,
        )

        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            lambda_block=_lambda_block(2, 2, template=jnp.eye(2)),
            manifest_chol_block=_manifest_chol_block(
                2,
                diag_support=full_diagonal_support(2),
                template=jnp.zeros((2, 2)),
            ),
            manifest_names=["m0", "m1"],
        )
        priors = {
            "t0_var_diag_free": distribution_from_params(
                PriorDistributionFamily.HALF_NORMAL,
                {"sigma": [1.0, 2.0]},
            )
        }
        artifact = CompiledSSMArtifact.model_construct(
            observation_bindings={},
            parameters=[],
            auxiliary_coordinates=[],
            schema_version=2,
            structure=CompiledStructure(
                spec=serialize_ssm_spec(spec),
                edge_lag_days=[],
                bindings=[],
                anchor_certificates=[],
            ),
            compiled_prior_semantics=compile_prior_semantics(spec, priors),
            parameter_bindings=[],
            compile_diagnostics=[],
        )
        wide = pl.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "m0": [0.1, 0.2, 0.3],
                "m1": [0.4, 0.5, 0.6],
            }
        )

        model = hydrate_compiled_model(artifact, wide)
        observations, times, _manifest_names, _wide = prepare_fit_inputs(model.spec, wide)
        backend = get_laplace_backend(model, 6)
        trace = handlers.trace(handlers.seed(model.model, rng_seed=0)).get_trace(
            observations,
            times,
            likelihood_backend=backend,
        )

        assert trace["t0_var_diag_free"]["value"].shape == (2,)
