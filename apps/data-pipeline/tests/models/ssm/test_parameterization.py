"""Tests for the canonical site registry and compile-stable prior evaluation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.likelihood import LinkFunction
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform, SiteKind, SupportClass
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.distributions import (
    DistributionFamily,
)
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.compile.bindings import parameter_bindings
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.inference.backend_factory import get_laplace_backend
from nof1_causal_lab.models.ssm.inference.utils import _discover_sites
from nof1_causal_lab.models.ssm.model import SSMModel
from nof1_causal_lab.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
    sample_prior_parameters,
)
from nof1_causal_lab.models.ssm.priors import resolve_site_priors
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)
from tests.helpers import (
    complete_test_model,
    make_prior_model,
    model_with_prior_payloads,
    named_prior_payloads,
    native_axis_metadata,
)
from tests.model_fixtures import (
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
    model_fixture,
)
from tests.slot_fixtures import fixture_parameter_id

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
) -> ModelSpec:
    """Build an ModelSpec from explicit block specs for tests."""
    if dynamics_spec is None:
        dynamics_spec = full_dense_matrix_dynamics_spec(n_latent)
    return model_fixture(
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
def scientific_model_and_priors():
    return complete_test_model(_mood_structure()), {
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
    }


def _mood_structure() -> ModelSpec:

    return ModelSpec.model_validate(
        {
            "edges": [
                {
                    "id": "edge:test-outcome-0",
                    "cause": {
                        "id": "construct:bbc87212909e45b9e6c3",
                        "name": "mood",
                        "description": "Mood",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                        "indicators": [
                            {
                                "id": "indicator:45f78731e3e0c6f3efe1",
                                "name": "mood_score",
                                "how_to_measure": "Mood score",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                                "construct_polarity": "positive",
                            }
                        ],
                    },
                    "effect": {
                        "id": "construct:unmeasured_outcome",
                        "name": "unmeasured_outcome",
                        "description": "Downstream response outside the measured test states.",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                    },
                    "description": "Test state affects an unmeasured downstream response",
                }
            ],
            "measurement_clock": "1d",
        }
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
        obs = jnp.zeros((T, numeric.n_observations(spec)))
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
        obs = jnp.zeros((T, numeric.n_observations(spec)))
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
        obs = jnp.zeros((T, numeric.n_observations(spec)))
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
        obs = jnp.zeros((T, numeric.n_observations(spec)))
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
        assert len([site for site in registry if site.site_kind == SiteKind.DYNAMICS_DECAY]) == 2

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
            n_manifest=3,
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
        assert all(
            site.support == SupportClass.POSITIVE
            for site in registry
            if site.site_kind == SiteKind.DYNAMICS_DECAY
        )
        # POSITIVE support sites (HalfNormal priors)
        assert support_map["diffusion_diag_free"] == SupportClass.POSITIVE
        assert support_map["manifest_var_diag_free"] == SupportClass.POSITIVE
        assert support_map["t0_var_diag_free"] == SupportClass.POSITIVE

    def test_mixed_diffusion_includes_proc_df_site(self):
        """Any student-t latent in diffusion_dists should expose proc_df."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
            diffusion_dists=[DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T],
        )
        registry = build_site_registry(spec)
        assert "proc_df" in {site.name for site in registry}

    def test_mixed_diffusion_sampling_emits_proc_df(self):
        """The traced model should sample proc_df when diffusion_dists include student_t."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
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
            n_manifest=2,
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
        obs = jnp.zeros((5, numeric.n_observations(spec)))
        times = jnp.arange(5, dtype=jnp.float32)
        site_info = _discover_sites(model, obs, times, random.PRNGKey(0), backend)
        _assert_registry_matches_trace(registry, site_info)
        assert site_info["static_state_sd_free"]["shape"] == (1,)


class TestSpecBlockAssembly:
    def test_assemble_t0_cov_adds_low_rank_baseline_factor_covariance(self):
        """Static baseline factors should add `B diag(tau^2) B^T` to the t0 covariance."""
        spec = _make_spec(
            n_latent=2,
            n_manifest=2,
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
            for block in numeric.parameter_blocks(spec)
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
            jnp.broadcast_to(numeric.diffusion_block(spec).assemble(), (3, 2, 2)),
        )
        assert jnp.allclose(
            det["lambda"],
            jnp.broadcast_to(numeric.loading_block(spec).assemble(), (3, 2, 2)),
        )
        manifest_chol = numeric.observation_noise_block(spec).assemble()
        expected_manifest_cov = manifest_chol @ manifest_chol.T
        assert jnp.allclose(det["manifest_cov"], jnp.broadcast_to(expected_manifest_cov, (3, 2, 2)))
        assert isinstance(numeric.initial_mean_block(spec).assemble(), jnp.ndarray)
        assert jnp.allclose(
            det["t0_means"],
            jnp.broadcast_to(numeric.initial_mean_block(spec).assemble(), (3, 2)),
        )
        expected_t0_cov = numeric.initial_covariance_block(spec).assemble_cov()
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
        decay_site = next(
            site.name for site in registry if site.site_kind == SiteKind.DYNAMICS_DECAY
        )
        priors = resolve_site_priors(registry, {decay_site: dist.Gamma(4.0, 2.0)})
        np.testing.assert_allclose(priors[decay_site].mean, 2.0)
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


class TestCompiledArtifactIntegration:
    """Test that compiled_prior_semantics is emitted and correctly consumed."""

    def test_global_ordered_threshold_priors_are_not_authorable(self):

        spec = _make_spec(
            n_latent=1,
            n_manifest=1,
            latent_names=["burden"],
            dynamics_spec=DynamicsSpec(n_latent=1, components=()),
            manifest_names=["scale"],
            manifest_dists=[DistributionFamily.ORDERED_LOGISTIC],
            manifest_links=[LinkFunction.CUMULATIVE_LOGIT],
            manifest_level_counts=[4],
        )
        owners = (ConstructRef(id=spec.constructs[0].id),)
        parameter = ParameterSpec(
            id=fixture_parameter_id(SiteKind.OBS_ORDERED_BASE, owners),
            name="obs_ordered_base",
            description="Unbound threshold",
        )
        with pytest.raises(ValueError, match="not referenced by component slots"):
            spec.revised(parameters=(*spec.parameters, parameter))

    def test_ordered_threshold_priors_bind_per_manifest_component_and_row(self):
        from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors

        spec = _make_spec(
            n_latent=1,
            n_manifest=2,
            latent_names=["burden"],
            dynamics_spec=DynamicsSpec(n_latent=1, components=()),
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
        scientific_model = spec
        priors, bindings, _diagnostics = compile_priors(
            model_with_prior_payloads(
                scientific_model,
                named_prior_payloads(
                    scientific_model,
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
            )
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

    def test_execution_checks_return_anchoring_evidence(self, scientific_model_and_priors):
        """Execution checks derive evidence without creating a persisted artifact."""
        from nof1_causal_lab.models.model_checks import check_execution

        scientific_model, priors = scientific_model_and_priors
        artifact = check_execution(
            make_prior_model(scientific_model, priors),
        )
        assert artifact
        assert all(
            anchor.construct_id in {c.id for c in scientific_model.constructs}
            for anchor in artifact
        )

    def test_known_input_beta_binds_to_input_effect_site(self):
        """A beta from a known input compiles to B, not the latent dynamics matrix."""
        from nof1_causal_lab.models.model_checks import check_execution

        scientific_definition = {
            "default_outcome": {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
            "edges": [
                {
                    "cause": {
                        "id": "construct:16176a18c25802dee8a1",
                        "name": "dose",
                        "description": "Dose",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                        "indicators": [
                            {
                                "id": "indicator:5806a6a8417abd85897f",
                                "name": "dose_mg",
                                "construct_polarity": "positive",
                                "how_to_measure": "Dose in mg",
                                "measurement_dtype": "continuous",
                                "aggregation": "sum",
                            }
                        ],
                        "usage": {
                            "kind": "known_input",
                            "source_indicator_id": "indicator:5806a6a8417abd85897f",
                            "scale": 10.0,
                            "missing_policy": "forward_fill",
                        },
                    },
                    "effect": {
                        "id": "construct:bbc87212909e45b9e6c3",
                        "name": "mood",
                        "description": "Mood",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                        "indicators": [
                            {
                                "id": "indicator:45f78731e3e0c6f3efe1",
                                "name": "mood_score",
                                "construct_polarity": "positive",
                                "how_to_measure": "Mood score",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            }
                        ],
                    },
                    "id": "edge:3d7176b256a26799c7c4",
                    "description": "Dose affects mood",
                    "lagged": True,
                }
            ],
            "measurement_clock": "1d",
        }
        scientific_model = complete_test_model(ModelSpec.model_validate(scientific_definition))
        priors = {
            "rho_mood": {"distribution": "Beta", "params": {"alpha": 2.0, "beta": 2.0}},
            "beta_dose_mood": {"distribution": "Normal", "params": {"mu": 0.3, "sigma": 0.1}},
            "sigma_mood": {"distribution": "HalfNormal", "params": {"sigma": 1.0}},
        }

        ModelSpec.model_validate(scientific_definition)
        typed_scientific_model = ModelSpec.model_validate(scientific_model)
        check_execution(
            make_prior_model(typed_scientific_model, priors),
        )

        assert numeric.observation_names(typed_scientific_model) == ["mood_score"]
        assert numeric.input_names(typed_scientific_model) == ["dose"]
        assert numeric.input_sources(typed_scientific_model) == ["dose_mg"]
        assert numeric.input_lagged(typed_scientific_model) == [True]
        assert numeric.input_effect_block(typed_scientific_model).free_support.tolist() == [[True]]
        beta_binding = next(
            binding
            for binding in parameter_bindings(make_prior_model(typed_scientific_model, priors))[0]
            if binding.parameter_id
            == next(p.id for p in typed_scientific_model.parameters if p.name == "beta_dose_mood")
        )
        assert {
            "parameter_id": beta_binding.parameter_id,
            "site_name": beta_binding.site_name,
            "flat_index": beta_binding.flat_index,
        } == {
            "parameter_id": next(
                p.id for p in typed_scientific_model.parameters if p.name == "beta_dose_mood"
            ),
            "site_name": "input_effect_free",
            "flat_index": 0,
        }
        assert beta_binding.site_kind is SiteKind.INPUT_EFFECT
        assert beta_binding.transform is PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE

    def test_runtime_derives_the_authored_priors(self, scientific_model_and_priors):
        import polars as pl

        from nof1_causal_lab.models.ssm.runtime import build_ssm_model

        scientific_model, priors = scientific_model_and_priors
        definition = make_prior_model(scientific_model, priors)
        model = build_ssm_model(
            pl.DataFrame({"time": [0.0], "mood_score": [5.0]}), model_spec=definition
        )
        assert model.spec is definition
        assert set(model.get_prior_runtime_bundle().priors) == {
            site.name for site in build_site_registry(definition)
        }

    def test_readiness_rejects_a_second_model_definition(self, scientific_model_and_priors):
        from nof1_causal_lab.artifacts.execution import ExecutionReadiness

        scientific_model, priors = scientific_model_and_priors
        definition = make_prior_model(scientific_model, priors)
        with pytest.raises(ValueError, match="Extra inputs"):
            ExecutionReadiness.model_validate(
                {**definition.execution_readiness.model_dump(), "spec": definition.model_dump()}
            )
