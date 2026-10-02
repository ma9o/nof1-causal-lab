"""Tests for the canonical site registry and compile-stable prior evaluation."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.parameters import sample_sites
from nof1_causal_lab.models.ssm.inference.utils import _discover_sites, _DummyLikelihoodBackend
from nof1_causal_lab.models.ssm.model import sample_parameters
from nof1_causal_lab.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
    process_sites,
    sample_prior_parameters,
)
from nof1_causal_lab.models.ssm.priors import resolve_site_priors
from tests.model_fixtures import (
    bind_panel_fixture,
    compile_fit_fixture,
    compile_model_fixture,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_spec():
    return ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "common/two_state_gaussian_model.json"
        ).read_text()
    )


@pytest.fixture
def simple_model(simple_spec):
    return compile_fit_fixture(simple_spec)


@pytest.fixture
def dag_spec():
    return ModelSpec.model_validate_json(
        (
            Path(__file__).resolve().parents[2]
            / "fixtures/models"
            / "parameterization/dag_spec_model.json"
        ).read_text()
    )


@pytest.fixture
def dag_model(dag_spec):
    return compile_fit_fixture(dag_spec)


# ---------------------------------------------------------------------------
# Site registry tests
# ---------------------------------------------------------------------------


def _assert_registry_matches_trace(registry, site_info):
    """Keep the registry/trace integration assertion owned by this test module."""
    assert {site.name for site in registry} == set(site_info)
    for site in registry:
        assert site.shape == site_info[site.name].shape


class TestSiteRegistry:
    @pytest.mark.inference(concern="sampling")
    def test_registry_names_match_trace(self, simple_model):
        """Registry produces the same site names as model tracing."""
        compiled = simple_model.compiled
        registry = build_site_registry(compiled)
        backend = _DummyLikelihoodBackend()
        T = 2
        obs = jnp.zeros((T, numeric.n_observations(compiled)))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(
            simple_model.prior_runtime_bundle,
            bind_panel_fixture(simple_model.compiled, obs, times),
            random.PRNGKey(0),
            backend,
        )
        _assert_registry_matches_trace(registry, site_info)

    @pytest.mark.inference(concern="sampling")
    def test_registry_names_match_trace_dag(self, dag_model):
        """Registry matches trace for DAG-constrained model with cint."""
        compiled = dag_model.compiled
        registry = build_site_registry(compiled)
        backend = _DummyLikelihoodBackend()
        T = 2
        obs = jnp.zeros((T, numeric.n_observations(compiled)))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(
            dag_model.prior_runtime_bundle,
            bind_panel_fixture(dag_model.compiled, obs, times),
            random.PRNGKey(0),
            backend,
        )
        _assert_registry_matches_trace(registry, site_info)

    @pytest.mark.inference(concern="sampling")
    def test_registry_shapes_match_trace_partial_manifest_variance_mask(self):
        """Masked manifest variance exposes only free diagonal entries as a site."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/partial_manifest_variance_model.json"
            ).read_text()
        )
        model = compile_fit_fixture(spec)
        registry = build_site_registry(compile_model_fixture(spec))
        backend = _DummyLikelihoodBackend()
        T = 2
        obs = jnp.zeros((T, numeric.n_observations(compile_model_fixture(spec))))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(
            model.prior_runtime_bundle,
            bind_panel_fixture(model.compiled, obs, times),
            random.PRNGKey(0),
            backend,
        )

        _assert_registry_matches_trace(registry, site_info)
        manifest_site = next(site for site in registry if site.name == "manifest_var_diag_free")
        assert manifest_site.shape == (1,)

    @pytest.mark.contract
    def test_fixed_dynamics_excludes_dynamics_sites(self):
        """When dynamics is a fixed array, no dynamics sites appear."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "common/two_state_fixed_drift_model.json"
            ).read_text()
        )
        registry = build_site_registry(compile_model_fixture(spec))
        assert len([site for site in registry if site.site_kind == SiteKind.DYNAMICS_DECAY]) == 2

    @pytest.mark.contract
    def test_diag_diffusion_excludes_lower(self):
        """Diagonal diffusion has no lower-triangle sites."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/testsiteregistry_test_diag_diffusion_excludes_lower__make_spec.json"
            ).read_text()
        )
        registry = build_site_registry(compile_model_fixture(spec))
        names = {s.name for s in registry}
        assert "diffusion_diag_free" in names
        assert "diffusion_lower_free" not in names

    @pytest.mark.contract
    def test_free_diffusion_includes_lower(self):
        """Free diffusion includes lower-triangle sites."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "common/two_state_gaussian_model.json"
            ).read_text()
        )
        registry = build_site_registry(compile_model_fixture(spec))
        names = {s.name for s in registry}
        assert "diffusion_diag_free" in names
        assert "diffusion_lower_free" in names

    @pytest.mark.contract
    def test_sparse_initial_state_correlations_only_include_authored_pairs(self):
        """Initial-state correlation sites should only exist for authored pairs."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 0] = True
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/testsiteregistry_test_sparse_initial_state_correlations_only_include_authored_pairs__make_spec.json"
            ).read_text()
        )
        registry = build_site_registry(compile_model_fixture(spec))
        site_map = {site.name: site for site in registry}
        assert site_map["t0_var_lower_free"].shape == (1,)

    @pytest.mark.contract
    def test_support_classes(self, simple_spec):
        """Check that support classes are correctly assigned."""
        registry = build_site_registry(compile_model_fixture(simple_spec))
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

    @pytest.mark.contract
    def test_mixed_diffusion_includes_proc_df_site(self):
        """Any student-t latent in diffusion_dists should expose proc_df."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/student_innovation_model.json"
            ).read_text()
        )
        registry = build_site_registry(compile_model_fixture(spec))
        assert "proc_df" in {site.name for site in registry}

    @pytest.mark.inference(concern="sampling")
    def test_mixed_diffusion_sampling_emits_proc_df(self):
        """The traced model should sample proc_df when diffusion_dists include student_t."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/student_innovation_model.json"
            ).read_text()
        )
        model = compile_fit_fixture(spec)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(
                lambda: sample_sites(
                    process_sites(model.compiled), model.prior_runtime_bundle.priors.__getitem__
                )
            ).get_trace()

        assert "proc_df" in trace

    @pytest.mark.inference(concern="sampling")
    def test_static_state_sd_site_is_registered_and_traced(self):
        """Compiled baseline factors should expose a positive static-state SD site."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/static_state_model.json"
            ).read_text()
        )
        model = compile_fit_fixture(spec)

        registry = build_site_registry(compile_model_fixture(spec))
        site_map = {site.name: site for site in registry}
        assert site_map["static_state_sd_free"].shape == (1,)
        assert site_map["static_state_sd_free"].support == SupportClass.POSITIVE

        backend = _DummyLikelihoodBackend()
        obs = jnp.zeros((5, numeric.n_observations(compile_model_fixture(spec))))
        times = jnp.arange(5, dtype=jnp.float32)
        site_info = _discover_sites(
            model.prior_runtime_bundle,
            bind_panel_fixture(model.compiled, obs, times),
            random.PRNGKey(0),
            backend,
        )
        _assert_registry_matches_trace(registry, site_info)
        assert site_info["static_state_sd_free"].shape == (1,)


@pytest.mark.inference(concern="sampling")
class TestSpecBlockAssembly:
    def test_assemble_t0_cov_adds_low_rank_baseline_factor_covariance(self):
        """Static baseline factors should add `B diag(tau^2) B^T` to the t0 covariance."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/static_state_model.json"
            ).read_text()
        )
        model = compile_fit_fixture(spec)
        values = {
            site.name: jnp.ones(site.shape)
            for block in numeric.parameter_blocks(compile_model_fixture(spec))
            for site in block.iter_sites()
        }
        values["static_state_sd_free"] = jnp.array([2.0])
        with handlers.substitute(data=values), handlers.trace() as trace:
            cov = sample_parameters(model.compiled, model.prior_runtime_bundle)["t0_cov"]
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
    @pytest.mark.contract
    def test_assemble_deterministics_from_registry_free_spec(self, simple_spec):
        """Registry-driven assembly builds the expected matrices."""
        samples = {
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[0.9, 1.1]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.zeros((1, 1), dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, compile_model_fixture(simple_spec))
        assert jnp.allclose(det["diffusion"][0], jnp.array([[0.4, 0.0], [0.25, 0.6]]))
        assert det["lambda"].shape == (1, 2, 2)
        assert jnp.allclose(det["manifest_cov"][0], jnp.diag(jnp.array([0.49, 0.64])))
        assert jnp.allclose(det["t0_means"][0], jnp.array([1.0, -1.0]))
        assert jnp.allclose(det["t0_cov"][0], jnp.diag(jnp.array([0.81, 1.21])))

    @pytest.mark.contract
    def test_missing_declared_free_value_is_rejected(self, simple_spec):
        with pytest.raises(KeyError, match="diffusion_diag_free"):
            assemble_deterministics_from_registry({}, compile_model_fixture(simple_spec), n_draws=2)

    @pytest.mark.contract
    def test_assemble_deterministics_from_registry_fixed_blocks(self):
        """Fixed spec matrices are broadcast without any sampled sites."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/testdeterministicassembly_test_assemble_deterministics_from_registry_fixed_blocks__make_spec.json"
            ).read_text()
        )
        det = assemble_deterministics_from_registry({}, compile_model_fixture(spec), n_draws=3)
        assert jnp.allclose(
            det["diffusion"],
            jnp.broadcast_to(compile_model_fixture(spec).diffusion_block.assemble(), (3, 2, 2)),
        )
        assert jnp.allclose(
            det["lambda"],
            jnp.broadcast_to(compile_model_fixture(spec).loading_block.assemble(), (3, 2, 2)),
        )
        manifest_chol = compile_model_fixture(spec).observation_noise_block.assemble()
        expected_manifest_cov = manifest_chol @ manifest_chol.T
        assert jnp.allclose(det["manifest_cov"], jnp.broadcast_to(expected_manifest_cov, (3, 2, 2)))
        assert isinstance(compile_model_fixture(spec).initial_mean_block.assemble(), jnp.ndarray)
        assert jnp.allclose(
            det["t0_means"],
            jnp.broadcast_to(compile_model_fixture(spec).initial_mean_block.assemble(), (3, 2)),
        )
        expected_t0_cov = compile_model_fixture(spec).initial_covariance_block.assemble_cov()
        assert jnp.allclose(det["t0_cov"], jnp.broadcast_to(expected_t0_cov, (3, 2, 2)))

    @pytest.mark.contract
    def test_assemble_deterministics_from_registry_partial_manifest_variance_mask(self):
        """Registry assembly respects mixed fixed/free manifest-noise diagonals."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/partial_manifest_variance_model.json"
            ).read_text()
        )
        samples = {
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.9]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[0.9, 1.1]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.zeros((1, 1), dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, compile_model_fixture(spec))
        assert jnp.allclose(det["manifest_cov"][0], jnp.diag(jnp.array([0.16, 0.81])))

    @pytest.mark.contract
    def test_assemble_deterministics_from_registry_initial_state_correlations(self):
        """Initial-state off-diagonal samples are interpreted as correlations."""
        mask = np.zeros((2, 2), dtype=bool)
        mask[1, 0] = True
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "common/two_state_gaussian_model.json"
            ).read_text()
        )
        samples = {
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[2.0, 3.0]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, compile_model_fixture(spec))

        assert jnp.allclose(
            det["t0_cov"][0],
            jnp.array([[4.0, 1.5], [1.5, 9.0]], dtype=jnp.float32),
        )

    @pytest.mark.inference(concern="sampling")
    def test_assemble_deterministics_repairs_invalid_initial_correlation_matrix(self):
        """Impossible authored initial correlations are repaired to a PSD covariance."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 0] = True
        mask[2, 0] = True
        mask[2, 1] = True
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/testdeterministicassembly_test_assemble_deterministics_repairs_invalid_initial_correlation_matrix__make_spec.json"
            ).read_text()
        )
        samples = {
            "diffusion_diag_free": jnp.array([[0.4, 0.6, 0.5]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25, 0.1, -0.15]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8, 0.9]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0, 0.5]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[1.0, 1.0, 1.0]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.array([[0.9, 0.9, -0.9]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, compile_model_fixture(spec))
        min_eig = jnp.min(jnp.linalg.eigvalsh(det["t0_cov"][0]))

        assert bool(jnp.isfinite(det["t0_cov"]).all())
        assert float(min_eig) > -1e-6

        model = compile_fit_fixture(spec)
        with handlers.substitute(data={name: value[0] for name, value in samples.items()}):
            trace = handlers.trace(sample_parameters).get_trace(
                model.compiled, model.prior_runtime_bundle
            )
        np.testing.assert_allclose(trace["t0_cov"]["value"], det["t0_cov"][0], atol=1e-6)
        factor = trace["t0_correlation_positive_definite"]
        assert float(factor["fn"].log_prob(factor["value"])) == pytest.approx(-800000.01, rel=1e-5)


# ---------------------------------------------------------------------------
# Prior runtime state
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestNativeRuntimePriors:
    def test_sites_have_native_distributions_with_the_declared_shapes(self, simple_spec):
        registry = build_site_registry(compile_model_fixture(simple_spec))
        priors = resolve_site_priors(registry)
        assert set(priors) == {site.name for site in registry}
        for site in registry:
            assert isinstance(priors[site.name], dist.Distribution)
            assert priors[site.name].batch_shape == site.shape

    def test_partial_native_overrides_preserve_other_defaults(self, simple_spec):
        registry = build_site_registry(compile_model_fixture(simple_spec))
        decay_site = next(
            site.name for site in registry if site.site_kind == SiteKind.DYNAMICS_DECAY
        )
        priors = resolve_site_priors(registry, {decay_site: dist.Gamma(4.0, 2.0)})
        np.testing.assert_allclose(priors[decay_site].mean, 2.0)
        assert priors["diffusion_diag_free"].batch_shape == (2,)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@pytest.mark.inference(concern="predictive")
class TestSampling:
    def test_sample_shapes_and_native_support(self, simple_spec):
        registry = build_site_registry(compile_model_fixture(simple_spec))
        state = resolve_site_priors(registry)
        samples = sample_prior_parameters(random.PRNGKey(0), registry, state, n_samples=4)
        assert set(samples) == {site.name for site in registry}
        for site in registry:
            assert samples[site.name].shape == (4, *site.shape)
            assert jnp.all(jnp.isfinite(samples[site.name]))
            if site.support == SupportClass.POSITIVE:
                assert jnp.all(samples[site.name] > 0)

    def test_site_streams_are_stable_under_registry_reordering(self, simple_spec):
        registry = build_site_registry(compile_model_fixture(simple_spec))
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

    @pytest.mark.contract
    def test_global_ordered_threshold_priors_are_not_authorable(self):

        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/testcompiledartifactintegration_test_global_ordered_threshold_priors_are_not_authorable__make_spec.json"
            ).read_text()
        )
        parameter = ParameterSpec(
            id="parameter:bd5e7c989f1fe5b6752e958831d287e988f8a43f9633886ba7281220ae168533",
            name="obs_ordered_base",
            description="Unbound threshold",
        )
        with pytest.raises(ValueError, match="not referenced by component slots"):
            spec.revised(parameters=(*spec.parameters, parameter))

    @pytest.mark.contract
    def test_ordered_threshold_priors_bind_per_manifest_component_and_row(self):
        from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors

        priors, bindings, _diagnostics = compile_priors(
            compile_model_fixture(
                ModelSpec.model_validate_json(
                    (
                        Path(__file__).resolve().parents[2]
                        / "fixtures/models"
                        / "parameterization/testcompiledartifactintegration_test_ordered_threshold_priors_bind_per_manifest_component_and_row_model_with_prior_payloads.json"
                    ).read_text()
                )
            ),
            ModelSpec.model_validate_json(
                (
                    Path(__file__).resolve().parents[2]
                    / "fixtures/models"
                    / "parameterization/testcompiledartifactintegration_test_ordered_threshold_priors_bind_per_manifest_component_and_row_model_with_prior_payloads.json"
                ).read_text()
            ),
        )

        binding_by_parameter = {
            binding.parameter_name: binding
            for binding in {binding.parameter_id: binding for binding in bindings}.values()
        }
        assert binding_by_parameter["obs_ordered_base_short_scale"].flat_index == 0
        assert binding_by_parameter["obs_ordered_base_long_scale"].flat_index == 1
        for row, name in enumerate(("obs_ordered_gaps_short_scale", "obs_ordered_gaps_long_scale")):
            assert {
                native.coordinate.indices[0]
                for native in binding_by_parameter[name].native_coordinates
            } == {row}

        base_prior = priors["obs_ordered_base"]
        np.testing.assert_allclose(base_prior.loc, [-1.0, -3.0])
        np.testing.assert_allclose(base_prior.scale, [0.5, 1.0])

        gap_prior = priors["obs_ordered_gaps"]
        gap_scales = np.asarray(gap_prior.scale).reshape(2, 8)
        np.testing.assert_allclose(gap_scales[0], 2.0)
        np.testing.assert_allclose(gap_scales[1], 0.5)

    @pytest.mark.contract
    def test_execution_checks_accept_complete_model(self):
        """A model with authored priors satisfies execution requirements."""

        compile_model_fixture(
            ModelSpec.model_validate_json(
                (
                    Path(__file__).resolve().parents[2]
                    / "fixtures/models"
                    / "parameterization/mood_model.json"
                ).read_text()
            )
        )

    @pytest.mark.inference(concern="sampling")
    def test_runtime_derives_the_authored_priors(self):

        definition = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "parameterization/mood_model.json"
            ).read_text()
        )
        model = compile_fit_fixture(definition)
        assert set(model.prior_runtime_bundle.priors) == {
            site.name for site in build_site_registry(compile_model_fixture(definition))
        }
