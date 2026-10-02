"""Causal structure, measurement masks, and authored priors in executable models.

Parameter traces use the prior-only backend; likelihood numerics are exercised
by the inference tests.
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.handlers as handlers
import polars as pl
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import coefficient
from nof1_causal_lab.artifacts.identity import ConstructId
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.distributions import PriorDistributionFamily
from nof1_causal_lab.models.model_structure import validate_execution_structure
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.inference.utils import _DummyLikelihoodBackend
from nof1_causal_lab.models.ssm.model import numpyro_model
from nof1_causal_lab.models.ssm.parameterization import (
    build_site_registry,
)
from nof1_causal_lab.models.ssm.priors import resolve_site_priors
from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor
from nof1_causal_lab.prior_distributions import distribution_from_params
from tests.model_fixtures import (
    bind_panel_fixture,
    compile_fit_fixture,
    compile_model_fixture,
)

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Fix 1: DAG-constrained dynamics
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.inference(concern="sampling")
class TestDynamicsMask:
    """Test that component translation constrains linear edge sampling."""

    def test_dynamics_support_zeros_non_edges(self):
        """Dynamics entries where mask is False should be zero."""
        offdiag_support = np.zeros((3, 3), dtype=bool)
        offdiag_support[1, 0] = True  # X→Y
        offdiag_support[2, 1] = True  # Y→Z

        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/testdynamicsmask_test_dynamics_support_zeros_non_edges__make_3latent_spec.json"
            ).read_text()
        )
        model = compile_fit_fixture(spec)

        rng = random.PRNGKey(42)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(model.compiled, jnp.zeros((2, 4)), jnp.arange(2, dtype=jnp.float32)),
            priors=model.prior_runtime_bundle,
            likelihood_backend=_DummyLikelihoodBackend(),
        )

        weight_sites = [
            site
            for site in build_site_registry(compile_model_fixture(spec))
            if site.site_kind == SiteKind.DYNAMICS_WEIGHT
        ]
        assert len(weight_sites) == 2
        assert all(site.name in trace for site in weight_sites)
        assert {site.positions[0] for site in weight_sites} == {
            (1, 0),
            (2, 1),
        }

    def test_no_mask_fully_free(self):
        """Default dynamics mask expands to a fully free dynamics structure."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/three_latent_unmasked.json"
            ).read_text()
        )
        model = compile_fit_fixture(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(model.compiled, jnp.zeros((2, 4)), jnp.arange(2, dtype=jnp.float32)),
            priors=model.prior_runtime_bundle,
            likelihood_backend=_DummyLikelihoodBackend(),
        )

        weight_sites = [
            site
            for site in build_site_registry(compile_model_fixture(spec))
            if site.site_kind == SiteKind.DYNAMICS_WEIGHT
        ]
        assert len(weight_sites) == 6
        assert all(site.name in trace for site in weight_sites)

    def test_dynamics_support_single_latent(self):
        """Single latent: no off-diagonal, mask should be identity."""
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "common/one_state_gaussian_model.json"
            ).read_text()
        )
        model = compile_fit_fixture(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(model.compiled, jnp.zeros((2, 1)), jnp.arange(2, dtype=jnp.float32)),
            priors=model.prior_runtime_bundle,
            likelihood_backend=_DummyLikelihoodBackend(),
        )

        dynamics_sites = [
            site
            for site in build_site_registry(compile_model_fixture(spec))
            if site.assembly_group == "dynamics"
        ]
        assert [site.site_kind for site in dynamics_sites] == [SiteKind.DYNAMICS_DECAY]
        assert dynamics_sites[0].name in trace


# ═══════════════════════════════════════════════════════════════════════
# Fix 2: Structured lambda
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.inference(concern="sampling")
class TestLambdaMask:
    """Test that lambda_support constrains factor loadings."""

    def test_lambda_template_plus_mask(self):
        """Template+mask mode: fixed reference + free additional loadings."""
        # X has 2 indicators (x1 ref, x2 free), Y has 1, Z has 1

        lambda_support = np.zeros((4, 3), dtype=bool)
        lambda_support[1, 0] = True  # x2→X (free)

        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/testlambdamask_test_lambda_template_plus_mask__make_3latent_spec.json"
            ).read_text()
        )
        model = compile_fit_fixture(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(model.compiled, jnp.zeros((2, 4)), jnp.arange(2, dtype=jnp.float32)),
            priors=model.prior_runtime_bundle,
            likelihood_backend=_DummyLikelihoodBackend(),
        )

        # Only 1 free loading sampled
        assert trace["lambda_free"]["value"].shape == (1,)

        # Check the assembled lambda
        lam = trace["lambda"]["value"]
        assert float(lam[0, 0]) == 1.0  # Fixed reference
        assert float(lam[2, 1]) == 1.0  # Fixed reference
        assert float(lam[3, 2]) == 1.0  # Fixed reference
        assert float(lam[1, 0]) != 0.0  # Free loading was sampled

    def test_lambda_no_mask_returns_fixed(self):
        """Array lambda_mat with default zero free-mask is returned as-is."""
        # Both x indicators belong to X; fixed loadings must follow that ownership.
        lambda_mat = jnp.array(
            [[1.0, 0.0, 0.0], [0.75, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/testlambdamask_test_lambda_no_mask_returns_fixed__make_3latent_spec.json"
            ).read_text()
        )
        model = compile_fit_fixture(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(model.compiled, jnp.zeros((2, 4)), jnp.arange(2, dtype=jnp.float32)),
            priors=model.prior_runtime_bundle,
            likelihood_backend=_DummyLikelihoodBackend(),
        )

        # No lambda_free sampled
        assert "lambda_free" not in trace
        # Lambda deterministic IS emitted (the block always emits its
        # assembled output, fixed or sampled); the value equals the template.
        assert "lambda" in trace
        np.testing.assert_allclose(np.asarray(trace["lambda"]["value"]), np.asarray(lambda_mat))


# ═══════════════════════════════════════════════════════════════════════
# Fix 3: Per-element priors
# ═══════════════════════════════════════════════════════════════════════


class TestPerElementPriors:
    """Test array-valued priors through canonical site runtime state."""

    @staticmethod
    def _real_site(shape: tuple[int, ...]) -> SiteDescriptor:
        return SiteDescriptor(
            name="test_site",
            shape=shape,
            support=SupportClass.REAL,
            assembly_group="test",
            site_kind=SiteKind.DYNAMICS_WEIGHT,
        )

    @pytest.mark.contract
    def test_make_prior_dist_scalar(self):
        """Scalar mu/sigma produces scalar Normal."""
        site = self._real_site(())
        priors = {
            "test_site": distribution_from_params(
                PriorDistributionFamily.NORMAL, {"mu": 0.0, "sigma": 1.0}
            )
        }
        state = resolve_site_priors([site], priors)
        d = state[site.name]
        assert d.batch_shape == ()

    @pytest.mark.contract
    def test_make_prior_dist_array(self):
        """Array mu/sigma produces batched Normal."""
        site = self._real_site((3,))
        priors = {
            "test_site": distribution_from_params(
                PriorDistributionFamily.NORMAL,
                {"mu": [0.1, 0.2, 0.3], "sigma": [1.0, 0.5, 0.3]},
            )
        }
        state = resolve_site_priors([site], priors)
        d = state[site.name]
        assert d.batch_shape == (3,)

    @pytest.mark.contract
    def test_make_prior_batch_scalar_expand(self):
        """Scalar prior expanded to batch shape."""
        site = self._real_site((5,))
        priors = {
            "test_site": distribution_from_params(
                PriorDistributionFamily.NORMAL, {"mu": 0.0, "sigma": 1.0}
            )
        }
        state = resolve_site_priors([site], priors)
        d = state[site.name]
        assert d.batch_shape == (5,)

    @pytest.mark.contract
    def test_make_prior_batch_array_passthrough(self):
        """Array prior with correct shape passes through."""
        site = self._real_site((2,))
        priors = {
            "test_site": distribution_from_params(
                PriorDistributionFamily.NORMAL,
                {"mu": [0.1, 0.2], "sigma": [1.0, 0.5]},
            )
        }
        state = resolve_site_priors([site], priors)
        d = state[site.name]
        assert d.batch_shape == (2,)

    @pytest.mark.contract
    def test_make_prior_batch_mismatch_raises(self):
        """Array prior with wrong shape raises."""
        site = self._real_site((3,))
        priors = {
            "test_site": distribution_from_params(
                PriorDistributionFamily.NORMAL,
                {"mu": [0.1, 0.2], "sigma": [1.0, 0.5]},
            )
        }
        with pytest.raises(ValueError, match="broadcast"):
            resolve_site_priors([site], priors)

    @pytest.mark.inference(concern="sampling")
    def test_per_element_prior_in_model(self):
        """Per-element dynamics priors are used in sampling."""
        offdiag_support = np.zeros((2, 2), dtype=bool)
        offdiag_support[1, 0] = True  # X→Y

        # Per-element prior: single off-diagonal has mu=2.0

        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/testperelementpriors_test_per_element_prior_in_model_with_parameter_distributions.json"
            ).read_text()
        )
        model = compile_fit_fixture(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(model.compiled, jnp.zeros((2, 2)), jnp.arange(2, dtype=jnp.float32)),
            priors=model.prior_runtime_bundle,
            likelihood_backend=_DummyLikelihoodBackend(),
        )

        # Verify the authored law, independently of one lucky prior draw.
        prior = trace["vf_2_p0"]["fn"]
        assert float(prior.mean) == pytest.approx(2.0)
        assert float(prior.variance) == pytest.approx(0.01)


# ═══════════════════════════════════════════════════════════════════════
# Runtime structural-support construction
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.contract
class TestRuntimeStructuralSupport:
    """Test that compilation constructs correct block support from ModelSpec."""

    def test_compiled_mechanisms_and_loadings_preserve_structural_coordinates(self):
        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models/dag_to_ssm/three_latent_unmasked.json"
            ).read_text()
        )
        compiled = compile_model_fixture(model)
        components = compiled.dynamics.spec.components
        assert {
            (component.source, component.target) for component in components if component.edge_owned
        } == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}
        assert {component.target for component in components if not component.edge_owned} == {
            0,
            1,
            2,
        }
        assert compiled.loading_block.free_support.shape == (4, 3)
        np.testing.assert_array_equal(compiled.loading_block.template[[0, 2, 3], [0, 1, 2]], 1.0)

    def test_model_build_accepts_only_already_compiled_ssm_spec(self):
        """Runtime construction consumes an ModelSpec without structural authoring inputs."""

        pl.DataFrame(
            {
                "time": list(range(5)),
                "x1": [1.0] * 5,
                "x2": [2.0] * 5,
                "y1": [3.0] * 5,
                "z1": [4.0] * 5,
            }
        )

        model = compile_fit_fixture(
            ModelSpec.model_validate_json(
                (
                    Path(__file__).resolve().parents[2]
                    / "fixtures/models"
                    / "dag_to_ssm/three_latent_unmasked.json"
                ).read_text()
            )
        )
        assert numeric.n_states(model.compiled) == 3

    @pytest.mark.parametrize(
        ("source_count", "complete_test_model_payload"),
        [
            pytest.param(
                1,
                "dag_to_ssm/static_baseline_stress_sleep.json",
                id="1",
            ),
            pytest.param(
                2,
                "dag_to_ssm/static_baseline_stress_sleep.json",
                id="2",
            ),
        ],
    )
    def test_translate_spec_compiles_static_baseline_factor_from_induced_dependency(
        self, source_count, complete_test_model_payload
    ):

        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / complete_test_model_payload
            ).read_text()
        )
        if source_count == 2:
            source = model.get_construct(ConstructId("construct:766f6091724c163a3404"))
            second = source.revised(id="construct:second-common-cause", name="second_common_cause")
            model = model.revised(
                edges=replace_constructs(
                    (
                        *model.edges,
                        *(
                            edge.revised(id=f"{edge.id}-second", cause=second)
                            for edge in model.edges
                        ),
                    ),
                    (*model.constructs, second),
                )
            )
            # Equivalent marginalized roots reference one aggregate scale, not two draws.
            assert second.coefficient("initial_scale") == source.coefficient("initial_scale")
        compile_model_fixture(model)
        validate_execution_structure(model)
        spec = model
        np.testing.assert_array_equal(
            compile_model_fixture(spec).static_scale_block.free_support, [True]
        )
        np.testing.assert_allclose(
            compile_model_fixture(spec).static_scale_block.template, np.zeros(1)
        )
        np.testing.assert_allclose(
            compile_model_fixture(spec).static_factor_loadings, [[1.0], [1.0]]
        )
        assert numeric.static_factor_names(compile_model_fixture(spec)) == ("tau_u_shared",)
        np.testing.assert_array_equal(
            compile_model_fixture(spec).initial_covariance_block.correlation_support,
            np.zeros((2, 2), dtype=bool),
        )
        np.testing.assert_array_equal(
            compile_model_fixture(spec).initial_mean_block.free_support, [False, False]
        )
        np.testing.assert_array_equal(
            compile_model_fixture(spec).initial_covariance_block.diag_support,
            [False, False],
        )

    def test_translate_spec_marks_standardizable_gaussian_mean_indicators(self):

        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/x_z_measurements.json"
            ).read_text()
        )
        spec = model
        assert numeric.observation_standardized(compile_model_fixture(spec)) == (
            True,
            True,
            True,
            True,
        )

    def test_translate_spec_fixes_manifest_noise_for_single_indicator_constructs(self):

        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/x_z_measurements.json"
            ).read_text()
        )
        spec = model
        assert isinstance(compile_model_fixture(spec).observation_noise_block.template, jnp.ndarray)
        np.testing.assert_array_equal(
            compile_model_fixture(spec).observation_noise_block.diag_support,
            [True, True, False, False],
        )
        np.testing.assert_allclose(
            compile_model_fixture(spec).observation_noise_block.template, np.zeros((4, 4))
        )

    def test_translate_spec_rejects_initial_state_correlation_parameters_with_scientific_model(
        self,
    ):

        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/testruntimestructuralsupport_test_translate_spec_rejects_initial_state_correlation_parameters_with_scientific_model_attach_test_coefficients.json"
            ).read_text()
        )
        with pytest.raises(
            ValueError, match=r"explicit latent confounder|two distinct state owners"
        ):
            compile_model_fixture(model)

    def test_translate_spec_rejects_self_initial_state_correlation_with_scientific_model(self):
        from nof1_causal_lab.artifacts.parameter_spec import (
            InitialCorrelationTransformSpec,
            ParameterSpec,
        )

        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/x_z_measurements.json"
            ).read_text()
        )
        parameter = ParameterSpec(
            id="parameter:77f3ac548e5c828bd95649d677ae53ce71c8dd33c2b1526961592ea56d79f013",
            name="cor0",
            description="Unsupported pairwise initial correlation",
            transform=InitialCorrelationTransformSpec(),
        )

        with pytest.raises(ValueError, match="Joint coefficients require one other construct"):
            model.constructs[0].revised(
                coefficients=(
                    *model.constructs[0].coefficients,
                    coefficient(
                        parameter.id,
                        "initial_correlation",
                        construct_ids=(model.constructs[0].id,),
                    ),
                )
            )

    def test_model_build_end_to_end(self):

        science = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/x_z_measurements.json"
            ).read_text()
        )
        pl.DataFrame(
            {
                "time": list(range(10)),
                "x1": [1.0] * 10,
                "x2": [2.0] * 10,
                "y1": [3.0] * 10,
                "z1": [4.0] * 10,
            }
        )
        compile_model_fixture(science)
        spec = science
        assert (
            sum(
                site.site_kind == SiteKind.DYNAMICS_DECAY
                for site in compile_model_fixture(spec).site_registry
            )
            == 3
        )
        assert compile_model_fixture(spec).loading_block.free_support is not None
        assert numeric.n_states(compile_model_fixture(spec)) == 3
        assert numeric.n_observations(compile_model_fixture(spec)) == 4


# ═══════════════════════════════════════════════════════════════════════
# Site-registry mask awareness
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.contract
class TestSiteRegistryMasks:
    """Test that the canonical site registry respects SSM masks."""

    def test_site_registry_with_dynamics_support(self):
        """Site registry should size masked dynamics entries correctly."""
        from nof1_causal_lab.models.ssm.parameterization import build_site_registry

        # 3 latent, X→Y and Y→Z = 2 off-diagonal entries
        offdiag_support = np.zeros((3, 3), dtype=bool)
        offdiag_support[1, 0] = True
        offdiag_support[2, 1] = True

        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/testsiteregistrymasks_test_site_registry_with_dynamics_support_model_fixture.json"
            ).read_text()
        )

        registry = {site.name: site for site in build_site_registry(compile_model_fixture(spec))}

        weight_sites = sorted(
            site.name for site in registry.values() if site.site_kind == SiteKind.DYNAMICS_WEIGHT
        )
        assert len(weight_sites) == 2
        assert {
            site.positions[0]
            for site in build_site_registry(compile_model_fixture(spec))
            if site.name in weight_sites
        } == {
            (1, 0),
            (2, 1),
        }
        assert [
            site.shape for site in registry.values() if site.site_kind == SiteKind.DYNAMICS_DECAY
        ] == [
            (),
            (),
            (),
        ]

    def test_site_registry_with_lambda_support(self):
        """Site registry should size masked loading entries correctly."""
        from nof1_causal_lab.models.ssm.parameterization import build_site_registry

        spec = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/testsiteregistrymasks_test_site_registry_with_lambda_support_model_fixture.json"
            ).read_text()
        )

        registry = {site.name: site for site in build_site_registry(compile_model_fixture(spec))}
        assert registry["lambda_free"].shape == (1,)


# ═══════════════════════════════════════════════════════════════════════
# Integration: trace verification
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Gradual-build surface: self-limiting quartic + Hill (saturating) edges
# ═══════════════════════════════════════════════════════════════════════


class TestGradualBuildComponents:
    """Quartic self-limitation and Hill edges materialize from the ModelSpec."""

    @pytest.mark.contract
    def test_quartic_freed_only_for_self_limiting_construct(self):

        from nof1_causal_lab.artifacts.expressions import expression_coefficients

        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/testgradualbuildcomponents_test_quartic_freed_only_for_self_limiting_construct_complete_test_model.json"
            ).read_text()
        )
        quartics = {
            component.target: next(
                operand.value
                for operand in expression_coefficients(component.expression)
                if operand.role == "quartic"
            )
            for component in compile_model_fixture(model).dynamics.spec.components
            if not component.edge_owned
        }
        assert isinstance(quartics[1], str)
        assert quartics[0] == 0
        assert quartics[2] == 0

    @pytest.mark.contract
    def test_hill_edge_emitted_for_saturating_edge(self):
        from nof1_causal_lab.artifacts.expressions import hill_applications

        model = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/testgradualbuildcomponents_test_hill_edge_emitted_for_saturating_edge_complete_test_model.json"
            ).read_text()
        )
        edge_components = [
            item
            for item in compile_model_fixture(model).dynamics.spec.components
            if item.edge_owned
        ]
        hill = [item for item in edge_components if any(hill_applications(item.expression))]
        linear = [item for item in edge_components if not any(hill_applications(item.expression))]
        assert [(item.source, item.target) for item in hill] == [(0, 1)]
        assert (1, 2) in [(item.source, item.target) for item in linear]
        assert (0, 1) not in [(item.source, item.target) for item in linear]

    @pytest.mark.inference(concern="sampling")
    def test_freed_quartic_and_hill_sites_sample_finite(self):

        science = ModelSpec.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/models"
                / "dag_to_ssm/testgradualbuildcomponents_test_freed_quartic_and_hill_sites_sample_finite_complete_test_model.json"
            ).read_text()
        )
        spec = science
        model = compile_fit_fixture(spec)
        trace = handlers.trace(handlers.seed(numpyro_model, random.PRNGKey(0))).get_trace(
            bind_panel_fixture(model.compiled, jnp.zeros((2, 4)), jnp.arange(2, dtype=jnp.float32)),
            priors=model.prior_runtime_bundle,
            likelihood_backend=_DummyLikelihoodBackend(),
        )
        registry = build_site_registry(compile_model_fixture(spec))
        quartic_sites = [
            site.name for site in registry if site.site_kind == SiteKind.DYNAMICS_POTENTIAL_QUARTIC
        ]
        emax_sites = [site.name for site in registry if site.site_kind == SiteKind.HILL_EMAX]
        assert len(quartic_sites) == 1
        assert len(emax_sites) == 1
        for name in quartic_sites + emax_sites:
            assert bool(jnp.all(jnp.isfinite(trace[name]["value"])))
