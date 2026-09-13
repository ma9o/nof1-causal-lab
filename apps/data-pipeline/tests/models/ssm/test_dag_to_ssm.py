"""Tests for DAG-to-SSM constraint propagation (Fixes 1-3).

Tests that:
1. dynamics_support constrains off-diagonal sampling to causal edges only
2. lambda_support + template constrains factor loadings to measurement structure
3. Per-element priors align with mask positions
4. Builder constructs structural support from CausalDesign
5. Pipeline threading passes causal_design through
"""

from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.handlers as handlers
import polars as pl
import pytest
from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, IndicatorRef
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanism
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform, SiteKind, SupportClass
from nof1_causal_lab.artifacts.statistical_model_spec import LinkFunction
from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
from nof1_causal_lab.distributions import DistributionFamily, PriorDistributionFamily
from nof1_causal_lab.models.ssm.inference.backend_factory import get_laplace_backend
from nof1_causal_lab.models.ssm.model import SSMModel, SSMSpec
from nof1_causal_lab.models.ssm.parameterization import (
    SiteDescriptor,
)
from nof1_causal_lab.models.ssm.priors import resolve_site_priors
from nof1_causal_lab.models.ssm.structure import Fixed, Free, SparseMatrixBlockSpec
from nof1_causal_lab.prior_distributions import distribution_from_params
from tests.helpers import declare_test_dynamics, fixture_entity_id
from tests.ssm_spec_fixtures import (
    block_ssm_spec,
    dense_matrix_dynamics_spec,
    full_vector_support,
    zero_loading_support,
)

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


def _make_3latent_spec(
    edge_support: np.ndarray | None = None,
    lambda_block: SparseMatrixBlockSpec | None = None,
) -> SSMSpec:
    """3 latent, 4 manifest spec with optional masks."""
    n_l, n_m = 3, 4
    if edge_support is None:
        edge_support = np.ones((n_l, n_l), dtype=bool)
        np.fill_diagonal(edge_support, False)
    if lambda_block is None:
        lambda_block = SparseMatrixBlockSpec(
            n_rows=n_m,
            n_cols=n_l,
            free_support=zero_loading_support(n_m, n_l),
            template=jnp.eye(n_m, n_l),
            free_site_name="lambda_free",
            det_site_name="lambda",
            support=SupportClass.REAL,
            site_kind=SiteKind.LOADING,
            assembly_group="lambda",
            fixed_spec_field="lambda_mat",
            priors_field="lambda_free",
        )
    return block_ssm_spec(
        n_latent=n_l,
        n_manifest=n_m,
        dynamics_spec=dense_matrix_dynamics_spec(
            n_latent=n_l,
            decay_support=np.ones(n_l, dtype=bool),
            edge_support=edge_support,
            coupling_template=jnp.zeros((n_l, n_l)),
            intercept_support=np.zeros(n_l, dtype=bool),
            cint_template=jnp.zeros(n_l),
        ),
        lambda_block=lambda_block,
        latent_names=["X", "Y", "Z"],
        manifest_names=["x1", "x2", "y1", "z1"],
    )


def _make_causal_design_dict() -> dict[str, Any]:
    """Minimal CausalDesign dict: X→Y, Y→Z, 4 indicators."""
    return {
        "latent": {
            "default_outcome": {"kind": "construct", "id": "construct:d90c52e59b79004188dc"},
            "constructs": [
                {
                    "id": "construct:311c9047b5ede16a8f26",
                    "name": "X",
                    "description": "Cause",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "id": "construct:d90c52e59b79004188dc",
                    "name": "Y",
                    "description": "Mediator",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
                {
                    "id": "construct:a6b7873d58dac1ff1a02",
                    "name": "Z",
                    "description": "Downstream",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                },
            ],
            "edges": [
                {
                    "cause_id": "construct:311c9047b5ede16a8f26",
                    "effect_id": "construct:d90c52e59b79004188dc",
                    "id": "edge:39ba80b774e02c409662",
                    "description": "X causes Y",
                    "lagged": True,
                },
                {
                    "cause_id": "construct:d90c52e59b79004188dc",
                    "effect_id": "construct:a6b7873d58dac1ff1a02",
                    "id": "edge:57072ee1d1b7b3e7c0de",
                    "description": "Y causes Z",
                    "lagged": True,
                },
            ],
        },
        "measurement": {
            "model_clock": "1d",
            "indicators": [
                {
                    "id": "indicator:0f93ce57e1f1d1c96f5c",
                    "construct_id": "construct:311c9047b5ede16a8f26",
                    "name": "x1",
                    "construct_polarity": "positive",
                    "how_to_measure": "measure x",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "id": "indicator:27a6125b251378d8dd23",
                    "construct_id": "construct:311c9047b5ede16a8f26",
                    "name": "x2",
                    "construct_polarity": "positive",
                    "how_to_measure": "measure x alt",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "id": "indicator:dec7b4916899d2109674",
                    "construct_id": "construct:d90c52e59b79004188dc",
                    "name": "y1",
                    "construct_polarity": "positive",
                    "how_to_measure": "measure y",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
                {
                    "id": "indicator:c26d752dbca8b5a287ac",
                    "construct_id": "construct:a6b7873d58dac1ff1a02",
                    "name": "z1",
                    "construct_polarity": "positive",
                    "how_to_measure": "measure z",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                },
            ],
        },
        "estimation": {
            "state_order": ["X", "Y", "Z"],
            "edges": [
                {
                    "cause_id": "construct:311c9047b5ede16a8f26",
                    "effect_id": "construct:d90c52e59b79004188dc",
                    "id": "edge:39ba80b774e02c409662",
                    "description": "X causes Y",
                    "lagged": True,
                },
                {
                    "cause_id": "construct:d90c52e59b79004188dc",
                    "effect_id": "construct:a6b7873d58dac1ff1a02",
                    "id": "edge:57072ee1d1b7b3e7c0de",
                    "description": "Y causes Z",
                    "lagged": True,
                },
            ],
            "induced_dependencies": [],
        },
    }


def _make_structural_plan() -> StructuralPlan:
    """Compile the shared causal fixture to the executable structural artifact."""
    from nof1_causal_lab.models.structural import build_structural_plan

    return build_structural_plan(CausalDesign.model_validate(_make_causal_design_dict()))


# ═══════════════════════════════════════════════════════════════════════
# Fix 1: DAG-constrained dynamics
# ═══════════════════════════════════════════════════════════════════════


class TestDynamicsMask:
    """Test that component translation constrains linear edge sampling."""

    @pytest.mark.cpu_expensive
    def test_dynamics_support_zeros_non_edges(self):
        """Dynamics entries where mask is False should be zero."""
        offdiag_support = np.zeros((3, 3), dtype=bool)
        offdiag_support[1, 0] = True  # X→Y
        offdiag_support[2, 1] = True  # Y→Z

        spec = _make_3latent_spec(edge_support=offdiag_support)
        model = SSMModel(spec)

        rng = random.PRNGKey(42)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=get_laplace_backend(model, 6),
        )

        weight_sites = sorted(
            name for name in trace if name.startswith("vf_") and name.endswith("_weight")
        )
        assert weight_sites == ["vf_1_weight", "vf_2_weight"]

    @pytest.mark.cpu_expensive
    def test_no_mask_fully_free(self):
        """Default dynamics mask expands to a fully free dynamics structure."""
        spec = _make_3latent_spec()
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=get_laplace_backend(model, 6),
        )

        weight_sites = [
            name for name in trace if name.startswith("vf_") and name.endswith("_weight")
        ]
        assert len(weight_sites) == 6

    @pytest.mark.cpu_expensive
    def test_dynamics_support_single_latent(self):
        """Single latent: no off-diagonal, mask should be identity."""
        spec = block_ssm_spec(
            n_latent=1,
            n_manifest=1,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=1,
                decay_support=np.ones(1, dtype=bool),
                edge_support=np.zeros((1, 1), dtype=bool),
                coupling_template=jnp.zeros((1, 1)),
                intercept_support=np.zeros(1, dtype=bool),
                cint_template=jnp.zeros(1),
            ),
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((5, 1)),
            times=jnp.arange(5, dtype=jnp.float32),
            likelihood_backend=get_laplace_backend(model, 6),
        )

        assert not any(name.startswith("vf_") and name.endswith("_weight") for name in trace)


# ═══════════════════════════════════════════════════════════════════════
# Fix 2: Structured lambda
# ═══════════════════════════════════════════════════════════════════════


class TestLambdaMask:
    """Test that lambda_support constrains factor loadings."""

    @pytest.mark.cpu_expensive
    def test_lambda_template_plus_mask(self):
        """Template+mask mode: fixed reference + free additional loadings."""
        # X has 2 indicators (x1 ref, x2 free), Y has 1, Z has 1
        lambda_mat = jnp.zeros((4, 3))
        lambda_mat = lambda_mat.at[0, 0].set(1.0)  # x1→X (ref)
        lambda_mat = lambda_mat.at[2, 1].set(1.0)  # y1→Y (ref)
        lambda_mat = lambda_mat.at[3, 2].set(1.0)  # z1→Z (ref)

        lambda_support = np.zeros((4, 3), dtype=bool)
        lambda_support[1, 0] = True  # x2→X (free)

        spec = _make_3latent_spec(
            lambda_block=SparseMatrixBlockSpec(
                n_rows=4,
                n_cols=3,
                free_support=lambda_support,
                template=lambda_mat,
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            )
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=get_laplace_backend(model, 6),
        )

        # Only 1 free loading sampled
        assert trace["lambda_free"]["value"].shape == (1,)

        # Check the assembled lambda
        lam = trace["lambda"]["value"]
        assert float(lam[0, 0]) == 1.0  # Fixed reference
        assert float(lam[2, 1]) == 1.0  # Fixed reference
        assert float(lam[3, 2]) == 1.0  # Fixed reference
        assert float(lam[1, 0]) != 0.0  # Free loading was sampled

    @pytest.mark.cpu_expensive
    def test_lambda_no_mask_returns_fixed(self):
        """Array lambda_mat with default zero free-mask is returned as-is."""
        lambda_mat = jnp.eye(4, 3)
        spec = _make_3latent_spec(
            lambda_block=SparseMatrixBlockSpec(
                n_rows=4,
                n_cols=3,
                free_support=zero_loading_support(4, 3),
                template=lambda_mat,
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            )
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=get_laplace_backend(model, 6),
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

    @pytest.mark.cpu_expensive
    def test_per_element_prior_in_model(self):
        """Per-element dynamics priors are used in sampling."""
        offdiag_support = np.zeros((2, 2), dtype=bool)
        offdiag_support[1, 0] = True  # X→Y

        spec = block_ssm_spec(
            n_latent=2,
            n_manifest=2,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=2,
                decay_support=np.ones(2, dtype=bool),
                edge_support=offdiag_support,
                coupling_template=jnp.zeros((2, 2)),
                intercept_support=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
            latent_names=["X", "Y"],
            manifest_names=["x1", "y1"],
        )

        # Per-element prior: single off-diagonal has mu=2.0
        priors = {
            "vf_1_weight": distribution_from_params(
                PriorDistributionFamily.NORMAL,
                {"mu": 2.0, "sigma": 0.1},
            )
        }
        model = SSMModel(spec, priors)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((5, 2)),
            times=jnp.arange(5, dtype=jnp.float32),
            likelihood_backend=get_laplace_backend(model, 6),
        )

        # The off-diagonal value should be near 2.0 (tight prior)
        weight = float(trace["vf_1_weight"]["value"])
        assert abs(weight - 2.0) < 1.0, f"Expected ~2.0, got {weight}"


# ═══════════════════════════════════════════════════════════════════════
# Runtime structural-support construction
# ═══════════════════════════════════════════════════════════════════════


class TestRuntimeStructuralSupport:
    """Test that compilation constructs correct block support from CausalDesign."""

    def test_build_structural_support_from_plan(self):
        """Compilation constructs dynamics/lambda support from CausalDesign."""
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_plan,
        )

        structural_plan = _make_structural_plan()

        latent_names = ["X", "Y", "Z"]
        manifest_cols = ["x1", "x2", "y1", "z1"]

        (
            dynamics_support,
            _input_effect_support,
            lambda_mat,
            lambda_support,
            _cat,
            _edge_lag_days,
        ) = build_structural_support_from_plan(
            latent_names,
            manifest_cols,
            3,
            4,
            manifest_dists=[DistributionFamily.GAUSSIAN] * 4,
            structural_plan=structural_plan,
        )

        # Dynamics mask: baseline persistence diagonals + X→Y + Y→Z
        assert dynamics_support is not None
        assert dynamics_support[0, 0]  # X baseline persistence
        assert dynamics_support[1, 1]  # Y self
        assert dynamics_support[2, 2]  # Z self
        assert dynamics_support[1, 0]  # X→Y (effect=Y row, cause=X col)
        assert dynamics_support[2, 1]  # Y→Z (effect=Z row, cause=Y col)
        assert not dynamics_support[0, 1]  # No Y→X edge
        assert not dynamics_support[0, 2]  # No Z→X edge
        assert not dynamics_support[1, 2]  # No Z→Y edge
        assert not dynamics_support[2, 0]  # No X→Z edge

        # Lambda: x1 fixed ref for X, x2 free for X, y1 fixed ref for Y, z1 fixed ref for Z
        assert float(lambda_mat[0, 0]) == 1.0  # x1→X
        assert float(lambda_mat[2, 1]) == 1.0  # y1→Y
        assert float(lambda_mat[3, 2]) == 1.0  # z1→Z

        assert lambda_support is not None
        assert lambda_support[1, 0]  # x2→X is free
        assert not lambda_support[0, 0]  # x1→X is fixed
        assert not lambda_support[2, 1]  # y1→Y is fixed

    def test_no_causal_design_materializes_explicit_default_masks(self):
        """Without causal_design, structural defaults are still explicit."""
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_plan,
        )

        (
            dynamics_support,
            input_effect_support,
            _lambda_mat,
            lambda_support,
            _cat,
            _edge_lag_days,
        ) = build_structural_support_from_plan(
            None,
            ["x1"],
            1,
            1,
            manifest_dists=[DistributionFamily.GAUSSIAN],
            structural_plan=None,
        )
        np.testing.assert_array_equal(dynamics_support, np.array([[True]]))
        np.testing.assert_array_equal(input_effect_support, np.zeros((1, 0), dtype=bool))
        np.testing.assert_array_equal(lambda_support, np.array([[False]]))

    def test_known_input_edge_compiles_to_input_effect_support(self):
        """Known inputs are transition drivers, not latent dynamics columns."""
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_plan,
        )

        causal_design = {
            "latent": {
                "default_outcome": {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
                "constructs": [
                    {
                        "id": "construct:16176a18c25802dee8a1",
                        "name": "dose",
                        "description": "Medication dose",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                    },
                    {
                        "id": "construct:bbc87212909e45b9e6c3",
                        "name": "mood",
                        "description": "Mood state",
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
                        "how_to_measure": "record dose",
                        "measurement_dtype": "continuous",
                        "aggregation": "sum",
                    },
                    {
                        "id": "indicator:e05e217de7f4442abdc5",
                        "construct_id": "construct:bbc87212909e45b9e6c3",
                        "name": "mood_rating",
                        "construct_polarity": "positive",
                        "how_to_measure": "record mood",
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
                    "missing_policy": "zero",
                }
            ],
        }
        from nof1_causal_lab.models.structural import build_structural_plan

        structural_plan = build_structural_plan(CausalDesign.model_validate(causal_design))

        dynamics_support, input_effect_support, lambda_mat, lambda_support, _cat, edge_lag_days = (
            build_structural_support_from_plan(
                ["mood"],
                ["mood_rating"],
                1,
                1,
                manifest_dists=[DistributionFamily.GAUSSIAN],
                structural_plan=structural_plan,
            )
        )

        np.testing.assert_array_equal(dynamics_support, np.array([[True]]))
        np.testing.assert_array_equal(input_effect_support, np.array([[True]]))
        np.testing.assert_array_equal(lambda_mat, np.array([[1.0]]))
        np.testing.assert_array_equal(lambda_support, np.array([[False]]))
        assert edge_lag_days == {}

    @pytest.mark.parametrize(
        ("override", "error_pattern"),
        [
            pytest.param(
                {
                    "lambda_block": SparseMatrixBlockSpec(
                        n_rows=4,
                        n_cols=3,
                        free_support=cast("np.ndarray", None),
                        template=jnp.eye(4, 3),
                        free_site_name="lambda_free",
                        det_site_name="lambda",
                        support=SupportClass.REAL,
                        site_kind=SiteKind.LOADING,
                        assembly_group="lambda",
                        fixed_spec_field="lambda_mat",
                        priors_field="lambda_free",
                    )
                },
                "lambda_support must have shape",
                id="lambda_support_none",
            ),
            pytest.param(
                {
                    "lambda_block": SparseMatrixBlockSpec(
                        n_rows=4,
                        n_cols=3,
                        free_support=np.ones((4, 4), dtype=bool),
                        template=jnp.eye(4, 3),
                        free_site_name="lambda_free",
                        det_site_name="lambda",
                        support=SupportClass.REAL,
                        site_kind=SiteKind.LOADING,
                        assembly_group="lambda",
                        fixed_spec_field="lambda_mat",
                        priors_field="lambda_free",
                    )
                },
                r"lambda_support must have shape \(4, 3\)",
                id="lambda_support_wrong_shape",
            ),
            pytest.param(
                {"diffusion_dists": [DistributionFamily.GAUSSIAN] * 2},
                "diffusion_dists length must match n_latent",
                id="diffusion_dists_short",
            ),
            pytest.param(
                {"manifest_dists": [DistributionFamily.GAUSSIAN] * 3},
                "manifest_dists length must match n_manifest",
                id="manifest_dists_short",
            ),
            pytest.param(
                {
                    "manifest_dists": [DistributionFamily.GAUSSIAN] * 4,
                    "manifest_links": [LinkFunction.IDENTITY] * 3,
                },
                "manifest_links length must match n_manifest",
                id="manifest_links_short",
            ),
            pytest.param(
                {"manifest_level_counts": [0, 0, 0]},
                "manifest_level_counts length must match n_manifest",
                id="manifest_level_counts_short",
            ),
        ],
    )
    def test_ssm_spec_rejects_invalid_structural_metadata(
        self,
        override: dict[str, Any],
        error_pattern: str,
    ):
        """SSMSpec rejects invalid mask shapes, missing masks, and mismatched lengths."""
        base: dict[str, Any] = {
            "n_latent": 3,
            "n_manifest": 4,
            "dynamics_spec": dense_matrix_dynamics_spec(
                n_latent=3,
                decay_support=np.ones(3, dtype=bool),
                edge_support=np.ones((3, 3), dtype=bool),
                coupling_template=jnp.zeros((3, 3)),
                intercept_support=np.zeros(3, dtype=bool),
                cint_template=jnp.zeros(3),
            ),
            "lambda_block": SparseMatrixBlockSpec(
                n_rows=4,
                n_cols=3,
                free_support=zero_loading_support(4, 3),
                template=jnp.eye(4, 3),
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
            "latent_names": ["X", "Y", "Z"],
            "manifest_names": ["x1", "x2", "y1", "z1"],
        }
        with pytest.raises(ValueError, match=error_pattern):
            block_ssm_spec(**{**base, **override})

    def test_ssm_spec_rejects_string_lambda_mat(self):
        """Loading structure must be expressed as template + mask, not a string mode."""
        with pytest.raises(ValueError, match="lambda_mat must have shape"):
            block_ssm_spec(
                n_latent=2,
                n_manifest=2,
                dynamics_spec=dense_matrix_dynamics_spec(
                    n_latent=2,
                    decay_support=np.ones(2, dtype=bool),
                    edge_support=np.ones((2, 2), dtype=bool),
                    coupling_template=jnp.zeros((2, 2)),
                    intercept_support=np.zeros(2, dtype=bool),
                    cint_template=jnp.zeros(2),
                ),
                lambda_block=SparseMatrixBlockSpec(
                    n_rows=2,
                    n_cols=2,
                    free_support=zero_loading_support(2, 2),
                    template=cast("jax.Array", "free"),
                    free_site_name="lambda_free",
                    det_site_name="lambda",
                    support=SupportClass.REAL,
                    site_kind=SiteKind.LOADING,
                    assembly_group="lambda",
                    fixed_spec_field="lambda_mat",
                    priors_field="lambda_free",
                ),
            )

    def test_model_build_accepts_only_already_compiled_ssm_spec(self):
        """Runtime construction consumes an SSMSpec without structural authoring inputs."""
        from nof1_causal_lab.models.ssm.runtime import build_ssm_model

        X = pl.DataFrame(
            {
                "time": list(range(5)),
                "x1": [1.0] * 5,
                "x2": [2.0] * 5,
                "y1": [3.0] * 5,
                "z1": [4.0] * 5,
            }
        )

        model = build_ssm_model(X, ssm_spec=_make_3latent_spec())
        assert model.spec.n_latent == 3

    def test_model_build_has_no_autodetect_path(self):
        """Runtime construction requires an already compiled SSMSpec."""
        from nof1_causal_lab.models.ssm.runtime import build_ssm_model

        X = pl.DataFrame(
            {
                "time": list(range(5)),
                "x1": [1.0] * 5,
                "x2": [2.0] * 5,
                "y1": [3.0] * 5,
                "z1": [4.0] * 5,
            }
        )

        with pytest.raises(TypeError, match="ssm_spec"):
            cast("Any", build_ssm_model)(X)

    def test_translate_spec_compiles_static_baseline_factor_from_induced_dependency(self):
        """Initial-state confounders should compile to low-rank baseline factors."""
        from nof1_causal_lab.artifacts.statistical_model_spec import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
            StatisticalModelSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        causal_design = {
            "latent": {
                "default_outcome": {"kind": "construct", "id": "construct:cdc0b2958a9512b2abad"},
                "constructs": [
                    {
                        "id": "construct:766f6091724c163a3404",
                        "name": "u_shared",
                        "description": "Shared static confounder",
                        "role": "exogenous",
                        "temporal_status": "time_invariant",
                    },
                    {
                        "id": "construct:6b04dc42c531e7091eb8",
                        "name": "stress",
                        "description": "Stress",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                    },
                    {
                        "id": "construct:cdc0b2958a9512b2abad",
                        "name": "sleep",
                        "description": "Sleep",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                    },
                ],
                "edges": [
                    {
                        "cause_id": "construct:766f6091724c163a3404",
                        "effect_id": "construct:6b04dc42c531e7091eb8",
                        "id": "edge:0e72c3c33116c415bd39",
                        "description": "Shared baseline causes stress",
                    },
                    {
                        "cause_id": "construct:766f6091724c163a3404",
                        "effect_id": "construct:cdc0b2958a9512b2abad",
                        "id": "edge:a1528b0e1cd05dd511ef",
                        "description": "Shared baseline causes sleep",
                    },
                ],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "id": "indicator:3696aef3ff6f446744e5",
                        "construct_id": "construct:6b04dc42c531e7091eb8",
                        "name": "stress_score",
                        "construct_polarity": "positive",
                        "how_to_measure": "measure stress",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                    {
                        "id": "indicator:7f807162156d3eb1b611",
                        "construct_id": "construct:cdc0b2958a9512b2abad",
                        "name": "sleep_score",
                        "construct_polarity": "positive",
                        "how_to_measure": "measure sleep",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                    },
                ],
            },
        }
        from nof1_causal_lab.models.structural import build_structural_plan

        structural_plan = build_structural_plan(CausalDesign.model_validate(causal_design))
        statistical_model_spec = StatisticalModelSpec(
            mechanisms=TypeAdapter(list[DynamicsMechanism]).validate_python([]),
            likelihoods=[
                LikelihoodSpec(
                    indicator_id="indicator:3696aef3ff6f446744e5",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:7f807162156d3eb1b611",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
            ],
            parameters=[
                ParameterSpec(
                    id="parameter:edb7a87a6e0c5740c6d0e3361cf57ea2682f070b5c9e210e3e6941defacb0640",
                    owners=[ConstructRef(id="construct:766f6091724c163a3404")],
                    quantity=SiteKind.STATIC_STATE_SD,
                    name="tau_u_shared",
                    role=ParameterRole.STATIC_STATE_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="baseline confounder sd",
                ),
            ],
        )

        spec, _edge_lag_days = translate_spec(
            declare_test_dynamics(statistical_model_spec, structural_plan),
            structural_plan=structural_plan,
        )

        np.testing.assert_array_equal(spec.static_state_sd_block.free_support, np.array([True]))
        np.testing.assert_allclose(np.asarray(spec.static_state_sd_block.template), np.zeros(1))
        np.testing.assert_allclose(
            np.asarray(spec.static_factor_loadings),
            np.array([[1.0], [1.0]]),
        )
        assert spec.static_factor_names == ["tau_u_shared"]
        np.testing.assert_array_equal(
            spec.t0_chol_block.correlation_support,
            np.zeros((2, 2), dtype=bool),
        )
        np.testing.assert_array_equal(spec.t0_means_block.free_support, np.array([False, False]))
        np.testing.assert_array_equal(spec.t0_chol_block.diag_support, np.array([False, False]))

    def test_translate_spec_marks_standardizable_gaussian_mean_indicators(self):
        """Gaussian identity indicators with interval means should be auto-standardized."""
        from nof1_causal_lab.artifacts.statistical_model_spec import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            StatisticalModelSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        structural_plan = _make_structural_plan()
        statistical_model_spec = StatisticalModelSpec(
            mechanisms=TypeAdapter(list[DynamicsMechanism]).validate_python([]),
            likelihoods=[
                LikelihoodSpec(
                    indicator_id="indicator:0f93ce57e1f1d1c96f5c",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:27a6125b251378d8dd23",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:dec7b4916899d2109674",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:c26d752dbca8b5a287ac",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
            ],
            parameters=[],
        )

        spec, _edge_lag_days = translate_spec(
            declare_test_dynamics(statistical_model_spec, structural_plan),
            structural_plan=structural_plan,
        )

        assert spec.manifest_standardized == [True, True, True, True]

    def test_translate_spec_fixes_manifest_noise_for_single_indicator_constructs(self):
        """Single-indicator constructs get fixed zero manifest noise in the compiled spec."""
        from nof1_causal_lab.artifacts.statistical_model_spec import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
            StatisticalModelSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        structural_plan = _make_structural_plan()
        statistical_model_spec = StatisticalModelSpec(
            mechanisms=TypeAdapter(list[DynamicsMechanism]).validate_python(
                [
                    {
                        "kind": "node_potential",
                        "target_id": "construct:311c9047b5ede16a8f26",
                        "center": {"kind": "fixed", "value": 0},
                        "stiffness": {
                            "kind": "estimated",
                            "parameter_id": "parameter:d4f5f53b1e587883d9cd30cc9393726d9e06dc90b63b7cae47ce252f25f4b42a",
                        },
                        "quartic": {"kind": "fixed", "value": 0},
                    },
                    {
                        "kind": "node_potential",
                        "target_id": "construct:d90c52e59b79004188dc",
                        "center": {"kind": "fixed", "value": 0},
                        "stiffness": {
                            "kind": "estimated",
                            "parameter_id": "parameter:f6c297044713e7312c77a790e9a018f4881e7382620dec264eec8546bdd9c56a",
                        },
                        "quartic": {"kind": "fixed", "value": 0},
                    },
                    {
                        "kind": "node_potential",
                        "target_id": "construct:a6b7873d58dac1ff1a02",
                        "center": {"kind": "fixed", "value": 0},
                        "stiffness": {
                            "kind": "estimated",
                            "parameter_id": "parameter:d3fa7f07ce4589268f30bf007d008002a5dc55eb1c0c6b7f2c0bf4abb2f2b5c7",
                        },
                        "quartic": {"kind": "fixed", "value": 0},
                    },
                    {
                        "kind": "linear",
                        "edge_id": "edge:39ba80b774e02c409662",
                        "weight": {
                            "kind": "estimated",
                            "parameter_id": "parameter:11144ed541c47f7f351088902879f26bdb7eac1b92429dbb44d9bb4c95ba2dc1",
                        },
                    },
                    {
                        "kind": "linear",
                        "edge_id": "edge:57072ee1d1b7b3e7c0de",
                        "weight": {
                            "kind": "estimated",
                            "parameter_id": "parameter:300db78bd0bf69e3612bb8a4807b68188da5584b4f9f0ecedda7d3a5cff2b71f",
                        },
                    },
                ]
            ),
            likelihoods=[
                LikelihoodSpec(
                    indicator_id="indicator:0f93ce57e1f1d1c96f5c",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:27a6125b251378d8dd23",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:dec7b4916899d2109674",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:c26d752dbca8b5a287ac",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
            ],
            parameters=[
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("dt_persistence_to_ct_decay"),
                    id="parameter:d4f5f53b1e587883d9cd30cc9393726d9e06dc90b63b7cae47ce252f25f4b42a",
                    owners=[ConstructRef(id="construct:311c9047b5ede16a8f26")],
                    quantity=SiteKind.DYNAMICS_DECAY,
                    name="rho_X",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for X",
                ),
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("dt_persistence_to_ct_decay"),
                    id="parameter:f6c297044713e7312c77a790e9a018f4881e7382620dec264eec8546bdd9c56a",
                    owners=[ConstructRef(id="construct:d90c52e59b79004188dc")],
                    quantity=SiteKind.DYNAMICS_DECAY,
                    name="rho_Y",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for Y",
                ),
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("dt_persistence_to_ct_decay"),
                    id="parameter:d3fa7f07ce4589268f30bf007d008002a5dc55eb1c0c6b7f2c0bf4abb2f2b5c7",
                    owners=[ConstructRef(id="construct:a6b7873d58dac1ff1a02")],
                    quantity=SiteKind.DYNAMICS_DECAY,
                    name="rho_Z",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for Z",
                ),
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("dt_effect_to_ct_rate"),
                    id="parameter:11144ed541c47f7f351088902879f26bdb7eac1b92429dbb44d9bb4c95ba2dc1",
                    owners=[
                        ConstructRef(id="construct:311c9047b5ede16a8f26"),
                        ConstructRef(id="construct:d90c52e59b79004188dc"),
                        EdgeRef(id="edge:39ba80b774e02c409662"),
                    ],
                    quantity=SiteKind.DYNAMICS_WEIGHT,
                    name="beta_X_Y",
                    role=ParameterRole.FIXED_EFFECT,
                    constraint=ParameterConstraint.NONE,
                    description="X causes Y",
                ),
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("dt_effect_to_ct_rate"),
                    id="parameter:300db78bd0bf69e3612bb8a4807b68188da5584b4f9f0ecedda7d3a5cff2b71f",
                    owners=[
                        ConstructRef(id="construct:d90c52e59b79004188dc"),
                        ConstructRef(id="construct:a6b7873d58dac1ff1a02"),
                        EdgeRef(id="edge:57072ee1d1b7b3e7c0de"),
                    ],
                    quantity=SiteKind.DYNAMICS_WEIGHT,
                    name="beta_Y_Z",
                    role=ParameterRole.FIXED_EFFECT,
                    constraint=ParameterConstraint.NONE,
                    description="Y causes Z",
                ),
                ParameterSpec(
                    id="parameter:4d1d6a7ea877cc9346711b960e6d5c34ee74ba7da330fed57460d8510bb8a965",
                    owners=[ConstructRef(id="construct:311c9047b5ede16a8f26")],
                    quantity=SiteKind.DIFFUSION_DIAG,
                    name="sigma_X",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="residual sd X",
                ),
                ParameterSpec(
                    id="parameter:0211fbba3c9cacffff315088926ede62eb852aaa2dfa9f46cd6bc565c03eb54e",
                    owners=[ConstructRef(id="construct:d90c52e59b79004188dc")],
                    quantity=SiteKind.DIFFUSION_DIAG,
                    name="sigma_Y",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="residual sd Y",
                ),
                ParameterSpec(
                    id="parameter:16ad5b8c5e6063e35ab1a10813a5ec3475ac8c0f03c152ae40f643eaf2c60722",
                    owners=[ConstructRef(id="construct:a6b7873d58dac1ff1a02")],
                    quantity=SiteKind.DIFFUSION_DIAG,
                    name="sigma_Z",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="residual sd Z",
                ),
                ParameterSpec(
                    id="parameter:4a482718870c06d541bb044c41ca4d9f372a25fedd43478cc495204883bb6ae5",
                    owners=[
                        IndicatorRef(id="indicator:27a6125b251378d8dd23"),
                        ConstructRef(id="construct:311c9047b5ede16a8f26"),
                    ],
                    quantity=SiteKind.LOADING,
                    name="lambda_x2_X",
                    role=ParameterRole.LOADING,
                    constraint=ParameterConstraint.POSITIVE,
                    description="loading",
                ),
            ],
        )

        spec, _edge_lag_days = translate_spec(
            statistical_model_spec, structural_plan=structural_plan
        )

        assert isinstance(spec.manifest_chol_block.template, jnp.ndarray)
        np.testing.assert_array_equal(
            spec.manifest_chol_block.diag_support,
            np.array([True, True, False, False]),
        )
        np.testing.assert_allclose(np.asarray(spec.manifest_chol_block.template), np.zeros((4, 4)))

    def test_translate_spec_rejects_initial_state_correlation_parameters_with_causal_design(self):
        """Causal-spec compilation no longer accepts pairwise cor0 parameters."""
        from nof1_causal_lab.artifacts.statistical_model_spec import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
            StatisticalModelSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        structural_plan = _make_structural_plan()
        statistical_model_spec = StatisticalModelSpec(
            mechanisms=TypeAdapter(list[DynamicsMechanism]).validate_python([]),
            likelihoods=[
                LikelihoodSpec(
                    indicator_id="indicator:0f93ce57e1f1d1c96f5c",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:27a6125b251378d8dd23",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:dec7b4916899d2109674",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:c26d752dbca8b5a287ac",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
            ],
            parameters=[
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("initial_state_correlation"),
                    id="parameter:418d0717cfc9857102f86ab03eaa94a90f0757b6ac7826fc239d3167f7adaf50",
                    owners=[
                        ConstructRef(id="construct:311c9047b5ede16a8f26"),
                        ConstructRef(id="construct:a6b7873d58dac1ff1a02"),
                    ],
                    quantity=SiteKind.T0_VAR_LOWER,
                    name="cor0_X_Z",
                    role=ParameterRole.INITIAL_STATE_CORRELATION,
                    constraint=ParameterConstraint.CORRELATION,
                    description="initial correlation",
                ),
            ],
        )

        with pytest.raises(
            ValueError,
            match="no longer accepts INITIAL_STATE_CORRELATION parameters",
        ):
            translate_spec(statistical_model_spec, structural_plan=structural_plan)

    def test_translate_spec_rejects_self_initial_state_correlation_with_causal_design(self):
        """Even self-pairs are rejected once causal-design compilation is active."""
        from nof1_causal_lab.artifacts.statistical_model_spec import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
            StatisticalModelSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        structural_plan = _make_structural_plan()
        statistical_model_spec = StatisticalModelSpec(
            mechanisms=TypeAdapter(list[DynamicsMechanism]).validate_python([]),
            likelihoods=[
                LikelihoodSpec(
                    indicator_id="indicator:0f93ce57e1f1d1c96f5c",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:27a6125b251378d8dd23",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:dec7b4916899d2109674",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
                LikelihoodSpec(
                    indicator_id="indicator:c26d752dbca8b5a287ac",
                    distribution=DistributionFamily.GAUSSIAN,
                    link=LinkFunction.IDENTITY,
                    reasoning="test",
                ),
            ],
            parameters=[
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("initial_state_correlation"),
                    id="parameter:6efa548cc7ef957857274e64c9693e3bb393bed5fa3c20a78b572ec53aa65963",
                    owners=[
                        ConstructRef(id="construct:311c9047b5ede16a8f26"),
                        ConstructRef(id="construct:311c9047b5ede16a8f26"),
                    ],
                    quantity=SiteKind.T0_VAR_LOWER,
                    name="cor0_X_X",
                    role=ParameterRole.INITIAL_STATE_CORRELATION,
                    constraint=ParameterConstraint.CORRELATION,
                    description="invalid self correlation",
                ),
            ],
        )

        with pytest.raises(
            ValueError,
            match="no longer accepts INITIAL_STATE_CORRELATION parameters",
        ):
            translate_spec(statistical_model_spec, structural_plan=structural_plan)

    def test_model_build_end_to_end(self):
        """Model construction with causal_design produces masked spec."""

        from nof1_causal_lab.artifacts.statistical_model_spec import (
            DistributionFamily,
            LikelihoodSpec,
            LinkFunction,
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
            StatisticalModelSpec,
        )

        def _lik(var: str) -> LikelihoodSpec:
            return LikelihoodSpec(
                indicator_id=fixture_entity_id("indicator", var),
                distribution=DistributionFamily.GAUSSIAN,
                link=LinkFunction.IDENTITY,
                reasoning="test",
            )

        statistical_model_spec = StatisticalModelSpec(
            mechanisms=TypeAdapter(list[DynamicsMechanism]).validate_python(
                [
                    {
                        "kind": "node_potential",
                        "target_id": "construct:311c9047b5ede16a8f26",
                        "center": {"kind": "fixed", "value": 0},
                        "stiffness": {
                            "kind": "estimated",
                            "parameter_id": "parameter:d4f5f53b1e587883d9cd30cc9393726d9e06dc90b63b7cae47ce252f25f4b42a",
                        },
                        "quartic": {"kind": "fixed", "value": 0},
                    },
                    {
                        "kind": "node_potential",
                        "target_id": "construct:d90c52e59b79004188dc",
                        "center": {"kind": "fixed", "value": 0},
                        "stiffness": {
                            "kind": "estimated",
                            "parameter_id": "parameter:f6c297044713e7312c77a790e9a018f4881e7382620dec264eec8546bdd9c56a",
                        },
                        "quartic": {"kind": "fixed", "value": 0},
                    },
                    {
                        "kind": "node_potential",
                        "target_id": "construct:a6b7873d58dac1ff1a02",
                        "center": {"kind": "fixed", "value": 0},
                        "stiffness": {
                            "kind": "estimated",
                            "parameter_id": "parameter:d3fa7f07ce4589268f30bf007d008002a5dc55eb1c0c6b7f2c0bf4abb2f2b5c7",
                        },
                        "quartic": {"kind": "fixed", "value": 0},
                    },
                    {
                        "kind": "linear",
                        "edge_id": "edge:39ba80b774e02c409662",
                        "weight": {
                            "kind": "estimated",
                            "parameter_id": "parameter:11144ed541c47f7f351088902879f26bdb7eac1b92429dbb44d9bb4c95ba2dc1",
                        },
                    },
                    {
                        "kind": "linear",
                        "edge_id": "edge:57072ee1d1b7b3e7c0de",
                        "weight": {
                            "kind": "estimated",
                            "parameter_id": "parameter:300db78bd0bf69e3612bb8a4807b68188da5584b4f9f0ecedda7d3a5cff2b71f",
                        },
                    },
                ]
            ),
            likelihoods=[_lik("x1"), _lik("x2"), _lik("y1"), _lik("z1")],
            parameters=[
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("dt_persistence_to_ct_decay"),
                    id="parameter:d4f5f53b1e587883d9cd30cc9393726d9e06dc90b63b7cae47ce252f25f4b42a",
                    owners=[ConstructRef(id="construct:311c9047b5ede16a8f26")],
                    quantity=SiteKind.DYNAMICS_DECAY,
                    name="rho_X",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for X",
                ),
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("dt_persistence_to_ct_decay"),
                    id="parameter:f6c297044713e7312c77a790e9a018f4881e7382620dec264eec8546bdd9c56a",
                    owners=[ConstructRef(id="construct:d90c52e59b79004188dc")],
                    quantity=SiteKind.DYNAMICS_DECAY,
                    name="rho_Y",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for Y",
                ),
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("dt_persistence_to_ct_decay"),
                    id="parameter:d3fa7f07ce4589268f30bf007d008002a5dc55eb1c0c6b7f2c0bf4abb2f2b5c7",
                    owners=[ConstructRef(id="construct:a6b7873d58dac1ff1a02")],
                    quantity=SiteKind.DYNAMICS_DECAY,
                    name="rho_Z",
                    role=ParameterRole.AR_COEFFICIENT,
                    constraint=ParameterConstraint.UNIT_INTERVAL,
                    description="AR for Z",
                ),
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("dt_effect_to_ct_rate"),
                    id="parameter:11144ed541c47f7f351088902879f26bdb7eac1b92429dbb44d9bb4c95ba2dc1",
                    owners=[
                        ConstructRef(id="construct:311c9047b5ede16a8f26"),
                        ConstructRef(id="construct:d90c52e59b79004188dc"),
                        EdgeRef(id="edge:39ba80b774e02c409662"),
                    ],
                    quantity=SiteKind.DYNAMICS_WEIGHT,
                    name="beta_X_Y",
                    role=ParameterRole.FIXED_EFFECT,
                    constraint=ParameterConstraint.NONE,
                    description="X→Y effect",
                ),
                ParameterSpec(
                    prior_transform=PriorAuthoringTransform("dt_effect_to_ct_rate"),
                    id="parameter:300db78bd0bf69e3612bb8a4807b68188da5584b4f9f0ecedda7d3a5cff2b71f",
                    owners=[
                        ConstructRef(id="construct:d90c52e59b79004188dc"),
                        ConstructRef(id="construct:a6b7873d58dac1ff1a02"),
                        EdgeRef(id="edge:57072ee1d1b7b3e7c0de"),
                    ],
                    quantity=SiteKind.DYNAMICS_WEIGHT,
                    name="beta_Y_Z",
                    role=ParameterRole.FIXED_EFFECT,
                    constraint=ParameterConstraint.NONE,
                    description="Y→Z effect",
                ),
                ParameterSpec(
                    id="parameter:4d1d6a7ea877cc9346711b960e6d5c34ee74ba7da330fed57460d8510bb8a965",
                    owners=[ConstructRef(id="construct:311c9047b5ede16a8f26")],
                    quantity=SiteKind.DIFFUSION_DIAG,
                    name="sigma_X",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Residual SD for X",
                ),
                ParameterSpec(
                    id="parameter:0211fbba3c9cacffff315088926ede62eb852aaa2dfa9f46cd6bc565c03eb54e",
                    owners=[ConstructRef(id="construct:d90c52e59b79004188dc")],
                    quantity=SiteKind.DIFFUSION_DIAG,
                    name="sigma_Y",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Residual SD for Y",
                ),
                ParameterSpec(
                    id="parameter:16ad5b8c5e6063e35ab1a10813a5ec3475ac8c0f03c152ae40f643eaf2c60722",
                    owners=[ConstructRef(id="construct:a6b7873d58dac1ff1a02")],
                    quantity=SiteKind.DIFFUSION_DIAG,
                    name="sigma_Z",
                    role=ParameterRole.RESIDUAL_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Residual SD for Z",
                ),
                ParameterSpec(
                    id="parameter:4a482718870c06d541bb044c41ca4d9f372a25fedd43478cc495204883bb6ae5",
                    owners=[
                        IndicatorRef(id="indicator:27a6125b251378d8dd23"),
                        ConstructRef(id="construct:311c9047b5ede16a8f26"),
                    ],
                    quantity=SiteKind.LOADING,
                    name="lambda_x2_X",
                    role=ParameterRole.LOADING,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Loading for x2 on X",
                ),
                ParameterSpec(
                    id="parameter:0e147e1b1e51c7070d8c565c9d545e8ddff815cc8878fed4ff60c7744ee0387a",
                    owners=[IndicatorRef(id="indicator:0f93ce57e1f1d1c96f5c")],
                    quantity=SiteKind.MANIFEST_VAR_DIAG,
                    name="obs_sd_x1",
                    role=ParameterRole.MEASUREMENT_ERROR_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Measurement-error SD for x1",
                ),
                ParameterSpec(
                    id="parameter:aedb301479cf2b81aabe25bf88afff064b6c90bd92c6c0c80d0b5acd02a121af",
                    owners=[IndicatorRef(id="indicator:27a6125b251378d8dd23")],
                    quantity=SiteKind.MANIFEST_VAR_DIAG,
                    name="obs_sd_x2",
                    role=ParameterRole.MEASUREMENT_ERROR_SD,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Measurement-error SD for x2",
                ),
            ],
        )

        structural_plan = _make_structural_plan()

        # Minimal wide data
        X = pl.DataFrame(
            {
                "time": list(range(10)),
                "x1": [1.0] * 10,
                "x2": [2.0] * 10,
                "y1": [3.0] * 10,
                "z1": [4.0] * 10,
            }
        )

        from nof1_causal_lab.models.prior_planning import complete_parameter_priors
        from nof1_causal_lab.models.ssm.compile.artifact import compile_ssm_artifact
        from nof1_causal_lab.models.ssm.runtime import hydrate_compiled_model

        compiled = compile_ssm_artifact(
            complete_parameter_priors(statistical_model_spec),
            structural_plan,
        )
        model = hydrate_compiled_model(compiled, X)
        spec = model.spec

        dynamics_sites = [
            site for site in spec.iter_sample_sites() if site.assembly_group == "dynamics"
        ]
        assert sum(site.site_kind == SiteKind.DYNAMICS_DECAY for site in dynamics_sites) == 3
        assert spec.lambda_block.free_support is not None
        assert spec.n_latent == 3
        assert spec.n_manifest == 4


# ═══════════════════════════════════════════════════════════════════════
# Site-registry mask awareness
# ═══════════════════════════════════════════════════════════════════════


class TestSiteRegistryMasks:
    """Test that the canonical site registry respects SSM masks."""

    def test_site_registry_with_dynamics_support(self):
        """Site registry should size masked dynamics entries correctly."""
        from nof1_causal_lab.models.ssm.parameterization import build_site_registry

        # 3 latent, X→Y and Y→Z = 2 off-diagonal entries
        offdiag_support = np.zeros((3, 3), dtype=bool)
        offdiag_support[1, 0] = True
        offdiag_support[2, 1] = True

        spec = block_ssm_spec(
            n_latent=3,
            n_manifest=3,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=3,
                decay_support=np.ones(3, dtype=bool),
                edge_support=offdiag_support,
                coupling_template=jnp.zeros((3, 3)),
                intercept_support=full_vector_support(3),
                cint_template=jnp.zeros(3),
            ),
        )

        registry = {site.name: site for site in build_site_registry(spec)}

        weight_sites = sorted(name for name in registry if name.endswith("_weight"))
        assert weight_sites == ["vf_1_weight", "vf_2_weight"]
        assert registry["vf_0_decay"].shape == (3,)

    def test_site_registry_with_lambda_support(self):
        """Site registry should size masked loading entries correctly."""
        from nof1_causal_lab.models.ssm.parameterization import build_site_registry

        lambda_mat = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        lambda_support = np.array([[False, False], [False, False], [True, False]])

        spec = block_ssm_spec(
            n_latent=2,
            n_manifest=3,
            dynamics_spec=dense_matrix_dynamics_spec(
                n_latent=2,
                decay_support=np.ones(2, dtype=bool),
                edge_support=np.ones((2, 2), dtype=bool),
                coupling_template=jnp.zeros((2, 2)),
                intercept_support=np.zeros(2, dtype=bool),
                cint_template=jnp.zeros(2),
            ),
            lambda_block=SparseMatrixBlockSpec(
                n_rows=3,
                n_cols=2,
                free_support=lambda_support,
                template=lambda_mat,
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
        )

        registry = {site.name: site for site in build_site_registry(spec)}
        assert registry["lambda_free"].shape == (1,)


# ═══════════════════════════════════════════════════════════════════════
# Integration: trace verification
# ═══════════════════════════════════════════════════════════════════════


class TestTraceVerification:
    """Verify parameter shapes via numpyro.handlers.trace."""

    @pytest.mark.cpu_expensive
    def test_masked_model_trace(self):
        """Full model trace with component edge sites."""
        offdiag_support = np.zeros((3, 3), dtype=bool)
        offdiag_support[1, 0] = True  # X→Y
        offdiag_support[2, 1] = True  # Y→Z

        lambda_mat = jnp.zeros((4, 3))
        lambda_mat = lambda_mat.at[0, 0].set(1.0)
        lambda_mat = lambda_mat.at[2, 1].set(1.0)
        lambda_mat = lambda_mat.at[3, 2].set(1.0)

        lambda_support = np.zeros((4, 3), dtype=bool)
        lambda_support[1, 0] = True

        spec = _make_3latent_spec(
            edge_support=offdiag_support,
            lambda_block=SparseMatrixBlockSpec(
                n_rows=4,
                n_cols=3,
                free_support=lambda_support,
                template=lambda_mat,
                free_site_name="lambda_free",
                det_site_name="lambda",
                support=SupportClass.REAL,
                site_kind=SiteKind.LOADING,
                assembly_group="lambda",
                fixed_spec_field="lambda_mat",
                priors_field="lambda_free",
            ),
        )
        model = SSMModel(spec)

        rng = random.PRNGKey(123)
        trace = handlers.trace(handlers.seed(model.model, rng)).get_trace(
            observations=jnp.zeros((10, 4)),
            times=jnp.arange(10, dtype=jnp.float32),
            likelihood_backend=get_laplace_backend(model, 6),
        )

        assert trace["vf_0_decay"]["value"].shape == (3,)
        weight_sites = [
            name for name in trace if name.startswith("vf_") and name.endswith("_weight")
        ]
        assert len(weight_sites) == 2

        # Lambda: 1 free loading
        assert trace["lambda_free"]["value"].shape == (1,)

        # Deterministic lambda should be 4x3
        assert trace["lambda"]["value"].shape == (4, 3)


# ═══════════════════════════════════════════════════════════════════════
# Gradual-build surface: self-limiting quartic + Hill (saturating) edges
# ═══════════════════════════════════════════════════════════════════════


def _lik_gaussian(var: str):
    from nof1_causal_lab.artifacts.statistical_model_spec import (
        DistributionFamily,
        LikelihoodSpec,
        LinkFunction,
    )

    return LikelihoodSpec(
        indicator_id=fixture_entity_id("indicator", var),
        distribution=DistributionFamily.GAUSSIAN,
        link=LinkFunction.IDENTITY,
        reasoning="test",
    )


class TestGradualBuildComponents:
    """Quartic self-limitation and Hill edges materialize from the StatisticalModelSpec."""

    def test_quartic_freed_only_for_self_limiting_construct(self):
        from nof1_causal_lab.artifacts.statistical_model_spec import (
            ParameterConstraint,
            ParameterRole,
            ParameterSpec,
            StatisticalModelSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec
        from nof1_causal_lab.models.ssm.dynamics.spec import NodePotentialSpec

        statistical_model_spec = StatisticalModelSpec(
            mechanisms=TypeAdapter(list[DynamicsMechanism]).validate_python([]),
            likelihoods=[_lik_gaussian(v) for v in ("x1", "x2", "y1", "z1")],
            parameters=[
                ParameterSpec(
                    id="parameter:87b312ac99a5a40cf7f04d8f013226bb26398e66a5761e9a5c7296360850e80d",
                    owners=[ConstructRef(id="construct:d90c52e59b79004188dc")],
                    quantity=SiteKind.DYNAMICS_POTENTIAL_QUARTIC,
                    name="self_limit_Y",
                    role=ParameterRole.DYNAMICS_PARAMETER_POSITIVE,
                    constraint=ParameterConstraint.POSITIVE,
                    description="Y self-limits (crowding)",
                ),
            ],
        )
        spec, _ = translate_spec(
            declare_test_dynamics(
                statistical_model_spec,
                _make_structural_plan(),
                quartic_states=("construct:d90c52e59b79004188dc",),
            ),
            structural_plan=_make_structural_plan(),
        )

        wells = {
            c.target: c for c in spec.dynamics_spec.components if isinstance(c, NodePotentialSpec)
        }
        # Y (index 1) is self-limiting → quartic freed; X, Z stay pinned at 0.
        assert isinstance(wells[1].quartic, Free)
        assert wells[0].quartic == Fixed(0.0)
        assert wells[2].quartic == Fixed(0.0)

    def test_hill_edge_emitted_for_saturating_edge(self):
        from nof1_causal_lab.artifacts.statistical_model_spec import (
            ParameterSpec,
            StatisticalModelSpec,
        )
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec
        from nof1_causal_lab.models.ssm.dynamics.spec import HillEdgeSpec, LinearEdgeSpec

        def _hill_param(name: str) -> ParameterSpec:
            from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import (
                ParamCatalog,
            )

            return ParameterSpec.model_validate(
                ParamCatalog.from_structural_plan(_make_structural_plan()).metadata_for(name)
            )

        statistical_model_spec = StatisticalModelSpec(
            mechanisms=TypeAdapter(list[DynamicsMechanism]).validate_python([]),
            likelihoods=[_lik_gaussian(v) for v in ("x1", "x2", "y1", "z1")],
            parameters=[
                _hill_param("hill_emax_X_Y"),
                _hill_param("hill_ec50_X_Y"),
                _hill_param("hill_n_X_Y"),
            ],
        )
        spec, _ = translate_spec(
            declare_test_dynamics(
                statistical_model_spec,
                _make_structural_plan(),
                hill_edges=("edge:39ba80b774e02c409662",),
            ),
            structural_plan=_make_structural_plan(),
        )

        edges = [
            c
            for c in spec.dynamics_spec.components
            if isinstance(c, (HillEdgeSpec, LinearEdgeSpec))
        ]
        # X→Y (0→1) is saturating → Hill; Y→Z (1→2) stays linear.
        hill = [e for e in edges if isinstance(e, HillEdgeSpec)]
        linear = [e for e in edges if isinstance(e, LinearEdgeSpec)]
        assert [(e.source, e.target) for e in hill] == [(0, 1)]
        assert (1, 2) in [(e.source, e.target) for e in linear]
        assert (0, 1) not in [(e.source, e.target) for e in linear]

    @pytest.mark.cpu_expensive
    def test_freed_quartic_and_hill_sites_sample_finite(self):
        """The freed quartic + Hill edge sample through the real model with defaults."""
        import jax.numpy as jnp
        import jax.random as random
        import numpyro.handlers as handlers

        from nof1_causal_lab.artifacts.statistical_model_spec import (
            ParameterSpec,
            StatisticalModelSpec,
        )
        from nof1_causal_lab.flows.transitions.model_spec.agentic.construct_flow import ParamCatalog
        from nof1_causal_lab.models.ssm.compile.inputs import translate_spec

        plan = _make_structural_plan()
        catalog = ParamCatalog.from_structural_plan(plan)
        statistical_model_spec = StatisticalModelSpec(
            mechanisms=[],
            likelihoods=[_lik_gaussian(v) for v in ("x1", "x2", "y1", "z1")],
            parameters=[
                ParameterSpec.model_validate(catalog.metadata_for(name))
                for name in ("self_limit_Y", "hill_emax_X_Y", "hill_ec50_X_Y", "hill_n_X_Y")
            ],
        )
        statistical_model_spec = declare_test_dynamics(
            statistical_model_spec,
            plan,
            quartic_states=("construct:d90c52e59b79004188dc",),
            hill_edges=("edge:39ba80b774e02c409662",),
        )
        spec, _ = translate_spec(statistical_model_spec, structural_plan=plan)
        model = SSMModel(spec)

        trace = handlers.trace(handlers.seed(model.model, random.PRNGKey(0))).get_trace(
            observations=jnp.zeros((8, 4)),
            times=jnp.arange(8, dtype=jnp.float32),
            likelihood_backend=get_laplace_backend(model, 6),
        )
        quartic_sites = [n for n in trace if n.startswith("vf_") and n.endswith("_quartic")]
        emax_sites = [n for n in trace if n.startswith("vf_") and n.endswith("_Emax")]
        assert len(quartic_sites) == 1  # only Y
        assert len(emax_sites) == 1  # only X→Y
        for name in quartic_sites + emax_sites:
            assert bool(jnp.all(jnp.isfinite(trace[name]["value"])))
