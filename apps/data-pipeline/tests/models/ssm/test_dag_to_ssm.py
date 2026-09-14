"""Tests for DAG-to-SSM constraint propagation (Fixes 1-3).

Tests that:
1. dynamics_support constrains off-diagonal sampling to causal edges only
2. lambda_support + template constrains factor loadings to measurement structure
3. Per-element priors align with mask positions
4. Builder constructs structural support from ModelSpec
5. Pipeline threading passes scientific_model through
"""

from typing import Any, cast

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.handlers as handlers
import polars as pl
import pytest

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.identity import ConstructRef
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform, SiteKind, SupportClass
from nof1_causal_lab.distributions import DistributionFamily, PriorDistributionFamily
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.inference.backend_factory import get_laplace_backend
from nof1_causal_lab.models.ssm.model import SSMModel
from nof1_causal_lab.models.ssm.parameterization import (
    SiteDescriptor,
    build_site_registry,
)
from nof1_causal_lab.models.ssm.priors import resolve_site_priors
from nof1_causal_lab.models.ssm.structure import SparseMatrixBlockSpec
from nof1_causal_lab.prior_distributions import distribution_from_params
from tests.helpers import complete_test_model
from tests.model_fixtures import (
    dense_matrix_dynamics_spec,
    full_vector_support,
    model_fixture,
    zero_loading_support,
)

# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


def _make_3latent_spec(
    edge_support: np.ndarray | None = None,
    lambda_block: SparseMatrixBlockSpec | None = None,
) -> ModelSpec:
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
    return model_fixture(
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


def _model_payload() -> dict[str, Any]:
    """Minimal ModelSpec dict: X→Y, Y→Z, 4 indicators."""
    return {
        "default_outcome": {"kind": "construct", "id": "construct:d90c52e59b79004188dc"},
        "edges": [
            {
                "cause": {
                    "id": "construct:311c9047b5ede16a8f26",
                    "name": "X",
                    "description": "Cause",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "indicators": [
                        {
                            "id": "indicator:0f93ce57e1f1d1c96f5c",
                            "name": "x1",
                            "construct_polarity": "positive",
                            "how_to_measure": "measure x",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                        {
                            "id": "indicator:27a6125b251378d8dd23",
                            "name": "x2",
                            "construct_polarity": "positive",
                            "how_to_measure": "measure x alt",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        },
                    ],
                },
                "effect": {
                    "id": "construct:d90c52e59b79004188dc",
                    "name": "Y",
                    "description": "Mediator",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "indicators": [
                        {
                            "id": "indicator:dec7b4916899d2109674",
                            "name": "y1",
                            "construct_polarity": "positive",
                            "how_to_measure": "measure y",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        }
                    ],
                },
                "id": "edge:39ba80b774e02c409662",
                "description": "X causes Y",
                "lagged": True,
            },
            {
                "cause": {"kind": "construct", "id": "construct:d90c52e59b79004188dc"},
                "effect": {
                    "id": "construct:a6b7873d58dac1ff1a02",
                    "name": "Z",
                    "description": "Downstream",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "indicators": [
                        {
                            "id": "indicator:c26d752dbca8b5a287ac",
                            "name": "z1",
                            "construct_polarity": "positive",
                            "how_to_measure": "measure z",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                        }
                    ],
                },
                "id": "edge:57072ee1d1b7b3e7c0de",
                "description": "Y causes Z",
                "lagged": True,
            },
        ],
        "measurement_clock": "1d",
    }


def _make_model() -> ModelSpec:
    """Compile the shared causal fixture to the executable structural artifact."""

    return ModelSpec.model_validate(_model_payload())


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
        assert len(weight_sites) == 2
        assert {
            site.positions[0] for site in build_site_registry(spec) if site.name in weight_sites
        } == {
            (1, 0),
            (2, 1),
        }

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
        spec = model_fixture(
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

        spec = model_fixture(
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
    """Test that compilation constructs correct block support from ModelSpec."""

    def test_build_structural_support_from_model(self):
        """Compilation constructs dynamics/lambda support from ModelSpec."""
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_model,
        )

        model = _make_model()

        latent_names = ["X", "Y", "Z"]
        manifest_cols = ["x1", "x2", "y1", "z1"]

        (
            dynamics_support,
            _input_effect_support,
            lambda_mat,
            lambda_support,
            _cat,
            _edge_lag_days,
        ) = build_structural_support_from_model(
            latent_names,
            manifest_cols,
            3,
            4,
            manifest_dists=[DistributionFamily.GAUSSIAN] * 4,
            model=model,
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

    def test_known_input_edge_compiles_to_input_effect_support(self):
        """Known inputs are transition drivers, not latent dynamics columns."""
        from nof1_causal_lab.models.ssm.compile.inputs import (
            build_structural_support_from_model,
        )

        scientific_model = {
            "default_outcome": {"kind": "construct", "id": "construct:bbc87212909e45b9e6c3"},
            "edges": [
                {
                    "cause": {
                        "id": "construct:16176a18c25802dee8a1",
                        "name": "dose",
                        "description": "Medication dose",
                        "role": "exogenous",
                        "temporal_status": "time_varying",
                        "indicators": [
                            {
                                "id": "indicator:5806a6a8417abd85897f",
                                "name": "dose_mg",
                                "construct_polarity": "positive",
                                "how_to_measure": "record dose",
                                "measurement_dtype": "continuous",
                                "aggregation": "sum",
                            }
                        ],
                        "usage": {
                            "kind": "known_input",
                            "source_indicator_id": "indicator:5806a6a8417abd85897f",
                            "scale": 10.0,
                            "missing_policy": "zero",
                        },
                    },
                    "effect": {
                        "id": "construct:bbc87212909e45b9e6c3",
                        "name": "mood",
                        "description": "Mood state",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                        "indicators": [
                            {
                                "id": "indicator:e05e217de7f4442abdc5",
                                "name": "mood_rating",
                                "construct_polarity": "positive",
                                "how_to_measure": "record mood",
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

        model = ModelSpec.model_validate(scientific_model)

        dynamics_support, input_effect_support, lambda_mat, lambda_support, _cat, edge_lag_days = (
            build_structural_support_from_model(
                ["mood"],
                ["mood_rating"],
                1,
                1,
                manifest_dists=[DistributionFamily.GAUSSIAN],
                model=model,
            )
        )

        np.testing.assert_array_equal(dynamics_support, np.array([[True]]))
        np.testing.assert_array_equal(input_effect_support, np.array([[True]]))
        np.testing.assert_array_equal(lambda_mat, np.array([[1.0]]))
        np.testing.assert_array_equal(lambda_support, np.array([[False]]))
        assert edge_lag_days == {}

    def test_model_build_accepts_only_already_compiled_ssm_spec(self):
        """Runtime construction consumes an ModelSpec without structural authoring inputs."""
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

        model = build_ssm_model(X, model_spec=_make_3latent_spec())
        assert numeric.n_states(model.spec) == 3

    def test_model_build_has_no_autodetect_path(self):
        """Runtime construction requires an already compiled ModelSpec."""
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

        with pytest.raises(TypeError, match="model_spec"):
            cast("Any", build_ssm_model)(X)

    @pytest.mark.parametrize("source_count", [1, 2])
    def test_translate_spec_compiles_static_baseline_factor_from_induced_dependency(
        self, source_count
    ):

        model = complete_test_model(
            ModelSpec.model_validate(
                {
                    "default_outcome": {
                        "kind": "construct",
                        "id": "construct:cdc0b2958a9512b2abad",
                    },
                    "edges": [
                        {
                            "cause": {
                                "id": "construct:766f6091724c163a3404",
                                "name": "u_shared",
                                "description": "Shared static confounder",
                                "role": "exogenous",
                                "temporal_status": "time_invariant",
                            },
                            "effect": {
                                "id": "construct:6b04dc42c531e7091eb8",
                                "name": "stress",
                                "description": "Stress",
                                "role": "endogenous",
                                "temporal_status": "time_varying",
                                "indicators": [
                                    {
                                        "id": "indicator:3696aef3ff6f446744e5",
                                        "name": "stress_score",
                                        "construct_polarity": "positive",
                                        "how_to_measure": "measure stress",
                                        "measurement_dtype": "continuous",
                                        "aggregation": "mean",
                                    }
                                ],
                            },
                            "id": "edge:0e72c3c33116c415bd39",
                            "description": "Shared baseline causes stress",
                        },
                        {
                            "cause": {"kind": "construct", "id": "construct:766f6091724c163a3404"},
                            "effect": {
                                "id": "construct:cdc0b2958a9512b2abad",
                                "name": "sleep",
                                "description": "Sleep",
                                "role": "endogenous",
                                "temporal_status": "time_varying",
                                "indicators": [
                                    {
                                        "id": "indicator:7f807162156d3eb1b611",
                                        "name": "sleep_score",
                                        "construct_polarity": "positive",
                                        "how_to_measure": "measure sleep",
                                        "measurement_dtype": "continuous",
                                        "aggregation": "mean",
                                    }
                                ],
                            },
                            "id": "edge:a1528b0e1cd05dd511ef",
                            "description": "Shared baseline causes sleep",
                        },
                    ],
                    "measurement_clock": "1d",
                }
            )
        )
        if source_count == 2:
            source = model.get_construct("construct:766f6091724c163a3404")
            second = source.model_copy(
                update={"id": "construct:second-common-cause", "name": "second_common_cause"}
            )
            model = model.revised(
                edges=replace_constructs(
                    (
                        *model.edges,
                        *(
                            edge.model_copy(update={"id": f"{edge.id}-second", "cause": second})
                            for edge in model.edges
                        ),
                    ),
                    (*model.constructs, second),
                )
            )
            # Equivalent marginalized roots reference one aggregate scale, not two draws.
            assert second.initial_state.scale == source.initial_state.scale
        model.check_execution()
        (model).require_execution_structure()
        spec, _ = (model, numeric.edge_lag_days(model))
        np.testing.assert_array_equal(numeric.static_scale_block(spec).free_support, [True])
        np.testing.assert_allclose(numeric.static_scale_block(spec).template, np.zeros(1))
        np.testing.assert_allclose(numeric.static_factor_loadings(spec), [[1.0], [1.0]])
        assert numeric.static_factor_names(spec) == ["tau_u_shared"]
        np.testing.assert_array_equal(
            numeric.initial_covariance_block(spec).correlation_support, np.zeros((2, 2), dtype=bool)
        )
        np.testing.assert_array_equal(numeric.initial_mean_block(spec).free_support, [False, False])
        np.testing.assert_array_equal(
            numeric.initial_covariance_block(spec).diag_support, [False, False]
        )

    def test_translate_spec_marks_standardizable_gaussian_mean_indicators(self):

        plan = _make_model()
        model = complete_test_model(plan)
        spec, _ = (model, numeric.edge_lag_days(model))
        assert numeric.observation_standardized(spec) == [True, True, True, True]

    def test_translate_spec_fixes_manifest_noise_for_single_indicator_constructs(self):

        plan = _make_model()
        model = complete_test_model(plan)
        spec, _ = (model, numeric.edge_lag_days(model))
        assert isinstance(numeric.observation_noise_block(spec).template, jnp.ndarray)
        np.testing.assert_array_equal(
            numeric.observation_noise_block(spec).diag_support, [True, True, False, False]
        )
        np.testing.assert_allclose(numeric.observation_noise_block(spec).template, np.zeros((4, 4)))

    def test_translate_spec_rejects_initial_state_correlation_parameters_with_scientific_model(
        self,
    ):
        from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
        from tests.slot_fixtures import fixture_parameter_id

        plan = _make_model()
        model = complete_test_model(plan)
        owners = tuple(ConstructRef(id=model.constructs[i].id) for i in (0, 2))
        parameter = ParameterSpec(
            id=fixture_parameter_id(SiteKind.T0_VAR_LOWER, owners),
            name="cor0",
            description="Unsupported pairwise initial correlation",
            distribution_transform=PriorAuthoringTransform.INITIAL_STATE_CORRELATION,
        )
        from nof1_causal_lab.artifacts.coefficient import ParameterCoefficient
        from tests.slot_fixtures import attach_test_coefficients

        model = attach_test_coefficients(
            model,
            [(SiteKind.T0_VAR_LOWER, owners, ParameterCoefficient(parameter_id=parameter.id))],
            parameters=(parameter,),
        )
        with pytest.raises(
            ValueError, match=r"explicit latent confounder|two distinct state owners"
        ):
            numeric.validate_execution(model)

    def test_translate_spec_rejects_self_initial_state_correlation_with_scientific_model(self):
        from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
        from tests.slot_fixtures import fixture_parameter_id

        plan = _make_model()
        model = complete_test_model(plan)
        owners = tuple(ConstructRef(id=model.constructs[i].id) for i in (0,))
        parameter = ParameterSpec(
            id=fixture_parameter_id(SiteKind.T0_VAR_LOWER, owners),
            name="cor0",
            description="Unsupported pairwise initial correlation",
            distribution_transform=PriorAuthoringTransform.INITIAL_STATE_CORRELATION,
        )
        from nof1_causal_lab.artifacts.coefficient import ParameterCoefficient
        from tests.slot_fixtures import attach_test_coefficients

        model = attach_test_coefficients(
            model,
            [(SiteKind.T0_VAR_LOWER, owners, ParameterCoefficient(parameter_id=parameter.id))],
            parameters=(parameter,),
        )
        with pytest.raises(
            ValueError, match=r"explicit latent confounder|two distinct state owners"
        ):
            numeric.validate_execution(model)

    def test_model_build_end_to_end(self):
        from nof1_causal_lab.models.model_checks import check_execution
        from nof1_causal_lab.models.ssm.runtime import build_ssm_model

        plan = _make_model()
        science = complete_test_model(plan)
        wide = pl.DataFrame(
            {
                "time": list(range(10)),
                "x1": [1.0] * 10,
                "x2": [2.0] * 10,
                "y1": [3.0] * 10,
                "z1": [4.0] * 10,
            }
        )
        check_execution(science)
        runtime = build_ssm_model(wide, model_spec=science)
        spec = runtime.spec
        assert (
            sum(
                site.site_kind == SiteKind.DYNAMICS_DECAY
                for site in numeric.iter_sample_sites(spec)
            )
            == 3
        )
        assert numeric.loading_block(spec).free_support is not None
        assert numeric.n_states(spec) == 3
        assert numeric.n_observations(spec) == 4


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

        spec = model_fixture(
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

        weight_sites = sorted(
            site.name for site in registry.values() if site.site_kind == SiteKind.DYNAMICS_WEIGHT
        )
        assert len(weight_sites) == 2
        assert {
            site.positions[0] for site in build_site_registry(spec) if site.name in weight_sites
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

        lambda_mat = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        lambda_support = np.array([[False, False], [False, False], [True, False]])

        spec = model_fixture(
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


class TestGradualBuildComponents:
    """Quartic self-limitation and Hill edges materialize from the ModelSpec."""

    def test_quartic_freed_only_for_self_limiting_construct(self):

        from nof1_causal_lab.artifacts.coefficient import FixedCoefficient, ParameterCoefficient
        from nof1_causal_lab.artifacts.expressions import expression_coefficients

        model = _make_model()
        model = complete_test_model(model, self_limiting=(model.state_order[1],))
        quartics = {
            component.target: next(
                operand.coefficient
                for operand in expression_coefficients(component.expression)
                if operand.role == "quartic"
            )
            for component in numeric.dynamics_expressions(model)
            if not component.edge_owned
        }
        assert isinstance(quartics[1], ParameterCoefficient)
        assert quartics[0] == FixedCoefficient(value=0)
        assert quartics[2] == FixedCoefficient(value=0)

    def test_hill_edge_emitted_for_saturating_edge(self):
        from nof1_causal_lab.artifacts.expressions import hill_applications

        model = _make_model()
        model = complete_test_model(model, hill_edges=(model.edges[0].id,))
        edge_components = [item for item in numeric.dynamics_expressions(model) if item.edge_owned]
        hill = [item for item in edge_components if any(hill_applications(item.expression))]
        linear = [item for item in edge_components if not any(hill_applications(item.expression))]
        assert [(item.source, item.target) for item in hill] == [(0, 1)]
        assert (1, 2) in [(item.source, item.target) for item in linear]
        assert (0, 1) not in [(item.source, item.target) for item in linear]

    @pytest.mark.cpu_expensive
    def test_freed_quartic_and_hill_sites_sample_finite(self):

        plan = _make_model()
        science = complete_test_model(
            plan, self_limiting=(plan.state_order[1],), hill_edges=(plan.edges[0].id,)
        )
        spec, _ = (science, numeric.edge_lag_days(science))
        model = SSMModel(spec)
        trace = handlers.trace(handlers.seed(model.model, random.PRNGKey(0))).get_trace(
            observations=jnp.zeros((8, 4)),
            times=jnp.arange(8, dtype=jnp.float32),
            likelihood_backend=get_laplace_backend(model, 6),
        )
        quartic_sites = [n for n in trace if n.startswith("vf_") and n.endswith("_quartic")]
        emax_sites = [n for n in trace if n.startswith("vf_") and n.endswith("_Emax")]
        assert len(quartic_sites) == 1
        assert len(emax_sites) == 1
        for name in quartic_sites + emax_sites:
            assert bool(jnp.all(jnp.isfinite(trace[name]["value"])))
