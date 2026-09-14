"""Test-owned ModelSpec construction helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import dynestyx as dsx
import jax.numpy as jnp
import jax.random as random
import jax.scipy.linalg as jla
import numpy as np
from dynestyx.inference.configs.discretizer import ExactAffineConfig

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    CoefficientExpression,
    StateExpression,
    linear_effect,
    map_expression,
)
from nof1_causal_lab.artifacts.expressions import (
    coefficient as expr_coefficient,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanismSpec
from nof1_causal_lab.artifacts.model_spec import ModelSpec
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.likelihoods import observation_law
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
    def configure(self, msg: dict[str, Any]):
        return _minimal_reparam(msg["fn"], msg.get("is_observed", False))


def zero_loading_support(n_manifest: int, n_latent: int) -> np.ndarray:
    return np.zeros((n_manifest, n_latent), dtype=bool)


def full_vector_support(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def zero_vector_support(n: int) -> np.ndarray:
    return np.zeros(n, dtype=bool)


def full_diagonal_support(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def zero_diagonal_support(n: int) -> np.ndarray:
    return np.zeros(n, dtype=bool)


def full_cholesky_support(n: int) -> np.ndarray:
    return np.tri(n, dtype=bool)


def zero_square_support(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=bool)


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

    spec = model_fixture(
        n_latent=n_latent,
        n_manifest=n_manifest,
        dynamics_spec=dense_matrix_dynamics_spec(
            n_latent=n_latent,
            decay_support=np.ones(n_latent, dtype=bool),
            edge_support=np.zeros((n_latent, n_latent), dtype=bool),
            coupling_template=jnp.zeros((n_latent, n_latent)),
            intercept_support=np.zeros(n_latent, dtype=bool),
            cint_template=jnp.zeros(n_latent),
        ),
        diffusion_block=DiffusionBlockSpec(
            n_latent=n_latent,
            diffusion_chol_support=np.diag(np.ones(n_latent, dtype=bool)),
            diffusion_chol_template=jnp.eye(n_latent),
        ),
        lambda_block=SparseMatrixBlockSpec(
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
        ),
        manifest_means_block=default_manifest_means_block(n_manifest),
        manifest_chol_block=default_manifest_chol_block(n_manifest),
        t0_means_block=SparseVectorBlockSpec(
            n=n_latent,
            free_support=np.zeros(n_latent, dtype=bool),
            template=jnp.zeros(n_latent),
            free_site_name="t0_means_free",
            det_site_name="t0_means",
            support=SupportClass.REAL,
            site_kind=SiteKind.T0_MEANS,
            assembly_group="t0",
            fixed_spec_field="t0_means",
            priors_field="t0_means",
        ),
        t0_chol_block=T0CholBlockSpec(
            n_latent=n_latent,
            diag_support=np.zeros(n_latent, dtype=bool),
            correlation_support=np.zeros((n_latent, n_latent), dtype=bool),
            template=jnp.eye(n_latent),
        ),
        static_state_sd_block=default_static_state_sd_block(),
    )

    return {
        "observations": observations,
        "times": times,
        "spec": spec,
        "true_decay_diag": decay_diag,
        "true_diff_diag": diff_sd,
        "true_obs_sd": obs_sd,
        "n_latent": n_latent,
    }


def model_fixture(
    *,
    n_latent: int,
    dynamics_spec: DynamicsSpec,
    n_manifest: int | None = None,
    diffusion_block: DiffusionBlockSpec | None = None,
    lambda_block: SparseMatrixBlockSpec | None = None,
    manifest_means_block: SparseVectorBlockSpec | None = None,
    manifest_chol_block: ManifestCholBlockSpec | None = None,
    t0_means_block: SparseVectorBlockSpec | None = None,
    t0_chol_block: T0CholBlockSpec | None = None,
    static_state_sd_block: SparseVectorBlockSpec | None = None,
    static_factor_loadings: jnp.ndarray | None = None,
    **metadata: Any,
) -> ModelSpec:
    """Author a scientific test model from expected numerical block values.

    Blocks are fixture inputs only. The returned value contains scientific
    entities, coefficient references, priors and constants, with no native spec.
    """
    from nof1_causal_lab.artifacts.construct import CausalEdgeSpec, ConstructSpec
    from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, IndicatorRef, MechanismRef
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.models.prior_planning import complete_model
    from tests.slot_fixtures import fixture_parameter_id

    n_manifest = n_latent if n_manifest is None else n_manifest
    axes = native_axis_metadata(n_latent, n_manifest, metadata)
    names, ids = axes.pop("latent_names"), axes.pop("latent_ids")
    obs_names, obs_ids = axes.pop("manifest_names"), axes.pop("manifest_ids")
    families, links = resolve_manifest_families_and_links(
        axes.pop("manifest_dists", [DistributionFamily.GAUSSIAN] * n_manifest),
        manifest_links=axes.pop("manifest_links", None),
    )
    counts = axes.pop("manifest_level_counts", None) or [0] * n_manifest
    standardized = axes.pop("manifest_standardized", None) or [False] * n_manifest
    axes.pop("manifest_cat_anchor", None)  # Anchors derive from indicator ownership.
    static = axes.pop("time_invariant_mask", None)
    static = np.zeros(n_latent, dtype=bool) if static is None else np.asarray(static, dtype=bool)
    innovations = axes.pop("diffusion_dists", [DistributionFamily.GAUSSIAN] * n_latent)
    axes.pop("static_factor_ids", None)
    axes.pop("static_factor_names", None)
    loadings = lambda_block or default_lambda_block(n_manifest, n_latent)
    owners = [
        int(np.argmax(np.abs(np.asarray(loadings.template)[row])))
        if np.any(np.asarray(loadings.template)[row] != 0)
        else row % n_latent
        for row in range(n_manifest)
    ]
    indicators = []
    for i, (name, identity, family, link) in enumerate(
        zip(obs_names, obs_ids, families, links, strict=True)
    ):
        dtype = {
            "bernoulli": "binary",
            "poisson": "count",
            "negative_binomial": "count",
            "ordered_logistic": "ordinal",
            "categorical": "categorical",
        }.get(family.value, "continuous")
        levels = tuple(str(level) for level in range(counts[i]))
        indicators.append(
            IndicatorSpec(
                id=identity,
                name=name,
                how_to_measure="Read the fixture value",
                construct_polarity="negative"
                if float(loadings.template[i, owners[i]]) < 0
                else "positive",
                measurement_dtype=cast("MeasurementDtype", dtype),
                aggregation="last",
                ordinal_levels=levels if dtype == "ordinal" else None,
                categorical_levels=levels if dtype == "categorical" else None,
                likelihood=LikelihoodSpec(
                    law=observation_law(ids[owners[i]], family, link),
                    standardized=standardized[i],
                    reasoning="Test likelihood",
                ),
            )
        )
    constructs = [
        ConstructSpec(
            id=identity,
            name=name,
            description="Test state",
            role="endogenous",
            temporal_status="time_invariant" if static[i] else "time_varying",
            indicators=tuple(ind for j, ind in enumerate(indicators) if owners[j] == i),
        )
        for i, (name, identity) in enumerate(zip(names, ids, strict=True))
    ]
    edges = {}
    parameters = {}
    coefficient_recipes = []
    node_terms = {identity: [] for identity in ids}
    edge_terms = {}

    def edge(cause, effect, *, lagged=True):
        pair = (cause, effect)
        if pair not in edges:
            identity = fixture_entity_id("edge", f"{cause}:{effect}")
            edges[pair] = CausalEdgeSpec(
                id=identity,
                cause=next(item for item in constructs if item.id == cause),
                effect=next(item for item in constructs if item.id == effect),
                lagged=lagged,
                description="Fixture causal assumption",
            )
            edge_terms[identity] = []
        return edges[pair]

    outcome = ConstructSpec(
        id="construct:numerical_test_outcome",
        name="numerical_test_outcome",
        description="Unmeasured downstream response joining the numerical test states.",
        role="endogenous",
        temporal_status="time_varying",
    )
    constructs.append(outcome)
    node_terms[outcome.id] = []
    for identity in ids:
        edge(identity, outcome.id)

    def quantity(kind, refs, *, value=None, name=None):
        refs = tuple({ref.id: ref for ref in refs}.values())
        identity = fixture_parameter_id(kind, refs)
        parameters[identity] = ParameterSpec(
            id=identity,
            name=name or identity,
            description="Fixture quantity",
            value=value,
        )
        reference = identity
        coefficient_recipes.append((kind, refs, reference))
        return reference

    for index, component in enumerate(dynamics_spec.components):
        if static[component.target]:
            continue
        target = ids[component.target]
        mechanism_id = fixture_entity_id("mechanism", f"fixture-term:{index}")
        refs = [ConstructRef(id=target), MechanismRef(id=mechanism_id)]
        owner = None
        if component.source is not None:
            source = ids[component.source]
            owner = edge(source, target)
            refs.extend((ConstructRef(id=source), EdgeRef(id=owner.id)))
            for other in sorted(component.sources - {component.source, component.target}):
                edge(ids[other], target)
                refs.append(ConstructRef(id=ids[other]))
        references = {}

        def bind(node, component=component, references=references, refs=refs):
            if isinstance(node, StateExpression):
                return expr_state(ids[component.state_ids.index(node.construct_id)])
            if isinstance(node, CoefficientExpression) and isinstance(node.value, str):
                key = node.value
                if key not in references:
                    references[key] = quantity(node.meaning.quantity, refs)
                return expr_coefficient(references[key], node.role)
            return node

        term = DynamicsMechanismSpec(
            id=mechanism_id,
            kind=component.kind,
            expression=map_expression(component.expression, bind),
        )
        (node_terms[target] if owner is None else edge_terms[owner.id]).append(term)
    for identity in ids:
        if not node_terms[identity] and not static[ids.index(identity)]:
            node_terms[identity].append(
                DynamicsMechanismSpec(
                    id=fixture_entity_id("mechanism", identity + ":zero-drift"),
                    expression=expr_coefficient(0, "intercept"),
                )
            )
    constructs = [
        item.model_copy(update={"dynamics": tuple(node_terms[item.id])}) for item in constructs
    ]

    assert not axes, f"Unexpected fixture metadata: {sorted(axes)}"

    def confounder(label, children, *, invariant, weights=None):
        identity = fixture_entity_id("construct", label)
        constructs.append(
            ConstructSpec(
                id=identity,
                name=label,
                description="Explicit latent common cause",
                role="exogenous",
                temporal_status="time_invariant" if invariant else "time_varying",
            )
        )
        for index, child in enumerate(children):
            owner = edge(identity, child, lagged=False)
            if weights is not None:
                edge_terms[owner.id].append(
                    DynamicsMechanismSpec(
                        id=fixture_entity_id("mechanism", owner.id),
                        expression=linear_effect(owner.cause.id, weights[index]),
                    )
                )
        return identity

    diffusion = diffusion_block or default_diffusion_block(n_latent)
    initial_mean = t0_means_block or default_t0_means_block(n_latent)
    initial = t0_chol_block or default_t0_chol_block(n_latent)
    for row in range(n_latent):
        refs = [ConstructRef(id=ids[row])]
        quantity(
            SiteKind.DIFFUSION_DIAG,
            refs,
            value=0.0
            if static[row]
            else None
            if diffusion.diffusion_chol_support[row, row]
            else float(diffusion.diffusion_chol_template[row, row]),
        )
        quantity(
            SiteKind.T0_MEANS,
            refs,
            value=None if initial_mean.free_support[row] else float(initial_mean.template[row]),
        )
        covariance = np.asarray(initial.template) @ np.asarray(initial.template).T
        standard_deviation = np.sqrt(np.diag(covariance))
        quantity(
            SiteKind.T0_VAR_DIAG,
            refs,
            value=None if initial.diag_support[row] else float(standard_deviation[row]),
        )
        for col in range(row):
            pair = [ConstructRef(id=ids[row]), ConstructRef(id=ids[col])]
            if (
                (
                    diffusion.diffusion_chol_support[row, col]
                    or float(diffusion.diffusion_chol_template[row, col]) != 0
                )
                and not static[row]
                and not static[col]
            ):
                confounder(f"innovation_{row}_{col}", (ids[row], ids[col]), invariant=False)
                quantity(
                    SiteKind.DIFFUSION_LOWER,
                    pair,
                    value=None
                    if diffusion.diffusion_chol_support[row, col]
                    else float(diffusion.diffusion_chol_template[row, col]),
                )
            if initial.correlation_support[row, col] or covariance[row, col] != 0:
                confounder(f"initial_{row}_{col}", (ids[row], ids[col]), invariant=True)
                quantity(
                    SiteKind.T0_VAR_LOWER,
                    pair,
                    value=None
                    if initial.correlation_support[row, col]
                    else float(
                        covariance[row, col] / (standard_deviation[row] * standard_deviation[col])
                    ),
                )
    for row in range(n_manifest):
        for col in range(n_latent):
            if (
                loadings.free_support[row, col]
                or float(loadings.template[row, col]) != 0
                or col == owners[row]
            ):
                quantity(
                    SiteKind.LOADING,
                    [IndicatorRef(id=obs_ids[row]), ConstructRef(id=ids[col])],
                    value=None
                    if loadings.free_support[row, col]
                    else float(loadings.template[row, col]),
                )
        means = manifest_means_block or default_manifest_means_block(n_manifest)
        noise = manifest_chol_block or default_manifest_chol_block(n_manifest)
        refs = [IndicatorRef(id=obs_ids[row]), ConstructRef(id=ids[owners[row]])]
        quantity(
            SiteKind.MANIFEST_MEANS,
            refs,
            value=None if means.free_support[row] else float(means.template[row]),
        )
        if families[row].uses_manifest_noise:
            quantity(
                SiteKind.MANIFEST_VAR_DIAG,
                refs,
                value=None if noise.diag_support[row] else float(noise.template[row, row]),
            )
    scales = static_state_sd_block or default_static_state_sd_block()
    for index in range(scales.n):
        weights = np.asarray(static_factor_loadings)[:, index]
        children = [ids[i] for i in np.flatnonzero(weights)]
        identity = confounder(
            f"baseline_{index}", children, invariant=True, weights=weights[weights != 0].tolist()
        )
        quantity(
            SiteKind.STATIC_STATE_SD,
            [ConstructRef(id=identity)],
            value=None if scales.free_support[index] else float(scales.template[index]),
        )
    model = ModelSpec.model_construct(
        edges=replace_constructs(
            tuple(
                item.model_copy(update={"mechanisms": tuple(edge_terms[item.id])})
                for item in edges.values()
            ),
            constructs,
        ),
        parameters=tuple(parameters.values()),
        measurement_clock="1d",
    )
    from tests.slot_fixtures import attach_test_coefficients

    model = attach_test_coefficients(model, coefficient_recipes)
    model = model.revised(
        edges=replace_constructs(
            model.edges,
            tuple(
                construct.model_copy(
                    update={"innovation_family": innovations[ids.index(construct.id)]}
                )
                if construct.id in ids
                else construct
                for construct in model.constructs
            ),
        )
    )
    return complete_model(model)


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
