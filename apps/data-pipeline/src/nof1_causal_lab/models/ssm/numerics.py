"""Numerical views derived from the one scientific ModelSpec.

No numerical view is a second model definition or a persisted specification.
These functions read canonical components and produce only the arrays,
blocks, and axis selections needed by an operation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.models.model_parameters import coefficient_value, iter_coefficient_uses
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.likelihood import LinkFunction
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponentSpec


def state_ids(model: ModelSpec) -> list[ConstructId]:
    return list(model.state_order)


def state_names(model: ModelSpec) -> list[str]:
    return [model.get_construct(identity).name for identity in state_ids(model)]


def n_states(model: ModelSpec) -> int:
    return len(state_ids(model))


def observed_indicators(model: ModelSpec) -> tuple[IndicatorSpec, ...]:
    return tuple(model.indicator(identity) for identity in model.manifest_indicator_order)


def observation_ids(model: ModelSpec) -> list[IndicatorId]:
    return [indicator.id for indicator in observed_indicators(model)]


def observation_names(model: ModelSpec) -> list[str]:
    return [indicator.name for indicator in observed_indicators(model)]


def n_observations(model: ModelSpec) -> int:
    return len(observed_indicators(model))


def _likelihoods(model: ModelSpec):
    for indicator in observed_indicators(model):
        if indicator.likelihood is None:
            raise IncompleteModelError(f"Retained indicator {indicator.id!r} requires a likelihood")
        yield indicator.likelihood


def observation_families(model: ModelSpec) -> list[DistributionFamily]:
    return [likelihood.law.family for likelihood in _likelihoods(model)]


def observation_links(model: ModelSpec) -> list[LinkFunction]:
    return [likelihood.terms.link for likelihood in _likelihoods(model)]


def observation_level_counts(model: ModelSpec) -> list[int]:
    from nof1_causal_lab.models.ssm.compile.support import (
        build_manifest_level_counts_from_model,
    )

    counts = build_manifest_level_counts_from_model(
        observation_names(model), observation_families(model), model=model
    )
    return [0] * n_observations(model) if counts is None else counts


def observation_standardized(model: ModelSpec) -> list[bool]:
    return [likelihood.standardized for likelihood in _likelihoods(model)]


def _structural_support(model: ModelSpec):
    from nof1_causal_lab.models.ssm.compile.support import (
        build_structural_support_from_model,
    )

    return build_structural_support_from_model(
        state_names(model),
        observation_names(model),
        n_states(model),
        n_observations(model),
        manifest_dists=observation_families(model),
        model=model,
    )


def categorical_anchors(model: ModelSpec) -> list[bool]:
    return _structural_support(model)[3].tolist()


def edge_lag_days(model: ModelSpec) -> dict[tuple[int, int], float]:
    return _structural_support(model)[4]


def quantity_position(model: ModelSpec, parameter) -> tuple[int, ...]:
    """Locate a scientific scalar by its owners in the derived execution axes."""
    state = {key: i for i, key in enumerate(state_ids(model))}
    observation = {key: i for i, key in enumerate(observation_ids(model))}
    owners = {owner.id for owner in parameter.owners}

    def one(axis):
        matches = [index for key, index in axis.items() if key in owners]
        if len(matches) != 1:
            raise ValueError(f"Quantity {parameter.slot!r} needs one owner on this axis")
        return matches[0]

    kind = parameter.quantity
    if kind == SiteKind.LOADING:
        return (one(observation), one(state))
    if kind in {SiteKind.DIFFUSION_LOWER, SiteKind.T0_VAR_LOWER}:
        pair = sorted(index for key, index in state.items() if key in owners)
        if len(pair) != 2:
            raise ValueError(
                f"Covariance quantity {parameter.slot!r} requires two distinct state owners"
            )
        return (pair[1], pair[0])
    if kind in {SiteKind.MANIFEST_MEANS, SiteKind.MANIFEST_VAR_DIAG}:
        return (one(observation),)
    if kind == SiteKind.STATIC_STATE_SD:
        from nof1_causal_lab.models.model_parameters import baseline_factor_groups

        matches = [
            index
            for index, group in enumerate(baseline_factor_groups(model))
            if owners & {construct.id for construct in group}
        ]
        if len(matches) != 1:
            raise ValueError("A baseline scale must belong to one identifiable factor")
        return (matches[0],)
    return (one(state),)


def _quantity_values(model: ModelSpec, kind: SiteKind, template, support, *, diagonal=False):
    """Apply explicit scalar definitions to the scientific default policy."""
    values = np.array(template, dtype=float, copy=True)
    free = np.array(support, dtype=bool, copy=True)
    occupied = {}
    for parameter in iter_coefficient_uses(model):
        if parameter.quantity != kind:
            continue
        if kind == SiteKind.T0_MEANS and not any(
            owner.id in state_ids(model) for owner in parameter.owners
        ):
            continue
        position = quantity_position(model, parameter)
        value = coefficient_value(model, parameter.value)
        if (
            kind in {SiteKind.DIFFUSION_DIAG, SiteKind.DIFFUSION_LOWER}
            and any(time_invariant_mask(model)[index] for index in position)
            and value != 0.0
        ):
            raise ValueError("Time-invariant constructs cannot have innovations")
        if kind in {SiteKind.DIFFUSION_LOWER, SiteKind.T0_VAR_LOWER}:
            expected_kind = (
                "innovation_correlation"
                if kind == SiteKind.DIFFUSION_LOWER
                else "initial_state_correlation"
            )
            pair = {state_ids(model)[index] for index in position}
            if not any(
                kind == expected_kind and {first, second} == pair
                for first, second, kind in model.induced_dependencies
            ):
                raise ValueError(
                    "Correlated quantities require an explicit latent confounder in the scientific DAG"
                )
        if (
            value is not None
            and kind
            in {
                SiteKind.DIFFUSION_DIAG,
                SiteKind.MANIFEST_VAR_DIAG,
                SiteKind.T0_VAR_DIAG,
                SiteKind.STATIC_STATE_SD,
            }
            and value < 0
        ):
            raise ValueError("A fixed standard deviation cannot be negative")
        if kind == SiteKind.STATIC_STATE_SD and occupied.get(position) == parameter.value:
            continue
        if position in occupied:
            raise ValueError(f"Multiple scientific definitions for {kind.value} at {position}")
        occupied[position] = parameter.value
        index = (position[0], position[0]) if diagonal else position
        support_index = index if free.ndim == len(index) else position
        free[support_index] = value is None
        if value is not None:
            values[index] = value
    return values, free


def loading_block(model: ModelSpec) -> SparseMatrixBlockSpec:
    _, template, support, _, _ = _structural_support(model)
    template, support = _quantity_values(model, SiteKind.LOADING, template, np.zeros_like(support))
    return SparseMatrixBlockSpec(
        n_rows=n_observations(model),
        n_cols=n_states(model),
        free_support=support,
        template=jnp.asarray(template),
        free_site_name="lambda_free",
        det_site_name="lambda",
        support=SupportClass.REAL,
        site_kind=SiteKind.LOADING,
        assembly_group="lambda",
        fixed_spec_field="lambda_mat",
        priors_field="lambda_free",
    )


def observation_mean_block(model: ModelSpec) -> SparseVectorBlockSpec:
    template, support = _quantity_values(
        model,
        SiteKind.MANIFEST_MEANS,
        np.zeros(n_observations(model)),
        np.zeros(n_observations(model), dtype=bool),
    )
    return SparseVectorBlockSpec(
        n=n_observations(model),
        free_support=support,
        template=jnp.asarray(template),
        free_site_name="manifest_means_free",
        det_site_name="manifest_means",
        support=SupportClass.REAL,
        site_kind=SiteKind.MANIFEST_MEANS,
        assembly_group="manifest",
        fixed_spec_field="manifest_means",
        priors_field="manifest_means",
    )


def observation_noise_block(model: ModelSpec) -> ManifestCholBlockSpec:
    from nof1_causal_lab.models.ssm.compile.support import (
        build_manifest_variance_from_model,
    )

    template, support = build_manifest_variance_from_model(
        state_names(model),
        observation_names(model),
        observation_families(model),
        model=model,
    )
    template, support = _quantity_values(
        model, SiteKind.MANIFEST_VAR_DIAG, template, np.zeros_like(support), diagonal=True
    )
    return ManifestCholBlockSpec(
        n_manifest=n_observations(model), diag_support=support, template=jnp.asarray(template)
    )


def time_invariant_mask(model: ModelSpec) -> np.ndarray:
    return np.asarray(
        [
            model.get_construct(identity).temporal_status == "time_invariant"
            for identity in state_ids(model)
        ],
        dtype=bool,
    )


def diffusion_families(model: ModelSpec) -> list[DistributionFamily]:
    from nof1_causal_lab.distributions import DistributionFamily

    result = []
    for identity in state_ids(model):
        construct = model.get_construct(identity)
        if construct.temporal_status == "time_invariant":
            result.append(DistributionFamily.GAUSSIAN)
        elif construct.coefficient("diffusion_scale") is None:
            raise IncompleteModelError(f"Construct {identity!r} requires a diffusion scale")
        else:
            result.append(construct.innovation_family)
    return result


def diffusion_block(model: ModelSpec) -> DiffusionBlockSpec:
    count = n_states(model)
    support = np.eye(count, dtype=bool)
    axis = {identity: index for index, identity in enumerate(state_ids(model))}
    for first_id, second_id, kind in model.induced_dependencies:
        if kind == "innovation_correlation":
            first, second = axis[first_id], axis[second_id]
            support[max(first, second), min(first, second)] = True
    static = time_invariant_mask(model)
    support[static, :] = False
    support[:, static] = False
    template, support = _quantity_values(
        model, SiteKind.DIFFUSION_DIAG, np.eye(count), np.zeros_like(support), diagonal=True
    )
    template, support = _quantity_values(model, SiteKind.DIFFUSION_LOWER, template, support)
    return DiffusionBlockSpec(
        n_latent=count,
        diffusion_chol_support=support,
        diffusion_chol_template=jnp.asarray(template),
        time_invariant_mask=static if static.any() else None,
    )


def initial_mean_block(model: ModelSpec) -> SparseVectorBlockSpec:
    support = np.zeros(n_states(model), dtype=bool)
    template, support = _quantity_values(
        model, SiteKind.T0_MEANS, np.zeros(n_states(model)), support
    )
    return SparseVectorBlockSpec(
        n=n_states(model),
        free_support=support,
        template=jnp.asarray(template),
        free_site_name="t0_means_free",
        det_site_name="t0_means",
        support=SupportClass.REAL,
        site_kind=SiteKind.T0_MEANS,
        assembly_group="t0",
        fixed_spec_field="t0_means",
        priors_field="t0_means",
    )


def initial_covariance_block(model: ModelSpec) -> T0CholBlockSpec:
    count = n_states(model)
    support = np.zeros(count, dtype=bool)
    std, support = _quantity_values(model, SiteKind.T0_VAR_DIAG, np.ones(count), support)
    correlations, correlation_support = _quantity_values(
        model, SiteKind.T0_VAR_LOWER, np.eye(count), np.zeros((count, count), dtype=bool)
    )
    lower = np.tril(np.asarray(correlations), -1)
    corr = np.eye(count) + lower + lower.T
    template = jnp.asarray(np.asarray(std)[:, None] * np.linalg.cholesky(corr))
    return T0CholBlockSpec(
        n_latent=count,
        diag_support=support,
        correlation_support=correlation_support,
        template=jnp.asarray(template),
    )


def static_factor_ids(model: ModelSpec) -> list[ConstructId]:
    from nof1_causal_lab.models.model_parameters import baseline_factor_groups

    return [group[0].id for group in baseline_factor_groups(model)]


def static_factor_names(model: ModelSpec) -> list[str]:

    names = []
    for identity in static_factor_ids(model):
        construct = model.get_construct(identity)
        coefficient = construct.coefficient("initial_scale")
        assert coefficient is not None
        names.append(
            model.parameter(coefficient).name if isinstance(coefficient, str) else construct.name
        )
    return names


def _static_structure(model: ModelSpec):
    from nof1_causal_lab.models.ssm.compile.support import _build_static_factor_structure

    return _build_static_factor_structure(model, state_names(model))


def static_factor_loadings(model: ModelSpec) -> jnp.ndarray:
    return jnp.asarray(_static_structure(model)[2])


def static_scale_block(model: ModelSpec) -> SparseVectorBlockSpec:
    support, values, _, _ = _static_structure(model)
    values, support = _quantity_values(model, SiteKind.STATIC_STATE_SD, values, support)
    return SparseVectorBlockSpec(
        n=len(values),
        free_support=support,
        template=values,
        free_site_name="static_state_sd_free",
        det_site_name="static_state_sds",
        support=SupportClass.POSITIVE,
        site_kind=SiteKind.STATIC_STATE_SD,
        assembly_group="t0",
        fixed_spec_field="static_state_sds",
        priors_field="static_state_sd",
    )


def dynamics_expressions(model: ModelSpec) -> tuple[ExpressionComponentSpec, ...]:
    from nof1_causal_lab.models.ssm.compile.mechanisms import lower_mechanisms

    return lower_mechanisms(model)


def dynamics_components(model: ModelSpec) -> DynamicsSpec:
    return DynamicsSpec(n_latent=n_states(model), components=dynamics_expressions(model))


def parameter_blocks(model: ModelSpec):
    return (
        diffusion_block(model),
        loading_block(model),
        observation_mean_block(model),
        observation_noise_block(model),
        initial_mean_block(model),
        initial_covariance_block(model),
        static_scale_block(model),
    )


def iter_sample_sites(model: ModelSpec):
    for index, component in enumerate(dynamics_expressions(model)):
        yield from component.iter_sites(prefix=f"vf_{index}", n_latent=n_states(model))
    for block in parameter_blocks(model):
        yield from block.iter_sites()


def validate_execution(model: ModelSpec) -> None:
    """Check that the scientific value supplies everything numerical execution needs."""
    model.require_execution_structure()
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.ssm.compile.structural import compile_anchor_certificates
    from nof1_causal_lab.models.ssm.compile.support import (
        NumericalSupportError,
        _build_manifest_intercept_support,
    )
    from nof1_causal_lab.models.ssm.execution.observation_families import (
        supported_distribution_families,
    )

    def require_hyperparameter(coefficient, label):
        if not isinstance(coefficient, str):
            raise IncompleteModelError(f"{label} requires a prior parameter")
        if model.parameter(coefficient).value is not None:
            raise ValueError("Native distribution hyperparameters require prior laws")

    _, intercept_errors = _build_manifest_intercept_support(
        model, observation_names(model), observation_standardized(model)
    )
    if intercept_errors:
        raise NumericalSupportError(intercept_errors)
    for identity in state_ids(model):
        construct = model.get_construct(identity)
        if any(construct.coefficient(role) is None for role in ("initial_mean", "initial_scale")):
            raise IncompleteModelError(
                f"Construct {identity!r} requires initial-state coefficients"
            )
        if (
            construct.temporal_status == "time_varying"
            and construct.coefficient("diffusion_scale") is None
        ):
            raise IncompleteModelError(
                f"Construct {identity!r} requires an innovation distribution"
            )
        if construct.innovation_family == DistributionFamily.STUDENT_T:
            require_hyperparameter(
                construct.coefficient("process_degrees_of_freedom"),
                f"{identity}.process_degrees_of_freedom",
            )
    supported = supported_distribution_families()
    for indicator in observed_indicators(model):
        likelihood = indicator.likelihood
        if likelihood is None:
            raise IncompleteModelError(f"Retained indicator {indicator.id!r} requires a likelihood")
        if likelihood.law.family not in supported:
            raise ValueError(f"Indicator {indicator.id!r} has no native emission function")
        terms = likelihood.terms
        missing = [operand.role for operand in terms.operands if operand.value is None]
        if missing:
            raise IncompleteModelError(
                f"Indicator {indicator.id!r} requires explicit measurement coefficients: {missing}"
            )
        for operand in terms.auxiliary:
            if operand.role != "observation_scale":
                require_hyperparameter(operand.value, f"{indicator.id}.likelihood.{operand.role}")
    observation_level_counts(model)
    parameter_blocks(model)
    dynamics_components(model)
    compile_anchor_certificates(model)
