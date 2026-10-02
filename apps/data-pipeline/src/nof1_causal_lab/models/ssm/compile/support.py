"""Pure derivations of numerical support from scientific structure and policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.expressions import COEFFICIENT_MEANINGS
from nof1_causal_lab.artifacts.likelihood import OBSERVATION_FAMILY_SPECS, CategoricalLawSpec
from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.compilation_errors import IncompleteModelError
from nof1_causal_lab.models.model_parameters import coefficient_value, execution_coefficient_uses
from nof1_causal_lab.models.model_structure import selected_indicators, selected_state_ids
from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
from nof1_causal_lab.models.ssm.structure import (
    DiffusionBlockSpec,
    ManifestCholBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from nof1_causal_lab.artifacts.identity import ConstructId, IndicatorId, ParameterId
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.model_parameters import CoefficientUse
    from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponentSpec
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor

from nof1_causal_lab.artifacts.likelihood import DistributionFamily
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.model_semantics import (
    indicator_requires_observation_intercept,
)
from nof1_causal_lab.models.ssm.structure.sites import make_site
from nof1_causal_lab.utils.model_structure import (
    get_model_clock,
    get_reference_indicator_lookup,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


class NumericalSupportError(AggregatedCompileError):
    """Aggregate failures to derive numerical support from ModelSpec."""

    header = "Numerical derivation failed"


def get_construct_dt_days(
    model: ModelSpec,
) -> float:
    """Get the model clock interval in fractional days."""
    return get_model_clock(model).days


def categorical_anchors(model: ModelSpec) -> tuple[bool, ...]:
    """Pin the reference contrast only for states with exclusively nominal emissions."""
    references = get_reference_indicator_lookup(model)
    categorical_states = {
        identity
        for identity in selected_state_ids(model)
        if all(
            indicator.likelihood is not None
            and indicator.likelihood.law.family == DistributionFamily.CATEGORICAL
            for indicator in model.get_construct(identity).indicators
        )
    }
    return tuple(
        model.indicator_owner(indicator.id).id in categorical_states
        and indicator.name == references[model.indicator_owner(indicator.id).name]
        for indicator in observed_indicators(model)
    )


def _build_static_factor_structure(
    model: ModelSpec, latent_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Derive baseline-factor incidence and fixed loadings from explicit common causes."""
    from nof1_causal_lab.artifacts.expressions import linear_coefficient
    from nof1_causal_lab.models.model_parameters import baseline_factor_groups, coefficient_value

    groups = baseline_factor_groups(model)
    factors = [group[0] for group in groups]
    state_index = {identity: index for index, identity in enumerate(selected_state_ids(model))}
    loadings = np.zeros((len(latent_names), len(factors)))
    for index, group in enumerate(groups):
        sources = {construct.id for construct in group}
        for source in sources:
            construct = model.get_construct(source)
            initial_mean = construct.coefficient("initial_mean")
            assert initial_mean is not None
            if coefficient_value(initial_mean) != 0.0:
                raise NumericalSupportError(
                    ["Marginalized baseline factors require a fixed zero initial mean"]
                )
            if (
                construct.indicators
                or construct.temporal_status != "time_invariant"
                or any(edge.effect.id == source for edge in model.edges)
            ):
                raise NumericalSupportError(
                    ["A marginalized baseline factor must be an unmeasured time-invariant root"]
                )
            for edge in model.edges:
                if edge.cause.id != source or edge.effect.id not in state_index:
                    continue
                weight = 1.0
                if edge.mechanisms:
                    if len(edge.mechanisms) != 1:
                        raise NumericalSupportError(
                            ["A marginalized factor loading requires one fixed linear coefficient"]
                        )
                    coefficient = coefficient_value(
                        linear_coefficient(edge.mechanisms[0].expression, source)
                    )
                    if coefficient is None:
                        raise NumericalSupportError(
                            ["A marginalized factor loading requires one fixed linear coefficient"]
                        )
                    weight = coefficient
                loadings[state_index[edge.effect.id], index] = weight
        if not np.any(loadings[:, index]):
            raise NumericalSupportError(["A baseline factor must affect a retained state"])
    return (
        np.ones(len(factors), dtype=bool),
        np.zeros(len(factors)),
        np.asarray(loadings),
        tuple(factor.name for factor in factors),
    )


def state_ids(model: ModelSpec) -> tuple[ConstructId, ...]:
    return selected_state_ids(model)


def state_names(model: ModelSpec) -> tuple[str, ...]:
    return tuple(model.get_construct(identity).name for identity in state_ids(model))


def n_states(model: ModelSpec) -> int:
    return len(state_ids(model))


def observed_indicators(model: ModelSpec) -> tuple[IndicatorSpec, ...]:
    return tuple(
        model.indicator(identity)
        for identity in tuple(indicator.id for indicator in selected_indicators(model))
    )


def observation_ids(model: ModelSpec) -> tuple[IndicatorId, ...]:
    return tuple(indicator.id for indicator in observed_indicators(model))


def observation_names(model: ModelSpec) -> tuple[str, ...]:
    return tuple(indicator.name for indicator in observed_indicators(model))


def n_observations(model: ModelSpec) -> int:
    return len(observed_indicators(model))


def _likelihoods(model: ModelSpec) -> Iterator[LikelihoodSpec]:
    for indicator in observed_indicators(model):
        if indicator.likelihood is None:
            raise IncompleteModelError(
                f"Retained indicator {indicator.name!r} requires a likelihood"
            )
        yield indicator.likelihood


def observation_families(model: ModelSpec) -> tuple[DistributionFamily, ...]:
    return tuple(likelihood.law.family for likelihood in _likelihoods(model))


def observation_level_counts(model: ModelSpec) -> tuple[int, ...]:
    return tuple(
        len(indicator.ordinal_levels or indicator.categorical_levels or ())
        if indicator.likelihood is not None
        and indicator.likelihood.law.family
        in {DistributionFamily.ORDERED_LOGISTIC, DistributionFamily.CATEGORICAL}
        else 0
        for indicator in observed_indicators(model)
    )


def quantity_position(model: ModelSpec, parameter: CoefficientUse) -> tuple[int, ...]:
    """Locate a scientific scalar by its owners in the derived execution axes."""
    state = {key: i for i, key in enumerate(state_ids(model))}
    observation = {key: i for i, key in enumerate(observation_ids(model))}
    owners = {owner.id for owner in parameter.owners}

    def one(axis: Mapping[ConstructId, int] | Mapping[IndicatorId, int]) -> int:
        matches = [index for key, index in axis.items() if key in owners]
        if len(matches) != 1:
            raise NumericalSupportError(
                [f"Quantity {parameter.slot!r} needs one owner on this axis"]
            )
        return matches[0]

    kind = parameter.quantity
    if kind == SiteKind.LOADING:
        return (one(observation), one(state))
    if kind in {SiteKind.DIFFUSION_LOWER, SiteKind.T0_VAR_LOWER}:
        pair = sorted(index for key, index in state.items() if key in owners)
        if len(pair) != 2:
            raise NumericalSupportError(
                [f"Covariance quantity {parameter.slot!r} requires two distinct state owners"]
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
            raise NumericalSupportError(["A baseline scale must belong to one identifiable factor"])
        return (matches[0],)
    return (one(state),)


def _quantity_values(
    model: ModelSpec,
    kind: SiteKind,
    template: np.ndarray,
    support: np.ndarray,
    *,
    diagonal: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply explicit scalar definitions to the scientific default policy."""
    values = np.array(template, dtype=float, copy=True)
    free = np.array(support, dtype=bool, copy=True)
    occupied = {}
    for parameter in execution_coefficient_uses(model):
        if parameter.quantity != kind:
            continue
        if kind == SiteKind.T0_MEANS and not any(
            owner.id in state_ids(model) for owner in parameter.owners
        ):
            continue
        position = quantity_position(model, parameter)
        value = coefficient_value(parameter.value)
        if (
            kind in {SiteKind.DIFFUSION_DIAG, SiteKind.DIFFUSION_LOWER}
            and any(time_invariant_mask(model)[index] for index in position)
            and value != 0.0
        ):
            raise NumericalSupportError(["Time-invariant constructs cannot have innovations"])
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
                raise NumericalSupportError(
                    [
                        "Correlated quantities require an explicit latent confounder in the scientific DAG"
                    ]
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
            raise NumericalSupportError(["A fixed standard deviation cannot be negative"])
        if kind == SiteKind.STATIC_STATE_SD and occupied.get(position) == parameter.value:
            continue
        if position in occupied:
            raise NumericalSupportError(
                [f"Multiple scientific definitions for {kind.value} at {position}"]
            )
        occupied[position] = parameter.value
        index = (position[0], position[0]) if diagonal else position
        support_index = index if free.ndim == len(index) else position
        free[support_index] = value is None
        if value is not None:
            values[index] = value
    return values, free


def loading_block(model: ModelSpec) -> SparseMatrixBlockSpec:
    shape = (n_observations(model), n_states(model))
    template, support = _quantity_values(
        model, SiteKind.LOADING, np.zeros(shape), np.zeros(shape, dtype=bool)
    )
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
    inactive = [
        indicator.name
        for indicator in observed_indicators(model)
        if indicator.likelihood is not None
        and isinstance(indicator.likelihood.parsed.intercept.value, str)
        and not indicator_requires_observation_intercept(
            indicator.likelihood.law.family,
            indicator.likelihood.parsed.link,
            standardized=indicator.likelihood.standardized,
        )
    ]
    if inactive:
        raise NumericalSupportError(
            [
                f"Observation intercept 'manifest_mean_{name}' is inactive for the locked likelihood "
                "semantics; fix the channel location through its compiler-owned anchor instead."
                for name in inactive
            ]
        )
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
    n = n_observations(model)
    template, support = _quantity_values(
        model, SiteKind.MANIFEST_VAR_DIAG, np.zeros((n, n)), np.zeros(n, dtype=bool), diagonal=True
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


def input_mask(model: ModelSpec) -> np.ndarray:
    """Coordinates read from the panel instead of generated under a state law."""
    return np.asarray(
        [model.get_construct(identity).role == "exogenous" for identity in state_ids(model)],
        dtype=bool,
    )


def diffusion_families(model: ModelSpec) -> tuple[DistributionFamily, ...]:
    from nof1_causal_lab.distributions import DistributionFamily

    result = []
    for identity in state_ids(model):
        construct = model.get_construct(identity)
        if construct.role == "exogenous" or construct.temporal_status == "time_invariant":
            result.append(DistributionFamily.GAUSSIAN)
        elif construct.coefficient("diffusion_scale") is None:
            raise IncompleteModelError(f"Construct {construct.name!r} requires a diffusion scale")
        else:
            result.append(construct.innovation_family)
    return tuple(result)


def diffusion_block(model: ModelSpec) -> DiffusionBlockSpec:
    count = n_states(model)
    support = np.eye(count, dtype=bool)
    axis = {identity: index for index, identity in enumerate(state_ids(model))}
    for first_id, second_id, kind in model.induced_dependencies:
        if kind == "innovation_correlation":
            first, second = axis[first_id], axis[second_id]
            support[max(first, second), min(first, second)] = True
    static = time_invariant_mask(model) | input_mask(model)
    support[static, :] = False
    support[:, static] = False
    template, support = _quantity_values(
        model, SiteKind.DIFFUSION_DIAG, np.eye(count), np.zeros_like(support), diagonal=True
    )
    template, support = _quantity_values(model, SiteKind.DIFFUSION_LOWER, template, support)
    template[static, :] = 0.0
    template[:, static] = 0.0
    support[static, :] = False
    support[:, static] = False
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
    std[input_mask(model)] = 0.0
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


def static_factor_ids(model: ModelSpec) -> tuple[ConstructId, ...]:
    from nof1_causal_lab.models.model_parameters import baseline_factor_groups

    return tuple(group[0].id for group in baseline_factor_groups(model))


def static_factor_names(model: ModelSpec) -> tuple[str, ...]:

    names = []
    for identity in static_factor_ids(model):
        construct = model.get_construct(identity)
        coefficient = construct.coefficient("initial_scale")
        assert coefficient is not None
        names.append(
            model.parameter(coefficient).name if isinstance(coefficient, str) else construct.name
        )
    return tuple(names)


def static_factor_loadings(model: ModelSpec) -> jnp.ndarray:
    return jnp.asarray(_build_static_factor_structure(model, state_names(model))[2])


def static_scale_block(model: ModelSpec) -> SparseVectorBlockSpec:
    n = len(static_factor_ids(model))
    values, support = _quantity_values(
        model, SiteKind.STATIC_STATE_SD, np.zeros(n), np.ones(n, dtype=bool)
    )
    return SparseVectorBlockSpec(
        n=len(values),
        free_support=support,
        template=jnp.asarray(values),
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


def parameter_blocks(
    model: ModelSpec,
) -> tuple[
    DiffusionBlockSpec,
    SparseMatrixBlockSpec,
    SparseVectorBlockSpec,
    ManifestCholBlockSpec,
    SparseVectorBlockSpec,
    T0CholBlockSpec,
    SparseVectorBlockSpec,
]:
    return (
        diffusion_block(model),
        loading_block(model),
        observation_mean_block(model),
        observation_noise_block(model),
        initial_mean_block(model),
        initial_covariance_block(model),
        static_scale_block(model),
    )


def _require_execution_choices(model: ModelSpec) -> None:
    """Check numerical execution requirements."""
    from nof1_causal_lab.models.model_structure import validate_execution_structure

    validate_execution_structure(model)
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.ssm.compile.support import (
        NumericalSupportError,
    )
    from nof1_causal_lab.models.ssm.execution.observation_families import (
        supported_distribution_families,
    )

    def require_hyperparameter(coefficient: float | ParameterId | None, label: str) -> None:
        if not isinstance(coefficient, str):
            raise IncompleteModelError(f"{label} requires a prior parameter")

    for identity in state_ids(model):
        construct = model.get_construct(identity)
        if construct.role == "exogenous":
            continue
        if any(construct.coefficient(role) is None for role in ("initial_mean", "initial_scale")):
            raise IncompleteModelError(
                f"Construct {construct.name!r} requires initial-state coefficients"
            )
        if (
            construct.temporal_status == "time_varying"
            and construct.coefficient("diffusion_scale") is None
        ):
            raise IncompleteModelError(
                f"Construct {construct.name!r} requires an innovation distribution"
            )
        if construct.innovation_family == DistributionFamily.STUDENT_T:
            require_hyperparameter(
                construct.coefficient("process_degrees_of_freedom"),
                f"{construct.name}.process_degrees_of_freedom",
            )
    supported = supported_distribution_families()
    for indicator in observed_indicators(model):
        likelihood = indicator.likelihood
        if likelihood is None:
            raise IncompleteModelError(
                f"Retained indicator {indicator.name!r} requires a likelihood"
            )
        if likelihood.law.family not in supported:
            raise NumericalSupportError(
                [f"Indicator {indicator.name!r} has no native emission function"]
            )
        terms = likelihood.parsed
        missing = [operand.role for operand in terms.operands if operand.value is None]
        if missing:
            raise IncompleteModelError(
                f"Indicator {indicator.name!r} requires explicit measurement coefficients: {missing}"
            )
        for operand in terms.auxiliary:
            if operand.role != "observation_scale":
                require_hyperparameter(operand.value, f"{indicator.name}.likelihood.{operand.role}")


def likelihood_sites(spec: ModelSpec) -> tuple[SiteDescriptor, ...]:
    """Declare emission/process hyperparameters in their native sampling order."""
    sites = []
    families = set(observation_families(spec))
    for law in OBSERVATION_FAMILY_SPECS:
        if len(law.parameter_roles) != 1:
            continue
        meaning = COEFFICIENT_MEANINGS[law.parameter_roles[0]]
        family, kind = law.family, meaning.quantity
        name = kind.value
        if family in families:
            sites.append(
                make_site(name, (), meaning.support, "likelihood", kind, priors_field=name)
            )
    n = n_observations(spec)
    cutpoints = max(max(observation_level_counts(spec), default=0) - 1, 0)
    if DistributionFamily.ORDERED_LOGISTIC in families and cutpoints:
        base = COEFFICIENT_MEANINGS["cutpoint_base"]
        sites.append(
            make_site(
                base.quantity.value,
                (n,),
                base.support,
                "likelihood",
                base.quantity,
                priors_field=base.quantity.value,
            )
        )
        if cutpoints > 1:
            gaps = COEFFICIENT_MEANINGS["cutpoint_gaps"]
            sites.append(
                make_site(
                    gaps.quantity.value,
                    (n, cutpoints - 1),
                    gaps.support,
                    "likelihood",
                    gaps.quantity,
                    priors_field=gaps.quantity.value,
                )
            )
    if DistributionFamily.CATEGORICAL in families and cutpoints:
        for role in CategoricalLawSpec.parameter_roles:
            meaning = COEFFICIENT_MEANINGS[role]
            kind = meaning.quantity
            name = kind.value
            sites.append(
                make_site(
                    name, (n, cutpoints), meaning.support, "likelihood", kind, priors_field=name
                )
            )
    if DistributionFamily.STUDENT_T in diffusion_families(spec):
        meaning = COEFFICIENT_MEANINGS["process_degrees_of_freedom"]
        sites.append(
            make_site(
                meaning.quantity.value,
                (),
                meaning.support,
                "likelihood",
                meaning.quantity,
                priors_field=meaning.quantity.value,
            )
        )
    return tuple(sites)
