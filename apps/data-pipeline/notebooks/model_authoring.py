"""Offline model-authoring helpers for notebooks, evaluation fixtures and tests.

The product never builds models in Python: agents write them through `edit_model`.
These helpers only build and revise fixture models outside `src`.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import TYPE_CHECKING, TypedDict

from nof1_causal_lab.artifacts.expressions import (
    BinaryExpression,
    CallExpression,
    LiteralExpression,
    coefficient,
    state,
)
from nof1_causal_lab.artifacts.identity import ConstructId, scientific_id
from nof1_causal_lab.artifacts.likelihood import (
    VALID_LINKS_FOR_DISTRIBUTION,
    LikelihoodSpec,
    LinkFunction,
    ObservationLawSpec,
)
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.likelihoods import function

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from nof1_causal_lab.artifacts.construct import ConstructSpec
    from nof1_causal_lab.artifacts.expressions import CoefficientExpression, Expression
    from nof1_causal_lab.artifacts.identity import DistributionId, ParameterId
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.model_structure import DependencyKey
    from nof1_causal_lab.numpyro_json import NumPyroDistribution


def observation_law(
    construct_id: ConstructId,
    family: DistributionFamily,
    link: LinkFunction,
) -> ObservationLawSpec:
    """Construct an editable scientific formula with explicit unassigned operands."""
    if link not in VALID_LINKS_FOR_DISTRIBUTION[family]:
        raise ValueError(f"link {link.value!r} is invalid for {family.value}")
    if family == DistributionFamily.DELTA:
        return ObservationLawSpec(distribution="Delta", arguments={"v": state(construct_id)})
    predictor = coefficient(None, "observation_intercept") + coefficient(None, "loading") * state(
        construct_id
    )
    match link:
        case LinkFunction.LOG:
            response = function("exp", predictor)
        case LinkFunction.LOGIT:
            response = function("sigmoid", predictor)
        case LinkFunction.PROBIT:
            response = function("normal_cdf", predictor)
        case LinkFunction.INVERSE:
            response = LiteralExpression(value=1) / predictor
        case _:
            response = predictor
    match family:
        case DistributionFamily.GAUSSIAN:
            name, arguments = (
                "Normal",
                {"loc": predictor, "scale": coefficient(None, "observation_scale")},
            )
        case DistributionFamily.STUDENT_T:
            name, arguments = (
                "StudentT",
                {
                    "df": coefficient(None, "degrees_of_freedom"),
                    "loc": predictor,
                    "scale": coefficient(None, "observation_scale"),
                },
            )
        case DistributionFamily.POISSON:
            name, arguments = "Poisson", {"rate": response}
        case DistributionFamily.GAMMA:
            shape = coefficient(None, "shape")
            name, arguments = "Gamma", {"concentration": shape, "rate": shape / response}
        case DistributionFamily.BERNOULLI:
            name, arguments = (
                "Bernoulli",
                {"logits": predictor} if link == LinkFunction.LOGIT else {"probs": response},
            )
        case DistributionFamily.NEGATIVE_BINOMIAL:
            name, arguments = (
                "NegativeBinomial2",
                {"mean": response, "concentration": coefficient(None, "dispersion")},
            )
        case DistributionFamily.BETA:
            concentration = coefficient(None, "concentration")
            name, arguments = (
                "Beta",
                {
                    "concentration1": response * concentration,
                    "concentration0": (LiteralExpression(value=1) - response) * concentration,
                },
            )
        case DistributionFamily.ORDERED_LOGISTIC:
            name, arguments = (
                "OrderedLogistic",
                {
                    "predictor": predictor,
                    "cutpoints": function(
                        "ordered_cutpoints",
                        coefficient(None, "cutpoint_base"),
                        coefficient(None, "cutpoint_gaps"),
                    ),
                },
            )
        case DistributionFamily.CATEGORICAL:
            name, arguments = (
                "Categorical",
                {
                    "logits": function(
                        "category_logits",
                        predictor,
                        coefficient(None, "category_intercepts"),
                        coefficient(None, "category_slopes"),
                    )
                },
            )
    return ObservationLawSpec(distribution=name, arguments=arguments)


def map_expression(value: Expression, transform: Callable[[Expression], Expression]) -> Expression:
    """Revise references or constants without interpreting the relationship's function."""
    if isinstance(value, BinaryExpression):
        value = BinaryExpression(
            operator=value.operator,
            left=map_expression(value.left, transform),
            right=map_expression(value.right, transform),
        )
    elif isinstance(value, CallExpression):
        value = CallExpression(
            function=value.function,
            arguments=tuple(map_expression(argument, transform) for argument in value.arguments),
        )
    return transform(value)


def revise_law(
    likelihood: LikelihoodSpec, transform: Callable[[Expression], Expression]
) -> LikelihoodSpec:
    """Revise scientific operands while preserving the conditional formula."""
    law = ObservationLawSpec(
        distribution=likelihood.law.distribution,
        arguments={
            name: map_expression(value, transform)
            for name, value in likelihood.law.arguments.items()
        },
    )
    return LikelihoodSpec(
        law=law,
        standardized=likelihood.standardized,
        reasoning=likelihood.reasoning,
        sources=likelihood.sources,
    )


def with_construct_coefficients(
    construct: ConstructSpec, *operands: CoefficientExpression
) -> ConstructSpec:
    """Replace the specified uses while preserving other authored coefficients."""
    values = {(operand.role, operand.construct_ids): operand for operand in construct.coefficients}
    values.update({(operand.role, operand.construct_ids): operand for operand in operands})
    return type(construct).model_validate(
        {**construct.model_dump(), "coefficients": tuple(values.values())}
    )


def parameter_distribution_id(parameter_id: ParameterId) -> DistributionId:
    """Name a parameter's individual law independently of its current constructor."""
    return scientific_id("distribution", ["parameter", parameter_id])


def with_parameter_distributions(
    model: ModelSpec, laws: Mapping[ParameterId, NumPyroDistribution]
) -> ModelSpec:
    """Replace individual parameter laws and their memberships in one validated revision."""
    from nof1_causal_lab.numpyro_json import distribution_shape

    identities = {}
    for identity in laws:
        parameter = model.parameter(identity)
        identities[identity] = (
            parameter.distribution
            if parameter.distribution is not None
            and distribution_shape(model.distributions[parameter.distribution]) == ((), ())
            else parameter_distribution_id(identity)
        )
    parameters = tuple(
        type(parameter).model_validate(
            {**parameter.model_dump(), "distribution": identities[parameter.id]}
        )
        if parameter.id in laws
        else parameter
        for parameter in model.parameters
    )
    referenced = {item.distribution for item in (*parameters, *model.constructs)}
    distributions = {
        identity: law for identity, law in model.distributions.items() if identity in referenced
    }
    distributions.update((identities[identity], law) for identity, law in laws.items())
    return model.revised(parameters=parameters, distributions=distributions)


def dependency_id(key: DependencyKey, sources: tuple[ConstructId, ...]) -> str:
    """Preserve the semantic identity of a projected dependence across axis reorderings."""
    first, second, kind = key
    identity = "\0".join((kind, *sorted((first, second)), *sorted(sources)))
    digest = sha256(f"dependency\0{identity}".encode()).hexdigest()[:20]
    return f"dependency:{digest}"


class MarginalizedScale(TypedDict):
    parameter: str
    kind: str
    source_ids: list[ConstructId]
    sources: list[str]
    affected_states: list[str]
    directions: list[tuple[str, str]]
    dependency_ids: list[str]


def get_marginalized_scales(
    model: ModelSpec,
) -> list[MarginalizedScale]:
    """Return identifiable marginalized-confounder scale equivalence classes."""
    footprint_by_confounder: dict[ConstructId, set[str]] = defaultdict(set)
    kind_by_confounder: dict[ConstructId, str] = {}
    directions_by_confounder: dict[ConstructId, list[tuple[str, str]]] = defaultdict(list)
    dependency_ids_by_confounder: dict[ConstructId, list[str]] = defaultdict(list)

    for key, sources in model.induced_dependencies.items():
        first, second, kind = key
        between = model.get_construct(first).name, model.get_construct(second).name
        identity = dependency_id(key, sources)
        for source_id in sources:
            footprint_by_confounder[source_id].update(between)
            directions_by_confounder[source_id].append(between)
            dependency_ids_by_confounder[source_id].append(identity)
            kind_by_confounder[source_id] = kind

    members_by_footprint: dict[tuple[str, frozenset[str]], list[ConstructId]] = defaultdict(list)
    for source_id, footprint in footprint_by_confounder.items():
        members_by_footprint[(kind_by_confounder[source_id], frozenset(footprint))].append(
            source_id
        )

    scales: list[MarginalizedScale] = []
    for (kind, footprint), source_ids in sorted(
        members_by_footprint.items(),
        key=lambda item: (
            item[0][0],
            sorted(item[0][1]),
            sorted(item[1]),
        ),
    ):
        source_ids = sorted(source_ids)
        source_names = sorted(model.get_construct(source_id).name for source_id in source_ids)
        directions: set[tuple[str, str]] = set()
        dependency_ids: set[str] = set()
        for source_id in source_ids:
            directions.update(directions_by_confounder[source_id])
            dependency_ids.update(dependency_ids_by_confounder[source_id])
        scales.append(
            {
                "parameter": "tau_" + "__".join(source_names),
                "kind": kind,
                "source_ids": source_ids,
                "sources": source_names,
                "affected_states": sorted(footprint),
                "directions": sorted(directions),
                "dependency_ids": sorted(dependency_ids),
            }
        )
    return scales
