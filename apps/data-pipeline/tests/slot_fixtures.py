"""Map numerical expectations in test recipes to explicit scientific component slots."""

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import COEFFICIENT_MEANINGS, CoefficientExpression, state
from nof1_causal_lab.artifacts.expressions import coefficient as expression_coefficient
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.artifacts.state_distribution import (
    InitialStateSpec,
    InnovationSpec,
    StateCoupling,
)
from nof1_causal_lab.models.likelihoods import revise_law


def with_likelihood_coefficients(likelihood, values):
    """Fill explicitly named expression operands in test declarations."""
    return revise_law(
        likelihood,
        lambda node: (
            node.model_copy(update={"coefficient": values[node.role]})
            if isinstance(node, CoefficientExpression) and node.role in values
            else node
        ),
    )


def fixture_parameter_id(quantity, owners):
    """Stable IDs for numerical test recipes; no runtime ownership metadata is authored."""
    from nof1_causal_lab.artifacts.identity import scientific_id

    return scientific_id("parameter", [quantity.value, sorted(owner.id for owner in owners)])


def attach_test_coefficients(model, recipes, *, parameters=()):
    constructs = {item.id: item for item in model.constructs}
    indicator_owners = {indicator.id: owner.id for owner, indicator in model.iter_indicators()}
    likelihood_roles = {
        meaning.quantity: role
        for role, meaning in COEFFICIENT_MEANINGS.items()
        if role
        not in {"center", "decay", "quartic", "intercept", "weight", "emax", "ec50", "exponent"}
    }
    for kind, owners, coefficient in recipes:
        construct_ids = [owner.id for owner in owners if owner.kind == "construct"]
        indicator_ids = [owner.id for owner in owners if owner.kind == "indicator"]
        if kind in likelihood_roles:
            for identity in indicator_ids:
                owner = constructs[indicator_owners[identity]]
                indicators = []
                for ind in owner.indicators:
                    if ind.id == identity:
                        likelihood = ind.likelihood
                        if kind == SiteKind.LOADING:
                            source = construct_ids[0]
                            new_term = expression_coefficient(coefficient, "loading") * state(
                                source
                            )
                            if source in likelihood.terms.loadings:
                                original = likelihood.terms.loadings[source] * state(source)
                                replacement = new_term
                            else:
                                original = likelihood.terms.predictor
                                replacement = original + new_term
                            likelihood = revise_law(
                                likelihood,
                                lambda node, original=original, replacement=replacement: (
                                    replacement if node == original else node
                                ),
                            )
                        else:
                            likelihood = with_likelihood_coefficients(
                                likelihood, {likelihood_roles[kind]: coefficient}
                            )
                        ind = ind.model_copy(update={"likelihood": likelihood})
                    indicators.append(ind)
                constructs[owner.id] = owner.model_copy(update={"indicators": tuple(indicators)})
        elif kind in {SiteKind.DIFFUSION_DIAG, SiteKind.DIFFUSION_LOWER, SiteKind.PROC_DF}:
            construct_ids.sort(key=list(constructs).index)
            owner = constructs[construct_ids[-1]]
            noise = owner.innovation or InnovationSpec(scale=FixedCoefficient(value=0))
            if kind == SiteKind.DIFFUSION_DIAG:
                noise = noise.model_copy(update={"scale": coefficient})
            elif kind == SiteKind.PROC_DF:
                noise = noise.model_copy(update={"degrees_of_freedom": coefficient})
            else:
                noise = noise.model_copy(
                    update={
                        "loadings": (
                            *noise.loadings,
                            StateCoupling(other_id=construct_ids[0], coefficient=coefficient),
                        )
                    }
                )
            constructs[owner.id] = owner.model_copy(update={"innovation": noise})
        elif kind in {
            SiteKind.T0_MEANS,
            SiteKind.T0_VAR_DIAG,
            SiteKind.T0_VAR_LOWER,
            SiteKind.STATIC_STATE_SD,
        }:
            construct_ids.sort(key=list(constructs).index)
            targets = construct_ids if kind == SiteKind.STATIC_STATE_SD else [construct_ids[-1]]
            for identity in targets:
                owner = constructs[identity]
                initial = owner.initial_state or InitialStateSpec(
                    mean=FixedCoefficient(value=0), scale=FixedCoefficient(value=1)
                )
                if kind == SiteKind.T0_MEANS:
                    initial = initial.model_copy(update={"mean": coefficient})
                elif kind == SiteKind.T0_VAR_LOWER:
                    initial = initial.model_copy(
                        update={
                            "correlations": (
                                *initial.correlations,
                                StateCoupling(other_id=construct_ids[0], coefficient=coefficient),
                            )
                        }
                    )
                else:
                    initial = initial.model_copy(update={"scale": coefficient})
                constructs[identity] = owner.model_copy(update={"initial_state": initial})
    return model.revised(
        edges=replace_constructs(model.edges, tuple(constructs.values())),
        parameters=(*model.parameters, *parameters),
    )


def flat_catalogue_payload(model):
    """Construct the retired schema explicitly for offline migration tests."""
    from nof1_causal_lab.artifacts.construct import CausalEdge
    from nof1_causal_lab.artifacts.expressions import linear_coefficient, restoring_coefficients

    value = model.model_dump(mode="json")
    value["constructs"] = [construct.model_dump(mode="json") for construct in model.constructs]
    value["edges"] = [
        {
            **edge.model_dump(mode="json", exclude={"cause", "effect"}),
            "cause_id": edge.cause.id,
            "effect_id": edge.effect.id,
        }
        for edge in model.edges
    ]

    def encoded_coefficient(reference):
        assert reference is not None
        return reference.model_dump(mode="json")

    retired_terms = {}
    for owner, mechanism in model.iter_mechanisms():
        if isinstance(owner, CausalEdge):
            retired_terms[mechanism.id] = {
                "id": mechanism.id,
                "kind": "linear",
                "weight": linear_coefficient(mechanism.expression, owner.cause.id).model_dump(
                    mode="json"
                ),
            }
        else:
            operands = restoring_coefficients(mechanism.expression, owner.id)
            assert {operand.role for operand in operands} == {"center", "decay", "quartic"}
            retired_terms[mechanism.id] = {
                "id": mechanism.id,
                "kind": "node_potential",
                **{
                    "stiffness" if operand.role == "decay" else operand.role: encoded_coefficient(
                        operand.coefficient
                    )
                    for operand in operands
                },
            }
    for field, terms in (("constructs", "dynamics"), ("edges", "mechanisms")):
        for owner in value[field]:
            owner[terms] = [retired_terms[term["id"]] for term in owner[terms]]
    value["policies"] = {"initialization": "stationary", "observation_intercept": "free"}
    for parameter in value["parameters"]:
        context = model.parameter_context(parameter["id"])
        parameter["quantity"] = context.quantity.value
        parameter["owners"] = [owner.model_dump(mode="json") for owner in context.owners]
    for construct in value["constructs"]:
        noise = construct.pop("innovation")
        construct["innovation_family"] = noise["distribution"] if noise else "gaussian"
        construct.pop("initial_state")
        for indicator in construct["indicators"]:
            if indicator["likelihood"] is not None:
                likelihood = model.indicator(indicator["id"]).likelihood
                indicator["likelihood"] = {
                    "distribution": likelihood.law.family.value,
                    "link": likelihood.terms.link.value,
                    "standardized": likelihood.standardized,
                    "reasoning": likelihood.reasoning,
                    "sources": indicator["likelihood"]["sources"],
                }

    def retire(node):
        if isinstance(node, dict):
            if node.get("kind") == "parameter":
                node["kind"] = "estimated"
            for child in node.values():
                retire(child)
        elif isinstance(node, list):
            for child in node:
                retire(child)

    retire(value)
    return value
