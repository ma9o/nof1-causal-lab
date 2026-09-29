"""Map numerical expectations in test recipes to explicit scientific component slots."""

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    COEFFICIENT_MEANINGS,
    CONSTRUCT_COEFFICIENT_ROLES,
    CoefficientExpression,
    CoefficientRole,
    state,
)
from nof1_causal_lab.artifacts.expressions import coefficient as expression_coefficient
from nof1_causal_lab.artifacts.parameter import SiteKind
from nof1_causal_lab.models.likelihoods import revise_law


def with_likelihood_coefficients(likelihood, values):
    """Fill explicitly named expression operands in test declarations."""
    return revise_law(
        likelihood,
        lambda node: (
            node.model_copy(update={"value": values[node.role]})
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
        | CONSTRUCT_COEFFICIENT_ROLES
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
            if owner.coefficient("diffusion_scale") is None:
                owner = owner.with_coefficients(expression_coefficient(0, "diffusion_scale"))
            diffusion_roles: dict[SiteKind, CoefficientRole] = {
                SiteKind.DIFFUSION_DIAG: "diffusion_scale",
                SiteKind.DIFFUSION_LOWER: "diffusion_loading",
                SiteKind.PROC_DF: "process_degrees_of_freedom",
            }
            role = diffusion_roles[kind]
            if kind == SiteKind.PROC_DF:
                owner = owner.model_copy(update={"innovation_family": "student_t"})
            constructs[owner.id] = owner.with_coefficients(
                expression_coefficient(
                    coefficient,
                    role,
                    construct_ids=(construct_ids[0],) if kind == SiteKind.DIFFUSION_LOWER else (),
                )
            )
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
                for role, value in (("initial_mean", 0), ("initial_scale", 1)):
                    if owner.coefficient(role) is None:
                        owner = owner.with_coefficients(expression_coefficient(value, role))
                role = (
                    "initial_mean"
                    if kind == SiteKind.T0_MEANS
                    else "initial_correlation"
                    if kind == SiteKind.T0_VAR_LOWER
                    else "initial_scale"
                )
                constructs[identity] = owner.with_coefficients(
                    expression_coefficient(
                        coefficient,
                        role,
                        construct_ids=(construct_ids[0],) if kind == SiteKind.T0_VAR_LOWER else (),
                    )
                )
    return model.revised(
        edges=replace_constructs(model.edges, tuple(constructs.values())),
        parameters=(*model.parameters, *parameters),
    )
