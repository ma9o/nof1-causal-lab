"""Author missing component slots before eliciting priors or compiling a model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    BinaryExpression,
    CoefficientExpression,
    LiteralExpression,
    StateExpression,
    coefficient,
)
from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.likelihoods import revise_law
from nof1_causal_lab.models.model_semantics import indicator_requires_observation_intercept

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def complete_component_slots(
    model: ModelSpec,
    *,
    free_initial: bool = False,
    free_observation_intercepts: bool = True,
) -> ModelSpec:
    """Apply authoring defaults once, leaving every selected coefficient in ModelSpec.

    Existing slots and parameter definitions are preserved. Numerical compilation
    never calls this transition or invents missing scientific parameters.
    """
    from nof1_causal_lab.models.model_distributions import with_parameter_distributions
    from nof1_causal_lab.models.ssm.priors import DEFAULT_PRIORS_BY_FIELD
    from nof1_causal_lab.utils.model_structure import get_marginalized_scales

    definitions = {parameter.id: parameter for parameter in model.parameters}
    laws = {}

    def parameter(owner: str, slot: str, name: str, *, prior=None):
        identity = scientific_id("parameter", [owner, slot])
        if identity not in definitions and prior is not None:
            laws[identity] = prior
        definitions.setdefault(
            identity,
            ParameterSpec(
                id=identity,
                name=name,
                description=f"{slot.replace('_', ' ')} for {name}",
            ),
        )
        return identity

    shared = {}
    process_df = next(
        (
            construct.coefficient("process_degrees_of_freedom")
            for construct in model.constructs
            if construct.coefficient("process_degrees_of_freedom") is not None
        ),
        None,
    )
    for _, likelihood in model.iter_likelihoods():
        for operand in likelihood.terms.auxiliary:
            if (
                operand.role not in {"observation_scale", "cutpoint_base", "cutpoint_gaps"}
                and operand.value is not None
            ):
                shared.setdefault((likelihood.law.family, operand.role), operand.value)

    states = set(model.state_order)
    manifests = set(model.manifest_indicator_order)
    constructs = {}
    for construct in model.constructs:
        updates = {}
        if construct.id in states:
            if (
                construct.temporal_status == "time_varying"
                and construct.coefficient("diffusion_scale") is None
            ):
                construct = construct.with_coefficients(
                    coefficient(
                        parameter(construct.id, "innovation.scale", f"sigma_{construct.name}"),
                        "diffusion_scale",
                    )
                )
            if (
                construct.innovation_family == DistributionFamily.STUDENT_T
                and construct.coefficient("process_degrees_of_freedom") is None
            ):
                if process_df is None:
                    process_df = parameter(
                        "innovation:student_t",
                        "degrees_of_freedom",
                        "proc_df",
                        prior=DEFAULT_PRIORS_BY_FIELD["proc_df"],
                    )
                construct = construct.with_coefficients(
                    coefficient(process_df, "process_degrees_of_freedom")
                )
            if (
                construct.coefficient("initial_mean") is None
                or construct.coefficient("initial_scale") is None
            ):
                static = construct.temporal_status == "time_invariant"
                anchored = any(
                    ind.likelihood is not None and ind.likelihood.standardized
                    for ind in construct.indicators
                )
                if construct.coefficient("initial_mean") is None:
                    initial_mean = (
                        parameter(
                            construct.id,
                            "initial_state.mean",
                            f"t0_mean_{construct.name}",
                            prior=DEFAULT_PRIORS_BY_FIELD["t0_means"],
                        )
                        if (free_initial and not static) or (static and anchored)
                        else 0
                    )
                    construct = construct.with_coefficients(
                        coefficient(initial_mean, "initial_mean")
                    )
                if construct.coefficient("initial_scale") is None:
                    initial_scale = (
                        parameter(
                            construct.id,
                            "initial_state.scale",
                            f"t0_sd_{construct.name}",
                            prior=DEFAULT_PRIORS_BY_FIELD["t0_var_diag"],
                        )
                        if free_initial or static
                        else 1
                    )
                    construct = construct.with_coefficients(
                        coefficient(initial_scale, "initial_scale")
                    )
        indicators = []
        for indicator in construct.indicators:
            likelihood = indicator.likelihood
            if indicator.id not in manifests or likelihood is None:
                indicators.append(indicator)
                continue
            terms = likelihood.terms
            family, link = terms.family, terms.link
            reference = model.reference_indicator_ids[construct.id] == indicator.id
            loadings = {}
            for identity, operand in terms.loadings.items():
                if operand.value is not None:
                    continue
                loadings[identity] = (
                    (
                        1
                        if family == DistributionFamily.CATEGORICAL
                        or indicator.construct_polarity == "positive"
                        else -1
                    )
                    if identity == construct.id
                    and (reference or family == DistributionFamily.CATEGORICAL)
                    else parameter(
                        indicator.id,
                        f"likelihood.loading.{identity}",
                        f"lambda_{indicator.name}_{model.get_construct(identity).name}",
                    )
                )
            replacements = {}
            if terms.intercept.value is None:
                replacements["observation_intercept"] = (
                    parameter(
                        indicator.id, "likelihood.intercept", f"manifest_mean_{indicator.name}"
                    )
                    if free_observation_intercepts
                    and indicator_requires_observation_intercept(
                        family, link, standardized=likelihood.standardized
                    )
                    else 0
                )
            for operand in terms.auxiliary:
                if operand.value is not None:
                    continue
                role = operand.role
                if role == "observation_scale":
                    replacements[role] = (
                        parameter(indicator.id, "likelihood.scale", f"obs_sd_{indicator.name}")
                        if len(construct.indicators) > 1
                        else 0
                    )
                elif role == "cutpoint_gaps" and len(indicator.ordinal_levels or ()) <= 2:
                    replacements[role] = None
                elif role in {"cutpoint_base", "cutpoint_gaps"}:
                    native_name = operand.meaning.quantity.value
                    replacements[role] = parameter(
                        indicator.id,
                        f"likelihood.{role}",
                        f"{native_name}_{indicator.name}",
                        prior=DEFAULT_PRIORS_BY_FIELD[native_name],
                    )
                else:
                    key = (family, role)
                    if key not in shared:
                        native_name = operand.meaning.quantity.value
                        shared[key] = parameter(
                            f"likelihood:{family.value}",
                            role,
                            native_name,
                            prior=DEFAULT_PRIORS_BY_FIELD[native_name],
                        )
                    replacements[role] = shared[key]

            def complete_operand(node, *, replacements=replacements, loadings=loadings):
                if isinstance(node, CoefficientExpression) and node.role in replacements:
                    value = replacements[node.role]
                    return (
                        LiteralExpression(value=0)
                        if value is None
                        else node.model_copy(update={"value": value})
                    )
                if isinstance(node, BinaryExpression) and node.operator == "multiply":
                    for coeff, source, reverse in (
                        (node.left, node.right, False),
                        (node.right, node.left, True),
                    ):
                        if (
                            isinstance(coeff, CoefficientExpression)
                            and coeff.role == "loading"
                            and isinstance(source, StateExpression)
                            and source.construct_id in loadings
                        ):
                            revised = coeff.model_copy(
                                update={"value": loadings[source.construct_id]}
                            )
                            return node.model_copy(update={"right" if reverse else "left": revised})
                return node

            indicators.append(
                indicator.model_copy(
                    update={"likelihood": revise_law(likelihood, complete_operand)}
                )
            )
        updates["indicators"] = tuple(indicators)
        constructs[construct.id] = construct.model_copy(update=updates)

    axis = {identity: index for index, identity in enumerate(model.state_order)}
    for first, second, kind in model.induced_dependencies:
        if kind != "innovation_correlation":
            continue
        first, second = sorted((first, second), key=axis.__getitem__)
        owner = constructs[second]
        if owner.coefficient("diffusion_scale") is None:
            continue
        # Either endpoint may own the declared pair; never duplicate a supplied coefficient.
        if (
            owner.coefficient("diffusion_loading", construct_ids=(first,)) is not None
            or constructs[first].coefficient("diffusion_loading", construct_ids=(second,))
            is not None
        ):
            continue
        name = f"cor_{constructs[first].name}_{owner.name}"
        constructs[second] = owner.with_coefficients(
            coefficient(
                parameter(second, f"innovation.loading.{first}", name),
                "diffusion_loading",
                construct_ids=(first,),
            )
        )

    for scale in get_marginalized_scales(model):
        if scale["kind"] != "initial_state_correlation":
            continue
        sources = [item for item in constructs.values() if item.name in scale["sources"]]
        affected = {
            item.id for item in constructs.values() if item.name in scale["affected_states"]
        }
        if any(
            item.id in affected
            and any(
                operand.role == "initial_correlation" and set(operand.construct_ids) <= affected
                for operand in item.coefficients
            )
            for item in constructs.values()
        ):
            continue
        existing = [
            item.coefficient("initial_scale")
            for item in sources
            if item.coefficient("initial_scale") is not None
        ]
        scale_coefficient = (
            existing[0]
            if existing
            else parameter(
                scientific_id("baseline", sorted(item.id for item in sources)),
                "initial_state.scale",
                scale["parameter"],
            )
        )
        for item in sources:
            if item.coefficient("initial_scale") is None:
                initial_mean = item.coefficient("initial_mean")
                constructs[item.id] = item.with_coefficients(
                    coefficient(
                        0 if initial_mean is None else initial_mean,
                        "initial_mean",
                    ),
                    coefficient(scale_coefficient, "initial_scale"),
                )
    candidate = model.revised(
        edges=replace_constructs(model.edges, tuple(constructs.values())),
        parameters=tuple(definitions.values()),
    )
    return with_parameter_distributions(candidate, laws)
