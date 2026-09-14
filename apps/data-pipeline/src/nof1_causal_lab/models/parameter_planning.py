"""Author missing component slots before eliciting priors or compiling a model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient, ParameterCoefficient
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    BinaryExpression,
    CoefficientExpression,
    LiteralExpression,
    StateExpression,
)
from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
from nof1_causal_lab.artifacts.state_distribution import (
    InitialStateSpec,
    InnovationSpec,
    StateCoupling,
)
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
    from nof1_causal_lab.models.ssm.priors import DEFAULT_PRIORS_BY_FIELD
    from nof1_causal_lab.utils.model_structure import get_marginalized_scales

    definitions = {parameter.id: parameter for parameter in model.parameters}

    def parameter(owner: str, slot: str, name: str, *, prior=None):
        identity = scientific_id("parameter", [owner, slot])
        definitions.setdefault(
            identity,
            ParameterSpec(
                id=identity,
                name=name,
                description=f"{slot.replace('_', ' ')} for {name}",
                distribution=prior,
            ),
        )
        return ParameterCoefficient(parameter_id=identity)

    shared = {}
    process_df = next(
        (
            construct.innovation.degrees_of_freedom
            for construct in model.constructs
            if construct.innovation is not None
            and construct.innovation.degrees_of_freedom is not None
        ),
        None,
    )
    for _, likelihood in model.iter_likelihoods():
        for operand in likelihood.terms.auxiliary:
            if (
                operand.role not in {"observation_scale", "cutpoint_base", "cutpoint_gaps"}
                and operand.coefficient is not None
            ):
                shared.setdefault((likelihood.law.family, operand.role), operand.coefficient)

    states = set(model.state_order)
    manifests = set(model.manifest_indicator_order)
    constructs = {}
    for construct in model.constructs:
        updates = {}
        if construct.id in states:
            if construct.temporal_status == "time_varying" and construct.innovation is None:
                updates["innovation"] = InnovationSpec(
                    scale=parameter(construct.id, "innovation.scale", f"sigma_{construct.name}")
                )
            noise = updates.get("innovation", construct.innovation)
            if (
                noise is not None
                and noise.distribution == DistributionFamily.STUDENT_T
                and noise.degrees_of_freedom is None
            ):
                if process_df is None:
                    process_df = parameter(
                        "innovation:student_t",
                        "degrees_of_freedom",
                        "proc_df",
                        prior=DEFAULT_PRIORS_BY_FIELD["proc_df"],
                    )
                updates["innovation"] = noise.model_copy(update={"degrees_of_freedom": process_df})
            if construct.initial_state is None:
                static = construct.temporal_status == "time_invariant"
                anchored = any(
                    ind.likelihood is not None and ind.likelihood.standardized
                    for ind in construct.indicators
                )
                updates["initial_state"] = InitialStateSpec(
                    mean=parameter(
                        construct.id,
                        "initial_state.mean",
                        f"t0_mean_{construct.name}",
                        prior=DEFAULT_PRIORS_BY_FIELD["t0_means"],
                    )
                    if (free_initial and not static) or (static and anchored)
                    else FixedCoefficient(value=0),
                    scale=parameter(
                        construct.id,
                        "initial_state.scale",
                        f"t0_sd_{construct.name}",
                        prior=DEFAULT_PRIORS_BY_FIELD["t0_var_diag"],
                    )
                    if free_initial or static
                    else FixedCoefficient(value=1),
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
                if operand.coefficient is not None:
                    continue
                loadings[identity] = (
                    FixedCoefficient(
                        value=1
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
            if terms.intercept.coefficient is None:
                replacements["observation_intercept"] = (
                    parameter(
                        indicator.id, "likelihood.intercept", f"manifest_mean_{indicator.name}"
                    )
                    if free_observation_intercepts
                    and indicator_requires_observation_intercept(
                        family, link, standardized=likelihood.standardized
                    )
                    else FixedCoefficient(value=0)
                )
            for operand in terms.auxiliary:
                if operand.coefficient is not None:
                    continue
                role = operand.role
                if role == "observation_scale":
                    replacements[role] = (
                        parameter(indicator.id, "likelihood.scale", f"obs_sd_{indicator.name}")
                        if len(construct.indicators) > 1
                        else FixedCoefficient(value=0)
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
                        else node.model_copy(update={"coefficient": value})
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
                                update={"coefficient": loadings[source.construct_id]}
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
        noise = owner.innovation
        if noise is None:
            continue
        # Either endpoint may own the declared pair; never duplicate a supplied coefficient.
        other_noise = constructs[first].innovation
        if any(item.other_id == first for item in noise.loadings) or (
            other_noise is not None
            and any(item.other_id == second for item in other_noise.loadings)
        ):
            continue
        name = f"cor_{constructs[first].name}_{owner.name}"
        loading = StateCoupling(
            other_id=first, coefficient=parameter(second, f"innovation.loading.{first}", name)
        )
        constructs[second] = owner.model_copy(
            update={"innovation": noise.model_copy(update={"loadings": (*noise.loadings, loading)})}
        )

    for scale in get_marginalized_scales(model):
        if scale["kind"] != "initial_state_correlation":
            continue
        sources = [item for item in constructs.values() if item.name in scale["sources"]]
        affected = {
            item.id for item in constructs.values() if item.name in scale["affected_states"]
        }
        if any(
            item.initial_state is not None
            and item.id in affected
            and any(
                correlation.other_id in affected for correlation in item.initial_state.correlations
            )
            for item in constructs.values()
        ):
            continue
        existing = [item.initial_state.scale for item in sources if item.initial_state is not None]
        coefficient = (
            existing[0]
            if existing
            else parameter(
                scientific_id("baseline", sorted(item.id for item in sources)),
                "initial_state.scale",
                scale["parameter"],
            )
        )
        for item in sources:
            if item.initial_state is None:
                constructs[item.id] = item.model_copy(
                    update={
                        "initial_state": InitialStateSpec(
                            mean=FixedCoefficient(value=0), scale=coefficient
                        )
                    }
                )
    return model.revised(
        edges=replace_constructs(model.edges, tuple(constructs.values())),
        parameters=tuple(definitions.values()),
    )
