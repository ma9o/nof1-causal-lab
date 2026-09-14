"""Author scientific components first; their slots declare the parameters to elicit."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.coefficient import FixedCoefficient, ParameterCoefficient
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.expressions import (
    hill as expr_hill,
)
from nof1_causal_lab.artifacts.expressions import (
    linear_effect,
    restoring_potential,
)
from nof1_causal_lab.artifacts.expressions import (
    state as expr_state,
)
from nof1_causal_lab.artifacts.identity import scientific_id
from nof1_causal_lab.artifacts.mechanism import DynamicsMechanism
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform
from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec

if TYPE_CHECKING:
    from collections.abc import Collection

    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def default_mechanism_id(owner_id: str, kind: str) -> str:
    """Allocate one default term; additional additive terms choose distinct persistent IDs."""
    return scientific_id("mechanism", [owner_id, kind])


def declare_dynamics(
    model: ModelSpec,
    *,
    self_limiting: Collection[str] = (),
    hill_edges: Collection[str] = (),
    centered_states: Collection[str] = (),
) -> ModelSpec:
    """Add missing dynamics using explicit choices, preserving existing components."""
    parameters = {parameter.id: parameter for parameter in model.parameters}

    def coefficient(owner_id, slot, name, transform=PriorAuthoringTransform.IDENTITY):
        identity = scientific_id("parameter", [owner_id, slot])
        parameters.setdefault(
            identity,
            ParameterSpec(
                id=identity,
                name=name,
                description=f"{slot} of {name}",
                distribution_transform=transform,
            ),
        )
        return ParameterCoefficient(parameter_id=identity)

    states = set(model.state_order)
    edge_ids = {edge.id for edge in model.execution_edges}
    if (
        not set(self_limiting) <= states
        or not set(centered_states) <= states
        or not set(hill_edges) <= edge_ids
    ):
        raise ValueError("Dynamics choices must reference retained constructs and edges")
    constructs = []
    for construct in model.constructs:
        if (
            construct.id not in states
            or construct.temporal_status == "time_invariant"
            or construct.dynamics
        ):
            constructs.append(construct)
            continue
        term = default_mechanism_id(construct.id, "node_potential")
        mechanism = DynamicsMechanism(
            id=term,
            kind="potential",
            expression=restoring_potential(
                construct.id,
                center=coefficient(term, "center", f"cint_{construct.name}")
                if construct.id in centered_states
                else FixedCoefficient(value=0),
                stiffness=coefficient(
                    term,
                    "stiffness",
                    f"rho_{construct.name}",
                    PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY,
                ),
                quartic=coefficient(term, "quartic", f"self_limit_{construct.name}")
                if construct.id in self_limiting
                else FixedCoefficient(value=0),
            ),
        )
        constructs.append(construct.model_copy(update={"dynamics": (mechanism,)}))
    inputs = set(model.known_inputs)
    edges = []
    for edge in model.edges:
        if edge.id not in edge_ids or edge.mechanisms:
            edges.append(edge)
            continue
        cause, effect = edge.cause, edge.effect
        if edge.id in hill_edges:
            if edge.cause.id in inputs:
                raise ValueError("Known inputs currently support linear mechanisms")
            term = default_mechanism_id(edge.id, "hill")
            mechanism = DynamicsMechanism(
                id=term,
                expression=expr_hill(
                    expr_state(edge.cause.id),
                    **{
                        slot: coefficient(term, slot, f"hill_{slot}_{cause.name}_{effect.name}")
                        for slot in ("emax", "ec50", "n")
                    },
                ),
            )
        else:
            term = default_mechanism_id(edge.id, "linear")
            mechanism = DynamicsMechanism(
                id=term,
                expression=linear_effect(
                    edge.cause.id,
                    coefficient(
                        term,
                        "weight",
                        f"beta_{cause.name}_{effect.name}",
                        PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE,
                    ),
                ),
            )
        edges.append(edge.model_copy(update={"mechanisms": (mechanism,)}))
    return model.revised(
        edges=replace_constructs(tuple(edges), tuple(constructs)),
        parameters=tuple(parameters.values()),
    )


def default_model(model: ModelSpec, **choices) -> ModelSpec:
    """Produce a concrete editable proposal; uncertainty remains on unresolved parameters."""
    from nof1_causal_lab.artifacts.likelihood import VALID_LINKS_FOR_DISTRIBUTION, LikelihoodSpec
    from nof1_causal_lab.distributions import VALID_LIKELIHOODS_FOR_DTYPE
    from nof1_causal_lab.models.model_semantics import should_auto_standardize_indicator
    from nof1_causal_lab.models.parameter_planning import complete_component_slots
    from nof1_causal_lab.utils.observation_semantics import get_observation_semantics

    model = declare_dynamics(model, **choices)
    manifests = set(model.manifest_indicator_order)
    constructs = []
    for construct in model.constructs:
        indicators = []
        for indicator in construct.indicators:
            if indicator.id not in manifests or indicator.likelihood is not None:
                indicators.append(indicator)
                continue
            allowed = VALID_LIKELIHOODS_FOR_DTYPE[indicator.measurement_dtype]
            family = allowed[0]
            link = sorted(VALID_LINKS_FOR_DISTRIBUTION[family], key=lambda value: value.value)[0]
            semantics = get_observation_semantics(indicator.model_dump(mode="json"))
            from nof1_causal_lab.models.likelihoods import observation_law

            likelihood = LikelihoodSpec(
                law=observation_law(construct.id, family, link),
                standardized=should_auto_standardize_indicator(
                    family, link, semantics.support_kind.value, semantics.summary_operator.value
                ),
                reasoning="Editable authoring default.",
            )
            indicators.append(indicator.model_copy(update={"likelihood": likelihood}))
        constructs.append(construct.model_copy(update={"indicators": tuple(indicators)}))
    model = model.revised(edges=replace_constructs(model.edges, tuple(constructs)))
    return complete_component_slots(model)
