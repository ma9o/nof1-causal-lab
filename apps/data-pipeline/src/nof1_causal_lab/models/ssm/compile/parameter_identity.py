"""Declare scientific parameter identities at model construction and compilation."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from nof1_causal_lab.artifacts.identity import ConstructRef, EdgeRef, IndicatorRef
from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform, SiteKind
from nof1_causal_lab.artifacts.statistical_model_spec import ParameterRole, ParameterSpec

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.identity import EntityRef, ParameterElementId
    from nof1_causal_lab.artifacts.statistical_model_spec import LikelihoodSpec
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan
    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.structure.sites import SemanticBinding, SiteDescriptor


# Producer metadata includes the candidate's authored fields and explicit
# structural subjects before it becomes a validated ParameterSpec.
type ParameterMetadata = dict[str, Any]


SHARED_OBSERVATION_FAMILIES = {
    SiteKind.OBS_DF: "student_t",
    SiteKind.OBS_SHAPE: "gamma",
    SiteKind.OBS_R: "negative_binomial",
    SiteKind.OBS_CONCENTRATION: "beta",
    SiteKind.OBS_CAT_INTERCEPTS: "categorical",
    SiteKind.OBS_CAT_SLOPES: "categorical",
}


_ROLE_QUANTITIES = {
    ParameterRole.AR_COEFFICIENT: SiteKind.DYNAMICS_DECAY,
    ParameterRole.FIXED_EFFECT: SiteKind.DYNAMICS_WEIGHT,
    ParameterRole.RESIDUAL_SD: SiteKind.DIFFUSION_DIAG,
    ParameterRole.STATE_INTERCEPT: SiteKind.DYNAMICS_POTENTIAL_CENTER,
    ParameterRole.OBSERVATION_INTERCEPT: SiteKind.MANIFEST_MEANS,
    ParameterRole.INITIAL_STATE_MEAN: SiteKind.T0_MEANS,
    ParameterRole.INITIAL_STATE_SD: SiteKind.T0_VAR_DIAG,
    ParameterRole.STATIC_STATE_SD: SiteKind.STATIC_STATE_SD,
    ParameterRole.CORRELATION: SiteKind.DIFFUSION_LOWER,
    ParameterRole.INITIAL_STATE_CORRELATION: SiteKind.T0_VAR_LOWER,
    ParameterRole.LOADING: SiteKind.LOADING,
    ParameterRole.MEASUREMENT_ERROR_SD: SiteKind.MANIFEST_VAR_DIAG,
}


def scientific_id(prefix: str, payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{prefix}:{hashlib.sha256(encoded.encode()).hexdigest()}"


def parameter_identity(quantity: SiteKind, owners: list[EntityRef]) -> str:
    return scientific_id("parameter", [quantity.value, sorted(owner.id for owner in owners)])


def define_model_parameters(
    candidates: list[ParameterSpec], likelihoods: list[LikelihoodSpec], plan: StructuralPlan
) -> list[ParameterSpec]:
    """Declare the completed model's quantities after emission and dynamics choices.

    A shared likelihood parameter owns exactly the channels using that family in
    this model. Mechanism coefficients already declare their quantities and owners;
    emission-family membership is finalized once all likelihoods are known.
    """
    definitions = []
    for candidate in candidates:
        quantity = candidate.quantity
        owners: list[EntityRef] = candidate.owners
        if quantity in SHARED_OBSERVATION_FAMILIES:
            indicator_ids = sorted(
                likelihood.indicator_id
                for likelihood in likelihoods
                if likelihood.distribution.value == SHARED_OBSERVATION_FAMILIES[quantity]
            )
            construct_ids = sorted(
                {plan.semantics.indicators[key].construct_id for key in indicator_ids}
            )
            owners = [ConstructRef(id=key) for key in construct_ids]
            owners.extend(IndicatorRef(id=key) for key in indicator_ids)
            if not owners:
                raise ValueError(f"Shared parameter {candidate.name!r} has no active indicators")
        definitions.append(
            candidate.model_copy(
                update={
                    "id": parameter_identity(quantity, owners),
                    "quantity": quantity,
                    "owners": owners,
                }
            )
        )
    return definitions


def declare_parameter(candidate: ParameterMetadata, plan: StructuralPlan) -> ParameterMetadata:
    """Attach identities to a producer's explicit subjects, never to a parsed parameter label."""
    constructs = {item.name: item for item in plan.semantics.constructs.values()}
    indicators = {item.name: item for item in plan.semantics.indicators.values()}
    construct_names = set(candidate.get("construct_names", ()))
    construct_names.update(
        candidate[key]
        for key in ("construct", "construct_1", "construct_2", "cause", "effect")
        if candidate.get(key) is not None
    )
    indicator_names = set(candidate.get("indicator_names", ()))
    if candidate.get("indicator") is not None:
        indicator_names.add(candidate["indicator"])
    owners: list[EntityRef] = [
        ConstructRef(id=constructs[name].id) for name in sorted(construct_names)
    ]
    owners.extend(IndicatorRef(id=indicators[name].id) for name in sorted(indicator_names))
    if "cause" in candidate and "effect" in candidate:
        cause_id = constructs[candidate["cause"]].id
        effect_id = constructs[candidate["effect"]].id
        owners.extend(
            EdgeRef(id=edge.id)
            for edge in plan.semantics.edges.values()
            if edge.cause_id == cause_id and edge.effect_id == effect_id
        )
    role = ParameterRole(candidate["role"])
    quantity = (
        SiteKind(candidate["quantity"]) if "quantity" in candidate else _ROLE_QUANTITIES[role]
    )
    if role == ParameterRole.FIXED_EFFECT and constructs[candidate["cause"]].id in {
        item.construct_id for item in plan.known_inputs
    }:
        quantity = SiteKind.INPUT_EFFECT
    if not owners:
        raise ValueError(f"Parameter {candidate['name']!r} has no declared scientific owners")
    transforms = {
        ParameterRole.AR_COEFFICIENT: PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY,
        ParameterRole.FIXED_EFFECT: PriorAuthoringTransform.DT_EFFECT_TO_CT_RATE,
        ParameterRole.INITIAL_STATE_CORRELATION: PriorAuthoringTransform.INITIAL_STATE_CORRELATION,
    }
    return {
        **candidate,
        "id": parameter_identity(quantity, owners),
        "owners": [owner.model_dump(mode="json") for owner in owners],
        "quantity": quantity.value,
        "prior_transform": candidate.get(
            "prior_transform", transforms.get(role, PriorAuthoringTransform.IDENTITY).value
        ),
    }


def validate_parameter_owners(parameters: list[ParameterSpec], plan: StructuralPlan) -> None:
    catalogs = {
        "construct": plan.semantics.constructs,
        "indicator": plan.semantics.indicators,
        "edge": plan.semantics.edges,
    }
    for parameter in parameters:
        if not parameter.owners or any(
            owner.id not in catalogs[owner.kind] for owner in parameter.owners
        ):
            raise ValueError(f"Parameter {parameter.name!r} references an unknown scientific owner")
        if parameter.id != parameter_identity(parameter.quantity, parameter.owners):
            raise ValueError(
                f"Parameter {parameter.name!r} has an inconsistent scientific identity"
            )
    ids = [parameter.id for parameter in parameters]
    if len(ids) != len(set(ids)):
        raise ValueError("Scientific parameter definitions must have unique IDs")


def component_identity(
    parameter: ParameterSpec,
    indices: tuple[int, ...],
    binding: SemanticBinding,
    site: SiteDescriptor,
    spec: SSMSpec,
    plan: StructuralPlan | None,
) -> tuple[ParameterElementId, str] | None:
    """Identify category components by labels and covariance components by their ordered basis.

    Padded likelihood coordinates are execution-only and have no scientific component.
    """
    kind = site.site_kind
    key: object = "scalar"
    label = parameter.name
    if kind in {
        SiteKind.OBS_ORDERED_BASE,
        SiteKind.OBS_ORDERED_GAPS,
        SiteKind.OBS_CAT_INTERCEPTS,
        SiteKind.OBS_CAT_SLOPES,
    }:
        if plan is None:
            raise ValueError(
                "Categorical component identity requires its scientific indicator definition"
            )
        indicators = {item.name: item for item in plan.semantics.indicators.values()}
        assert spec.manifest_names is not None
        indicator = indicators[spec.manifest_names[indices[0]]]
        levels = (
            indicator.ordinal_levels
            if kind in {SiteKind.OBS_ORDERED_BASE, SiteKind.OBS_ORDERED_GAPS}
            else indicator.categorical_levels
        )
        if levels is None:
            return None
        if kind == SiteKind.OBS_ORDERED_BASE:
            key = [indicator.id, "first_cutpoint", levels[:2]]
            label = f"{indicator.name}: {levels[0]} | {levels[1]}"
        elif kind == SiteKind.OBS_ORDERED_GAPS:
            column = indices[1]
            if column >= len(levels) - 2:
                return None
            key = [indicator.id, "cutpoint_gap", levels[column : column + 3]]
            label = f"{indicator.name}: gap {levels[column]} / {levels[column + 1]} / {levels[column + 2]}"
        else:
            column = indices[1]
            if column >= len(levels) - 1:
                return None
            key = [indicator.id, "category_contrast", levels[column + 1], levels[0]]
            label = f"{indicator.name}: {levels[column + 1]} vs {levels[0]}"
    elif (
        kind
        in {
            SiteKind.DIFFUSION_LOWER,
            SiteKind.DIFFUSION_DIAG,
            SiteKind.T0_VAR_LOWER,
            SiteKind.T0_VAR_DIAG,
        }
        and plan is not None
    ):
        # A Cholesky entry is conditional on the preceding ordered basis. Reordering
        # that basis changes the quantity even if the endpoint labels survive.
        constructs = {item.name: item.id for item in plan.semantics.constructs.values()}
        position = site.positions[binding.flat_index]
        row = position[0] if isinstance(position, tuple) else position
        assert spec.latent_names is not None
        basis = [constructs[name] for name in spec.latent_names[: row + 1]]
        key = ["cholesky", basis]
    elif any(n > 1 for n in site.shape) and binding.transform.value == "site_wide":
        raise ValueError(f"Parameter {parameter.name!r} needs explicit scientific component axes")
    return scientific_id("element", [parameter.id, key]), label
