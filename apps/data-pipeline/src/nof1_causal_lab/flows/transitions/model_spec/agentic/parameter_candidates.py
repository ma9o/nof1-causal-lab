"""Parameter prompt rows projected from the slots of a concrete model proposal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nof1_causal_lab.artifacts.parameter import PriorAuthoringTransform, SiteKind
from nof1_causal_lab.artifacts.parameter_spec import ParameterConstraint, ParameterRole
from nof1_causal_lab.models.model_distributions import parameter_distribution_id

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


# Display rows describe actual references; they are not a second scientific schema.
type ParameterMetadata = dict[str, Any]


_ROLES = {
    SiteKind.DYNAMICS_DECAY: (
        ParameterRole.DYNAMICS_PARAMETER_POSITIVE,
        ParameterConstraint.POSITIVE,
    ),
    SiteKind.DYNAMICS_WEIGHT: (ParameterRole.FIXED_EFFECT, ParameterConstraint.NONE),
    SiteKind.DYNAMICS_CINT: (ParameterRole.STATE_INTERCEPT, ParameterConstraint.NONE),
    SiteKind.DYNAMICS_POTENTIAL_CENTER: (ParameterRole.STATE_INTERCEPT, ParameterConstraint.NONE),
    SiteKind.DYNAMICS_POTENTIAL_QUARTIC: (
        ParameterRole.DYNAMICS_PARAMETER_POSITIVE,
        ParameterConstraint.POSITIVE,
    ),
    SiteKind.HILL_EMAX: (ParameterRole.DYNAMICS_PARAMETER_POSITIVE, ParameterConstraint.POSITIVE),
    SiteKind.HILL_EC50: (ParameterRole.DYNAMICS_PARAMETER_POSITIVE, ParameterConstraint.POSITIVE),
    SiteKind.HILL_N: (ParameterRole.DYNAMICS_PARAMETER_POSITIVE, ParameterConstraint.POSITIVE),
    SiteKind.DIFFUSION_DIAG: (ParameterRole.RESIDUAL_SD, ParameterConstraint.POSITIVE),
    SiteKind.DIFFUSION_LOWER: (ParameterRole.CORRELATION, ParameterConstraint.NONE),
    SiteKind.LOADING: (ParameterRole.LOADING, ParameterConstraint.NONE),
    SiteKind.MANIFEST_MEANS: (ParameterRole.OBSERVATION_INTERCEPT, ParameterConstraint.NONE),
    SiteKind.MANIFEST_VAR_DIAG: (ParameterRole.MEASUREMENT_ERROR_SD, ParameterConstraint.POSITIVE),
    SiteKind.T0_MEANS: (ParameterRole.INITIAL_STATE_MEAN, ParameterConstraint.NONE),
    SiteKind.T0_VAR_DIAG: (ParameterRole.INITIAL_STATE_SD, ParameterConstraint.POSITIVE),
    SiteKind.T0_VAR_LOWER: (
        ParameterRole.INITIAL_STATE_CORRELATION,
        ParameterConstraint.CORRELATION,
    ),
    SiteKind.STATIC_STATE_SD: (ParameterRole.STATIC_STATE_SD, ParameterConstraint.POSITIVE),
}


def describe_parameters(model: ModelSpec) -> list[ParameterMetadata]:
    """Display metadata is derived from actual parameter uses, never authored separately."""
    rows = []
    for parameter in model.parameters:
        if parameter.value is not None:
            continue
        context = model.parameter_context(parameter.id)
        kind = context.quantity
        positive = kind in {
            SiteKind.OBS_DF,
            SiteKind.OBS_SHAPE,
            SiteKind.OBS_R,
            SiteKind.OBS_CONCENTRATION,
            SiteKind.OBS_ORDERED_GAPS,
            SiteKind.PROC_DF,
        }
        role, constraint = _ROLES.get(
            kind,
            (ParameterRole.OBSERVATION_HYPERPARAMETER_POSITIVE, ParameterConstraint.POSITIVE)
            if positive
            else (ParameterRole.OBSERVATION_HYPERPARAMETER, ParameterConstraint.NONE),
        )
        if parameter.distribution_transform == PriorAuthoringTransform.DT_PERSISTENCE_TO_CT_DECAY:
            role, constraint = ParameterRole.AR_COEFFICIENT, ParameterConstraint.UNIT_INTERVAL
        constructs = [
            model.get_construct(owner.id) for owner in context.owners if owner.kind == "construct"
        ]
        indicators = [
            model.indicator(owner.id) for owner in context.owners if owner.kind == "indicator"
        ]
        edges = [model.edge(owner.id) for owner in context.owners if owner.kind == "edge"]
        row = {
            **parameter.model_dump(mode="json"),
            "distribution": parameter.distribution or parameter_distribution_id(parameter.id),
            "role": role,
            "constraint": constraint,
            "slots": [use.slot for use in context.uses],
            "construct_names": [item.name for item in constructs],
            "indicator_names": [item.name for item in indicators],
        }
        if len(constructs) == 1:
            row["construct"] = constructs[0].name
        if len(indicators) == 1:
            row["indicator"] = indicators[0].name
            if kind == SiteKind.LOADING:
                row["constraint"] = indicators[0].construct_polarity.value
        if len(edges) == 1:
            row.update(
                cause=edges[0].cause.name,
                effect=edges[0].effect.name,
                lagged=edges[0].lagged,
            )
        rows.append(row)
    return rows
