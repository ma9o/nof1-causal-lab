"""Check question admission and execution requirements on the scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.model_structure import (
    StructuralSelection,
    StructuralSelectionError,
    reference_indicators,
    selected_state_ids,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.artifacts.question import QuestionSpec


def question_edit_reason(
    dynamical_model_spec: DynamicalModelSpec, question: QuestionSpec
) -> str | None:
    """Reject edits that cannot answer the question or violate scoped parameter anchors."""
    from nof1_causal_lab.artifacts.construct import Role
    from nof1_causal_lab.utils.causal_design import get_all_treatments

    if missing := sorted(
        identity
        for identity in {question.outcome, *question.targets}
        if identity not in dynamical_model_spec._constructs
    ):
        return "The model must define the question's constructs: " + ", ".join(missing)
    outcome_construct = dynamical_model_spec.get_construct(question.outcome)
    if outcome_construct.role != Role.ENDOGENOUS:
        return "The question's outcome must reference an endogenous construct"
    ancestors = frozenset(
        get_all_treatments(
            dynamical_model_spec.constructs, dynamical_model_spec.edges, question.outcome
        )
    )
    outside = sorted(
        f"{construct.name!r} ({construct.id})"
        for construct in dynamical_model_spec.constructs
        if construct.id != question.outcome and construct.name not in ancestors
    )
    if outside:
        return (
            "Every construct must have a directed path to the question's outcome "
            f"{outcome_construct.name!r} ({outcome_construct.id}). "
            "No directed path: " + ", ".join(outside)
        )
    try:
        StructuralSelection(dynamical_model_spec, question.outcome)
    except StructuralSelectionError as exc:
        return str(exc)
    return None


def validate_parameter_anchors(selection: StructuralSelection) -> None:
    """Reject unanchored retained constructs once their anchor inputs are authored."""
    from nof1_causal_lab.artifacts.construct import CausalEdgeSpec
    from nof1_causal_lab.artifacts.expressions import StateExpression, restoring_coefficients
    from nof1_causal_lab.distributions import DistributionFamily

    dynamical_model_spec = selection.dynamical_model_spec
    if dynamical_model_spec.measurement_clock is None or not dynamical_model_spec.indicators:
        return

    for identity in selected_state_ids(selection):
        construct = dynamical_model_spec.get_construct(identity)
        initial_mean = construct.coefficient("initial_mean")
        channels = [
            (indicator, indicator.likelihood)
            for indicator in construct.indicators
            if indicator.likelihood is not None
        ]
        if initial_mean is None or len(channels) != len(construct.indicators):
            continue
        if any(
            operand.value is None
            for _, likelihood in channels
            for operand in likelihood.parsed.loadings.values()
        ):
            continue
        centers = [
            operand.value
            for owner, mechanism in dynamical_model_spec.iter_mechanisms()
            if (owner.effect.id if isinstance(owner, CausalEdgeSpec) else owner.id) == identity
            for operand in restoring_coefficients(
                mechanism.expression, identity, kind=mechanism.kind
            )
            if operand.role == "center"
        ]
        if any(center is None for center in centers):
            continue

        channel_location = any(
            likelihood.standardized
            or (
                likelihood.law.distribution == "Delta"
                and isinstance(likelihood.law.v, StateExpression)
            )
            for indicator, likelihood in channels
        )
        fixed_location = (
            not isinstance(initial_mean, str)
            if construct.temporal_status == "time_invariant"
            else any(not isinstance(center, str) for center in centers)
        )
        if not channel_location and not fixed_location:
            raise StructuralSelectionError(f"Construct {construct.name!r} has no location anchor")

        fixed_scale = False
        all_categorical = True
        for indicator, likelihood in channels:
            categorical = likelihood.law.family == DistributionFamily.CATEGORICAL
            loading = likelihood.parsed.loadings[identity].value
            value = None if isinstance(loading, str) else loading
            if categorical and value is None:
                raise StructuralSelectionError(
                    f"Construct {construct.name!r} has a free categorical loading on "
                    f"indicator {indicator.observation.name!r}"
                )
            all_categorical &= categorical
            fixed_scale |= not categorical and value is not None and value != 0.0
        if not fixed_scale and not all_categorical:
            reference = dynamical_model_spec.indicator(reference_indicators(selection)[identity])
            raise StructuralSelectionError(
                f"Construct {construct.name!r} has no scale anchor "
                f"(reference indicator {reference.observation.name!r})"
            )
