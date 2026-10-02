"""Check execution requirements directly on the current scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.model_structure import reference_indicators, selected_state_ids

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


def validate_parameter_anchors(model: ModelSpec) -> None:
    """Reject unanchored retained constructs once their anchor inputs are authored."""
    from nof1_causal_lab.artifacts.construct import CausalEdgeSpec
    from nof1_causal_lab.artifacts.expressions import StateExpression, restoring_coefficients
    from nof1_causal_lab.distributions import DistributionFamily

    if model.measurement_clock is None or not model.indicators:
        return

    for identity in selected_state_ids(model):
        construct = model.get_construct(identity)
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
            for owner, mechanism in model.iter_mechanisms()
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
            raise ValueError(f"Construct {construct.name!r} has no location anchor")

        fixed_scale = False
        all_categorical = True
        for indicator, likelihood in channels:
            categorical = likelihood.law.family == DistributionFamily.CATEGORICAL
            loading = likelihood.parsed.loadings[identity].value
            value = None if isinstance(loading, str) else loading
            if categorical and value is None:
                raise ValueError(
                    f"Construct {construct.name!r} has a free categorical loading on "
                    f"indicator {indicator.name!r}"
                )
            all_categorical &= categorical
            fixed_scale |= not categorical and value is not None and value != 0.0
        if not fixed_scale and not all_categorical:
            reference = model.indicator(reference_indicators(model)[identity])
            raise ValueError(
                f"Construct {construct.name!r} has no scale anchor "
                f"(reference indicator {reference.name!r})"
            )
