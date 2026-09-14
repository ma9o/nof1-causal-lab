"""Location and scale identification certificates for the source ModelSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.execution import AnchorCertificate
from nof1_causal_lab.artifacts.expressions import StateExpression, restoring_coefficients
from nof1_causal_lab.artifacts.likelihood import DistributionFamily
from nof1_causal_lab.compilation_errors import AggregatedCompileError
from nof1_causal_lab.models.model_semantics import indicator_has_additive_location_support
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.utils.model_structure import (
    get_manifest_indicators,
    get_state_ids,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec


class StructuralClosureError(AggregatedCompileError):
    """Aggregate failures in retained-state identification."""

    header = "Structural closure failed"


def compile_anchor_certificates(
    spec: ModelSpec,
) -> list[AnchorCertificate]:
    """Prove location and scale identification for every retained state."""
    latent_names = list(numeric.state_names(spec) or [])
    manifest_names = list(numeric.observation_names(spec) or [])
    state_ids = get_state_ids(spec)
    indicators = get_manifest_indicators(spec)
    indicator_by_name = {str(item["name"]): item for item in indicators}
    channels_by_construct: dict[str, list[int]] = {}
    for manifest_index, manifest_name in enumerate(manifest_names):
        indicator = indicator_by_name.get(manifest_name)
        if indicator is not None:
            channels_by_construct.setdefault(str(indicator["construct_name"]), []).append(
                manifest_index
            )

    standardized = list(numeric.observation_standardized(spec) or [False] * len(manifest_names))
    exact_location_channels = {
        index
        for index, indicator in enumerate(numeric.observed_indicators(spec))
        if indicator.likelihood is not None
        and indicator.likelihood.law.distribution == "Delta"
        and isinstance(indicator.likelihood.law.arguments["v"], StateExpression)
        and indicator_has_additive_location_support(
            indicator.support_kind, indicator.summary_operator
        )
    }
    categorical_anchors = list(numeric.categorical_anchors(spec) or [False] * len(manifest_names))
    loading_template = np.asarray(numeric.loading_block(spec).template, dtype=float)
    loading_support = np.asarray(numeric.loading_block(spec).free_support, dtype=bool)
    time_invariant = np.asarray(
        numeric.diffusion_block(spec).time_invariant_mask
        if numeric.diffusion_block(spec).time_invariant_mask is not None
        else np.zeros(len(latent_names), dtype=bool),
        dtype=bool,
    )
    t0_mean_support = np.asarray(numeric.initial_mean_block(spec).free_support, dtype=bool)
    fixed_dynamics_centers = {
        component.target
        for component in numeric.dynamics_expressions(spec)
        for operand in restoring_coefficients(
            component.expression, state_ids[component.target], kind=component.kind
        )
        if operand.role == "center" and isinstance(operand.value, (int, float))
    }

    errors: list[str] = []
    certificates: list[AnchorCertificate] = []
    for latent_index, (construct_id, construct_name) in enumerate(
        zip(state_ids, latent_names, strict=True)
    ):
        channels = channels_by_construct.get(construct_name, [])
        if not channels:
            errors.append(
                f"Construct {construct_name!r} retains no manifest channel; its latent "
                "location and scale are unidentified."
            )
            continue

        standardized_channels = [index for index in channels if standardized[index]]
        exact_channels = [index for index in channels if index in exact_location_channels]
        if standardized_channels:
            location_index = standardized_channels[0]
            location_anchor = "standardized_manifest"
            location_source_id = str(indicator_by_name[manifest_names[location_index]]["source_id"])
        elif exact_channels:
            location_index = exact_channels[0]
            location_anchor = "exact_state_observation"
            location_source_id = str(indicator_by_name[manifest_names[location_index]]["source_id"])
        elif time_invariant[latent_index] and not t0_mean_support[latent_index]:
            location_anchor = "fixed_initial_mean"
            location_source_id = None
        elif not time_invariant[latent_index] and latent_index in fixed_dynamics_centers:
            location_anchor = "fixed_dynamics_center"
            location_source_id = None
        else:
            location_parameter = "t0 mean" if time_invariant[latent_index] else "equilibrium center"
            errors.append(
                f"Construct {construct_name!r} has no location anchor: its free "
                f"{location_parameter} rides an exact additive ridge with channel-side "
                "location parameters."
            )
            continue

        fixed_loading_channels = [
            index
            for index in channels
            if numeric.observation_families(spec)[index] != DistributionFamily.CATEGORICAL
            and loading_template[index, latent_index] != 0.0
            and not loading_support[index, latent_index]
        ]
        categorical_anchor_channels = [index for index in channels if categorical_anchors[index]]
        if fixed_loading_channels:
            scale_index = fixed_loading_channels[0]
            scale_anchor = "fixed_manifest_loading"
        elif categorical_anchor_channels:
            scale_index = categorical_anchor_channels[0]
            scale_anchor = "categorical_slope_pin"
        else:
            errors.append(
                f"Construct {construct_name!r} has no scale anchor: no non-categorical "
                "channel carries a fixed loading and no categorical anchor slope is pinned."
            )
            continue

        free_categorical_channels = [
            manifest_names[index]
            for index in channels
            if numeric.observation_families(spec)[index] == DistributionFamily.CATEGORICAL
            and loading_support[index, latent_index]
        ]
        if free_categorical_channels:
            errors.append(
                "Categorical channels have free loadings that are exactly redundant "
                f"with their class slopes: {sorted(free_categorical_channels)}."
            )
            continue

        certificates.append(
            AnchorCertificate(
                construct_id=construct_id,
                construct_name=construct_name,
                location_anchor=location_anchor,
                location_source_id=location_source_id,
                scale_anchor=scale_anchor,
                scale_source_id=str(indicator_by_name[manifest_names[scale_index]]["source_id"]),
            )
        )

    if errors:
        raise StructuralClosureError(errors)
    return certificates
