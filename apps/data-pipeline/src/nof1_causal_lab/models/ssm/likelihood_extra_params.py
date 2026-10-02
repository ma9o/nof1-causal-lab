"""Leaf observation/process hyperparameter assembly for SSM likelihoods."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import jax.numpy as jnp

from nof1_causal_lab.artifacts.likelihood import DistributionFamily
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.observation_families import (
    any_family_needs_level_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.execution.contracts import (
        LikelihoodExtraParams,
        LikelihoodParameterValue,
    )


def assemble_sampled_extra_params(
    spec: CompiledModel,
    sampled_values: Mapping[str, jnp.ndarray],
) -> LikelihoodExtraParams:
    """Assemble likelihood hyperparameters and derived observation metadata."""
    extra_params: dict[str, LikelihoodParameterValue] = {}
    manifest_dist_set = set(numeric.observation_families(spec))

    scalar_keys = (
        "obs_df",
        "obs_shape",
        "obs_r",
        "obs_concentration",
        "proc_df",
    )
    for key in scalar_keys:
        if key in sampled_values:
            extra_params[key] = sampled_values[key]

    level_counts_list = numeric.observation_level_counts(spec)
    level_counts = jnp.asarray(level_counts_list, dtype=jnp.int32)
    extra_params["obs_level_counts"] = level_counts

    max_levels = max(level_counts_list) if level_counts_list else 0
    max_cutpoints = max(max_levels - 1, 0)

    if any_family_needs_level_metadata(manifest_dist_set) and max_cutpoints <= 0:
        raise ValueError(
            "ordered_logistic/categorical requires manifest_level_counts with at least 2 levels"
        )

    if DistributionFamily.ORDERED_LOGISTIC in manifest_dist_set:
        ordered_base = sampled_values["obs_ordered_base"]
        if max_cutpoints > 1:
            ordered_gaps = sampled_values["obs_ordered_gaps"]
        else:
            ordered_gaps = jnp.zeros((numeric.n_observations(spec), 0), dtype=ordered_base.dtype)

        # Cutpoints are NOT centered: the threshold base is the channel-side
        # location parameter, identified against the construct's latent-side
        # location anchor (see docs/assumptions.md#parameter-anchors).
        # Centering here would both kill the base (it cancels exactly) and
        # over-anchor constructs whose location is already pinned elsewhere.
        raw_cutpoints = jnp.concatenate(
            [
                ordered_base[:, None],
                ordered_base[:, None] + jnp.cumsum(ordered_gaps, axis=1),
            ],
            axis=1,
        )
        cutpoint_mask = jnp.arange(max_cutpoints)[None, :] < jnp.maximum(
            level_counts[:, None] - 1,
            0,
        )
        extra_params["obs_ordered_cutpoints"] = jnp.where(cutpoint_mask, raw_cutpoints, 0.0)

    if DistributionFamily.CATEGORICAL in manifest_dist_set:
        cat_mask = jnp.arange(max_cutpoints)[None, :] < jnp.maximum(level_counts[:, None] - 1, 0)
        extra_params["obs_cat_intercepts"] = jnp.where(
            cat_mask,
            sampled_values["obs_cat_intercepts"],
            0.0,
        )
        cat_slopes = jnp.where(cat_mask, sampled_values["obs_cat_slopes"], 0.0)
        if any(numeric.categorical_anchors(spec)):
            # Scale/sign anchor for all-categorical constructs: the anchor
            # channel's first non-baseline slope is pinned to +1 (see
            # docs/assumptions.md#parameter-anchors).
            anchor_rows = jnp.asarray(numeric.categorical_anchors(spec), dtype=bool)
            anchor_entries = anchor_rows[:, None] & (jnp.arange(max_cutpoints)[None, :] == 0)
            cat_slopes = jnp.where(anchor_entries, 1.0, cat_slopes)
        extra_params["obs_cat_slopes"] = cat_slopes

    return MappingProxyType(extra_params)
