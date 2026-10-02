"""Read predictive observation statistics from the execution layer's exact laws."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.likelihood import DistributionFamily
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.emissions import (
    categorical_probabilities,
    get_categorical_extra_params,
    get_ordered_logistic_extra_params,
    ordered_logistic_probabilities,
)
from nof1_causal_lab.models.ssm.execution.observation_distributions import (
    gaussian_distribution,
    mean_observation_variance,
    point_observation_scales,
)
from nof1_causal_lab.models.ssm.execution.observation_extra_params import (
    slice_observation_extra_params,
)
from nof1_causal_lab.models.ssm.execution.observation_operator import compile_observation_operator

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

    from .types import PredictiveDraws


def observation_signal_and_variance(
    spec: CompiledModel,
    prediction: PredictiveDraws,
    manifest_index: int,
    time_indices: np.ndarray,
    *,
    observation_support: ObservationSupportRuntime | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Conditional means/variances or categorical probability vectors for one channel.

    Scalar means already include the observation operator's interval projection.
    Discrete channels use the same assembled cutpoints, anchors, padding and
    probability functions as the likelihood and sampler.
    """
    paths = prediction.trajectory
    n_channels = paths.observations.shape[2]
    families = numeric.observation_families(spec)
    family = families[manifest_index]
    all_gaussian = all(item == DistributionFamily.GAUSSIAN for item in families)
    operator = compile_observation_operator(observation_support)
    interval_summary = operator is not None and manifest_index in operator.interval_summary_indices

    def per_draw(
        means: jax.Array,
        predictors: jax.Array,
        covariance: jax.Array,
        extra_params: Mapping[str, jax.Array],
    ):
        channel_params = slice_observation_extra_params(
            extra_params, [manifest_index], source_channel_count=n_channels
        )
        if family == DistributionFamily.ORDERED_LOGISTIC:
            levels, cutpoints = get_ordered_logistic_extra_params(channel_params)
            probabilities = jax.vmap(
                lambda eta: ordered_logistic_probabilities(eta, cutpoints, levels)
            )(predictors)
            return probabilities[:, 0, :], None
        if family == DistributionFamily.CATEGORICAL:
            levels, intercepts, slopes = get_categorical_extra_params(channel_params)
            probabilities = jax.vmap(
                lambda eta: categorical_probabilities(eta, intercepts, slopes, levels)
            )(predictors)
            return probabilities[:, 0, :], None
        if family == DistributionFamily.GAUSSIAN and (all_gaussian or interval_summary):
            marginal_covariance = covariance[
                manifest_index : manifest_index + 1, manifest_index : manifest_index + 1
            ]
            return means, gaussian_distribution(means[:, None], marginal_covariance).variance[:, 0]
        scale = (
            jnp.sqrt(covariance[manifest_index, manifest_index])
            if interval_summary
            else point_observation_scales(covariance)[manifest_index]
        )
        return means, mean_observation_variance(family, means, scale, channel_params)

    signal, variance = jax.vmap(per_draw)(
        paths.expected_observations[:, time_indices, manifest_index],
        paths.linear_predictors[:, time_indices, manifest_index : manifest_index + 1],
        prediction.parameters["manifest_cov"],
        prediction.likelihood_parameters,
    )
    return np.asarray(signal), None if variance is None else np.asarray(variance)
