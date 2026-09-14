"""Leaf factory for concrete marginal likelihood backend construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.models.ssm import numerics as numeric

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.model import SSMModel
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime


def build_laplace_backend(
    spec: ModelSpec,
    n_ieks_iters: int,
    observation_support: ObservationSupportRuntime | None = None,
):
    """Construct a Laplace likelihood backend for a compiled spec."""
    from nof1_causal_lab.models.ssm.inference.targets.laplace import LaplaceLikelihood
    from nof1_causal_lab.models.ssm.spec_metadata import (
        get_per_channel_links,
        get_per_channel_manifest,
    )

    return LaplaceLikelihood(
        n_latent=numeric.n_states(spec),
        n_manifest=numeric.n_observations(spec),
        manifest_dists=get_per_channel_manifest(spec),
        manifest_links=get_per_channel_links(spec),
        n_ieks_iters=n_ieks_iters,
        observation_support=observation_support,
    )


def get_laplace_backend(model: SSMModel, n_ieks_iters: int):
    """Construct or reuse the warmup-only Laplace backend for one model."""
    return model.get_cached_artifact(
        (
            "backend",
            "laplace",
            n_ieks_iters,
            id(model.observation_support),
        ),
        lambda: build_laplace_backend(
            model.spec,
            n_ieks_iters,
            observation_support=model.observation_support,
        ),
    )
