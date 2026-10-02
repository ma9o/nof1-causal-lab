"""Leaf factory for concrete marginal likelihood backend construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm import numerics as numeric

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime


def build_laplace_backend(
    spec: CompiledModel,
    n_ieks_iters: int,
    observation_support: ObservationSupportRuntime | None = None,
):
    """Construct a Laplace likelihood backend for a compiled spec."""
    from nof1_causal_lab.models.ssm.inference.targets.laplace import LaplaceLikelihood

    # Gaussian smoothing is an initialization view only. Its existing covariance
    # regularization lets it seed exact observations; the particle initialization
    # then substitutes their exact values and every retained draw uses Delta.
    warmup_families = [
        DistributionFamily.GAUSSIAN if family == DistributionFamily.DELTA else family
        for family in numeric.observation_families(spec)
    ]
    return LaplaceLikelihood(
        n_latent=numeric.n_states(spec),
        n_manifest=numeric.n_observations(spec),
        manifest_dists=warmup_families,
        manifest_links=list(numeric.observation_links(spec)),
        n_ieks_iters=n_ieks_iters,
        observation_support=observation_support,
    )
