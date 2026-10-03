"""Check prior reach against model-bound observations before numerical fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import numpyro.distributions as dist

from nof1_causal_lab.artifacts.likelihood import NormalLawSpec, StudentTLawSpec
from nof1_causal_lab.models.ssm import numerics as numeric

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.runtime import BoundPanel

LOCATION_REACH_SIGMAS = 6.0


@dataclass(frozen=True)
class ObservationPreflightFailure:
    """Observed data is inconsistent with the spec/prior configuration."""

    message: str

    @classmethod
    def rejected(cls, message: str) -> Self:
        """Own the rejection value shared by observation preparation and fitting."""
        return cls(message)


def validate_observation_support_for_fit(panel: BoundPanel) -> ObservationPreflightFailure | None:
    """Reject observation semantics the particle target cannot represent."""
    intervals = [
        indicator.name
        for indicator in panel.model.observations
        if indicator.support.support_kind == "interval"
        and not panel.model.states[indicator.state_index].is_input
    ]
    if intervals:
        names = ", ".join(intervals)
        return ObservationPreflightFailure.rejected(
            "Particle inference supports only point measurements; "
            f"unsupported interval summaries: {names}."
        )
    return None


def _prior_loc_scale(
    prior: dist.Distribution, n_free: int, free_idx: int
) -> tuple[str, float, float] | None:
    if isinstance(prior, dist.MixtureGeneral):
        prior = prior.component_distributions[free_idx]
    while isinstance(prior, (dist.ExpandedDistribution, dist.MaskedDistribution)):
        prior = prior.base_dist
    family: str = type(prior).__name__
    if isinstance(prior, dist.TwoSidedTruncatedDistribution):
        prior = prior.base_dist
        family = "TruncatedNormal"
    if not isinstance(prior, dist.Normal):
        return None
    mu = np.broadcast_to(np.asarray(prior.loc, dtype=np.float64), (n_free,))
    sigma = np.broadcast_to(np.asarray(prior.scale, dtype=np.float64), (n_free,))
    return family, float(mu[free_idx]), float(sigma[free_idx])


def validate_observations_for_fit(
    priors: PriorRuntimeBundle, panel: BoundPanel
) -> ObservationPreflightFailure | None:
    """Validate (spec, priors, observations) consistency before fitting.

    Return an expected scientific rejection listing every violating channel.
    """
    failure = validate_observation_support_for_fit(panel)
    if failure is not None:
        return failure
    spec = panel.model
    obs = np.asarray(panel.observations, dtype=np.float64)
    standardized = numeric.observation_standardized(spec)
    names = numeric.observation_names(spec)

    means_block = spec.observation_mean_block
    free_support = np.asarray(means_block.free_support, dtype=bool)
    n_free = int(free_support.sum())
    free_prior = priors.priors[means_block.free_site_name] if n_free else None

    problems: list[str] = []
    for j in range(numeric.n_observations(spec)):
        finite = obs[:, j][np.isfinite(obs[:, j])]
        if finite.size == 0:
            continue
        mean_j = float(finite.mean())

        if bool(standardized[j]):
            continue

        if not isinstance(spec.observations[j].law, (NormalLawSpec, StudentTLawSpec)):
            continue
        if not bool(free_support[j]):
            continue
        if free_prior is None:
            continue

        free_idx = int(free_support[:j].sum())
        prior_summary = _prior_loc_scale(free_prior, n_free, free_idx)
        if prior_summary is None:
            continue
        prior_family, mu_j, sigma_j = prior_summary
        if sigma_j <= 0.0:
            continue
        z = abs(mean_j - mu_j) / sigma_j
        if z > LOCATION_REACH_SIGMAS:
            problems.append(
                f"{names[j]}: observed mean {mean_j:.4g} lies {z:.1f} prior sd from its free "
                f"manifest-mean prior {prior_family}(mu={mu_j:.4g}, sigma={sigma_j:.4g}); "
                "the posterior cannot reach the data location — mark the indicator standardized "
                "(and standardize the data) or author the prior on the data scale"
            )

    if problems:
        return ObservationPreflightFailure.rejected(
            "Observation/prior preflight failed:\n- " + "\n- ".join(problems)
        )
    return None
