"""Compiled scientific site metadata and prior-predictive parameter assembly.

The registry derives authored sample-site shapes and bindings from ModelSpec
without tracing. NumPyro distributions preserve native parameter pytrees and
stable per-site random streams. Particle inference and MAP initialization use
NumPyro replay through Dynestyx for reparameterized sites, transformations, and
the prior density; this module does not define a second inference target.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as random

from nof1_causal_lab.artifacts.parameter import SiteKind, SupportClass
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.execution.parameters import assemble_model_matrices
from nof1_causal_lab.models.ssm.priors import resolve_site_priors
from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor
from nof1_causal_lab.models.ssm.structure.sites import (
    make_site as _site,
)

if TYPE_CHECKING:
    import numpyro.distributions as dist

    from nof1_causal_lab.artifacts.model_spec import ModelSpec
    from nof1_causal_lab.models.ssm.execution.contracts import LikelihoodExtraParams


@dataclass
class PriorRuntimeBundle:
    """Reusable runtime components derived from compiled prior semantics."""

    registry: list[SiteDescriptor]
    priors: dict[str, dist.Distribution]


# ---------------------------------------------------------------------------
# Registry builder
# ---------------------------------------------------------------------------


def build_site_registry(spec: ModelSpec) -> list[SiteDescriptor]:
    """Collect block, dynamics, and likelihood sites in stable name order."""
    return sorted(
        [*numeric.iter_sample_sites(spec), *likelihood_sites(spec)], key=lambda site: site.name
    )


def likelihood_sites(spec: ModelSpec) -> list[SiteDescriptor]:
    """Declare observation/process hyperparameters in their NumPyro sampling order."""
    from nof1_causal_lab.artifacts.likelihood import DistributionFamily

    sites: list[SiteDescriptor] = []
    n_m = numeric.n_observations(spec)

    # -- Likelihood extra-parameter sites -----------------------------------

    manifest_dist_set = set(numeric.observation_families(spec))

    if DistributionFamily.STUDENT_T in manifest_dist_set:
        sites.append(
            _site(
                "obs_df",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.OBS_DF,
                priors_field="obs_df",
            )
        )
    if DistributionFamily.GAMMA in manifest_dist_set:
        sites.append(
            _site(
                "obs_shape",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.OBS_SHAPE,
                priors_field="obs_shape",
            )
        )
    if DistributionFamily.NEGATIVE_BINOMIAL in manifest_dist_set:
        sites.append(
            _site(
                "obs_r",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.OBS_R,
                priors_field="obs_r",
            )
        )
    if DistributionFamily.BETA in manifest_dist_set:
        sites.append(
            _site(
                "obs_concentration",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.OBS_CONCENTRATION,
                priors_field="obs_concentration",
            )
        )

    if numeric.observation_level_counts(spec) is not None:
        level_counts_list = list(numeric.observation_level_counts(spec))
        max_levels = max(level_counts_list) if level_counts_list else 0
        max_cutpoints = max(max_levels - 1, 0)

        if DistributionFamily.ORDERED_LOGISTIC in manifest_dist_set and max_cutpoints > 0:
            sites.append(
                _site(
                    "obs_ordered_base",
                    (n_m,),
                    SupportClass.REAL,
                    "likelihood",
                    SiteKind.OBS_ORDERED_BASE,
                    priors_field="obs_ordered_base",
                )
            )
            if max_cutpoints > 1:
                sites.append(
                    _site(
                        "obs_ordered_gaps",
                        (n_m, max_cutpoints - 1),
                        SupportClass.POSITIVE,
                        "likelihood",
                        SiteKind.OBS_ORDERED_GAPS,
                        priors_field="obs_ordered_gaps",
                    )
                )

        if DistributionFamily.CATEGORICAL in manifest_dist_set and max_cutpoints > 0:
            cat_shape = (n_m, max_cutpoints)
            sites.append(
                _site(
                    "obs_cat_intercepts",
                    cat_shape,
                    SupportClass.REAL,
                    "likelihood",
                    SiteKind.OBS_CAT_INTERCEPTS,
                    priors_field="obs_cat_intercepts",
                )
            )
            sites.append(
                _site(
                    "obs_cat_slopes",
                    cat_shape,
                    SupportClass.REAL,
                    "likelihood",
                    SiteKind.OBS_CAT_SLOPES,
                    priors_field="obs_cat_slopes",
                )
            )

    from nof1_causal_lab.models.ssm.spec_metadata import has_student_t_diffusion

    if has_student_t_diffusion(spec):
        sites.append(
            _site(
                "proc_df",
                (),
                SupportClass.POSITIVE,
                "likelihood",
                SiteKind.PROC_DF,
                priors_field="proc_df",
            )
        )

    return sites


# ---------------------------------------------------------------------------
# Selection and assembly of authored parameter values
# ---------------------------------------------------------------------------


def select_site_samples(
    samples: dict[str, jnp.ndarray],
    registry: list[SiteDescriptor],
    *,
    assembly_group: str | None = None,
) -> dict[str, jnp.ndarray]:
    """Select sampled site values using registry metadata instead of name lists."""
    selected: dict[str, jnp.ndarray] = {}
    for site in registry:
        if assembly_group is not None and site.assembly_group != assembly_group:
            continue
        if site.name in samples:
            selected[site.name] = samples[site.name]
    return selected


def _resolve_num_draws(
    samples: dict[str, jnp.ndarray],
    n_draws: int | None,
) -> int:
    if n_draws is not None:
        return n_draws
    if samples:
        return int(next(iter(samples.values())).shape[0])
    raise ValueError("n_draws is required when assembling deterministic values without samples")


def assemble_deterministics_from_registry(
    samples: dict[str, jnp.ndarray],
    spec: ModelSpec,
    *,
    n_draws: int | None = None,
) -> dict[str, jnp.ndarray]:
    """Batch the same scientific matrix assembly used by NumPyro inference."""
    n_draws = _resolve_num_draws(samples, n_draws)

    def assemble_draw(index):
        return assemble_model_matrices(
            spec, {name: value[index] for name, value in samples.items()}
        )[0]

    return jax.vmap(assemble_draw)(jnp.arange(n_draws))


def assemble_extra_params_from_registry(
    spec: ModelSpec,
    samples: dict[str, jnp.ndarray],
    registry: list[SiteDescriptor],
) -> LikelihoodExtraParams:
    """Assemble likelihood extra parameters using registry metadata as authority."""
    from nof1_causal_lab.models.ssm.likelihood_extra_params import assemble_sampled_extra_params

    return assemble_sampled_extra_params(
        spec,
        select_site_samples(samples, registry, assembly_group="likelihood"),
    )


# ---------------------------------------------------------------------------
# Prior sampling
# ---------------------------------------------------------------------------


def _stable_site_key(rng_key: jnp.ndarray, site_name: str) -> jnp.ndarray:
    """Derive a site stream that is unchanged by registry insertion or reordering."""
    digest = hashlib.sha256(site_name.encode()).digest()
    first = int.from_bytes(digest[:4], "little")
    second = int.from_bytes(digest[4:8], "little")
    return random.fold_in(random.fold_in(rng_key, first), second)


def sample_prior_parameters(
    rng_key: jnp.ndarray,
    registry: list[SiteDescriptor],
    priors: dict[str, dist.Distribution],
    n_samples: int = 200,
) -> dict[str, jnp.ndarray]:
    """Draw authored parameters directly from their NumPyro distributions.

    Each (draw index, site name) keeps its independent stream. Prior prediction
    consumes constrained values, so no inference coordinate transform is needed.
    """
    draws = {}
    for site in registry:
        distribution = priors[site.name]
        draws[site.name] = jnp.stack(
            [
                distribution.sample(_stable_site_key(random.fold_in(rng_key, index), site.name))
                for index in range(n_samples)
            ]
        )
    return draws


def build_prior_runtime_bundle(
    spec: ModelSpec,
    priors: dict[str, dist.Distribution] | None = None,
) -> PriorRuntimeBundle:
    """Resolve the scientific site declarations to native NumPyro laws."""
    registry = build_site_registry(spec)
    return PriorRuntimeBundle(registry=registry, priors=resolve_site_priors(registry, priors))
