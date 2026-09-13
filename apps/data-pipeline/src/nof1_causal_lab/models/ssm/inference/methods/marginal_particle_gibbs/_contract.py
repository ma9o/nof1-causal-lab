"""Latent-smoother contract types and per-step context for MPGibbs."""

# References: see kernel.py and smoothers/dsmc.py for algorithm provenance.

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, NamedTuple

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.problem import ParticleContext

if TYPE_CHECKING:
    from jax.typing import DTypeLike
    from jaxtyping import Array, Float


type ParticleContextRuntimeFn = Callable[[jnp.ndarray, jnp.ndarray], ParticleContext]
type ObservationIncrementLogProbRuntimeFn = Callable[
    [ParticleContext, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
]
type TrajectoryLogProbRuntimeFn = Callable[[ParticleContext, jnp.ndarray, jnp.ndarray], jnp.ndarray]
type TransitionLogProbFn = Callable[
    [ParticleContext, jnp.ndarray, jnp.ndarray, jnp.ndarray], jnp.ndarray
]

_LATENT_SMOOTHER_DSMC = "dsmc"
_DSMC_LEAF_PROPOSAL_AMALA_EXACT = "amala_exact"
# Paid mixture leaf: the amala_exact z-anchored component plus a FIXED pilot
# component (IEKS warmup moments) and a wide tail, all inside one paid proposal
# density. Any component that is useless on a given fit costs only its share of
# proposal mass — never correctness — so the mixture strictly generalizes
# amala_exact (its z-component alone).
_DSMC_LEAF_PROPOSAL_PAID_MIX = "paid_mix"
_DSMC_LEAF_PROPOSALS = (
    _DSMC_LEAF_PROPOSAL_AMALA_EXACT,
    _DSMC_LEAF_PROPOSAL_PAID_MIX,
)
_LATENT_SMOOTHERS = (_LATENT_SMOOTHER_DSMC,)
type DSMCLeafProposal = Literal["amala_exact", "paid_mix"]


class MPGibbsLatentSmoother(NamedTuple):
    """Static metadata for an MPGibbs latent smoother implementation."""

    name: str
    algorithm: str
    family: str
    selection: str
    parallel: bool
    backward_sampling: bool


class MPGibbsLatentSmootherResult(NamedTuple):
    """Production result contract for MPGibbs latent smoothers."""

    latent_path: Float[Array, "T D"]
    final_label_log_probs: Float[Array, " K"]
    origin_path: jnp.ndarray
    diagnostics: dict[str, jnp.ndarray]


def _resolve_latent_smoother(name: str) -> MPGibbsLatentSmoother:
    if name == _LATENT_SMOOTHER_DSMC:
        return MPGibbsLatentSmoother(
            name=name,
            algorithm="conditional_desequentialized_smc",
            family="posterior_mixture_dsmc",
            selection="tree_stitch_combination",
            parallel=True,
            backward_sampling=False,
        )
    allowed = ", ".join(repr(candidate) for candidate in _LATENT_SMOOTHERS)
    raise ValueError(
        f"marginal_particle_gibbs latent_smoother must be one of {allowed}; got {name!r}."
    )


class MPGibbsStatic(NamedTuple):
    """Build-time configuration and target callables for the smoother context."""

    latent_context_runtime_fn: ParticleContextRuntimeFn
    log_prior_unc_fn: Callable[[jnp.ndarray], jnp.ndarray]
    obs_increment_fn: ObservationIncrementLogProbRuntimeFn
    trajectory_log_prob_fn: TrajectoryLogProbRuntimeFn
    runtime_observations: jnp.ndarray
    runtime_times: jnp.ndarray
    num_particles: int
    num_parameter_particles: int
    latent_delta: float
    amala_kappa: float
    amala_grad_clip: float
    dsmc_leaf_proposal: DSMCLeafProposal
    # Number of latent coordinates proposed per sweep (None = all). Restricting the
    # per-sweep update to a random coordinate block sidesteps the joint-coherence
    # weight degeneracy of full-state proposals at higher latent dimension; the tree
    # seams still pay the full transition density, so this is exactly the
    # coordinate-conditional cSMC on the same target.
    latent_block_coords: int | None
    # paid_mix leaf: mixture weights (wide weight = 1 - z - pilot) and the FIXED
    # per-time pilot moments derived from the IEKS warmup paths.
    paid_mix_z_weight: float
    paid_mix_pilot_weight: float
    pilot_means: jnp.ndarray | None
    pilot_vars: jnp.ndarray | None
    pilot_wide_vars: jnp.ndarray | None
    transition_initial_log_prob_fn: Callable[[ParticleContext, jnp.ndarray], jnp.ndarray]
    transition_log_prob_fn: TransitionLogProbFn
    transition_log_probs_for_pairs_fn: TransitionLogProbFn
    transition_pairwise_log_probs_fn: TransitionLogProbFn
    diagnostic_metrics: frozenset[str]


class SmootherContext(NamedTuple):
    """Per-step latent-smoother inputs.

    Explicit interface replacing the implicit closure capture the smoothers
    previously relied on. Built once per joint step by
    :func:`build_smoother_context`; consumed by the smoother modules.
    """

    contexts: ParticleContext
    initial_label_log_probs: Float[Array, " K"]
    num_steps: int
    num_free_particles: int
    num_parameter_particles: int
    latent_dtype: DTypeLike
    traj_dtype: DTypeLike
    obs_increment_fn: ObservationIncrementLogProbRuntimeFn
    runtime_observations: jnp.ndarray
    amala_delta: Float[Array, " D"]
    amala_kappa: float
    amala_grad_clip: float
    dsmc_leaf_proposal: DSMCLeafProposal
    latent_block_coords: int | None
    paid_mix_z_weight: float
    paid_mix_pilot_weight: float
    pilot_means: jnp.ndarray | None
    pilot_vars: jnp.ndarray | None
    pilot_wide_vars: jnp.ndarray | None
    initial_value_grad_by_param: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]
    transition_current_value_grad_by_param: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]
    transition_next_value_grad_by_param: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]
    selected_transition_log_probs: Callable[..., jnp.ndarray]
    pairwise_transition_log_probs: Callable[..., jnp.ndarray]
    trajectory_label_log_probs: Callable[..., jnp.ndarray]
