"""Owned sampler choices and explicit numerical initialization inputs."""

from __future__ import annotations

import math
from typing import Literal

import jax
from pydantic import Field, InstanceOf

from nof1_causal_lab.artifacts.base import Value

type DSMCLeafProposal = Literal["amala_exact", "paid_mix"]


class MarginalParticleGibbsSpec(Value):
    """Marginalized Particle Gibbs inference settings."""

    n_parameter_particles: int = 2
    latent_smoother: Literal["dsmc"] = "dsmc"
    latent_delta: float = 0.2
    parameter_proposal: Literal["random_walk", "pseudo_langevin"] = "pseudo_langevin"
    amala_delta_init: float = 1e-2
    amala_delta_min: float = 1e-5
    amala_delta_max: float = 1e1
    amala_target_accept: float = 0.75
    amala_adaptation_window: int = 100
    amala_adaptation_tolerance: float = 0.05
    amala_adaptation_rho: float = 0.5
    amala_adaptation_rho_min: float = 1e-3
    amala_adaptation_gamma: float = -0.5
    amala_kappa: float = 0.75
    amala_grad_clip: float = math.inf
    dsmc_leaf_proposal: DSMCLeafProposal = "amala_exact"
    # Coordinate-block proposals: number of latent coordinates proposed per sweep
    # (None = all). Blocks of 2-4 sidestep the joint-coherence weight degeneracy of
    # full-state proposals at higher latent dimension.
    latent_block_coords: int | None = None
    # paid_mix leaf mixture: z-anchored (amala_exact core) + fixed IEKS-pilot
    # component + wide tail (weight = 1 - z - pilot). Pilot variances are
    # pilot_var_scale x the IEKS paths' per-coordinate spread; the wide tail is
    # wide_mult x the same spread.
    paid_mix_z_weight: float = 0.85
    paid_mix_pilot_weight: float = 0.10
    paid_mix_pilot_var_scale: float = 0.25
    paid_mix_wide_mult: float = 4.0
    diagnostic_metrics_all: bool = False
    diagnostic_metrics: tuple[str, ...] = ()
    param_step_size: float = 0.02
    param_step_size_min: float = 1e-6
    param_step_size_max: float = 1e3
    param_target_accept: float = 0.35
    adaptation_rate: float = 0.05
    adaptation_scheme: Literal["simple", "dual_averaging"] = "simple"
    init_method: Literal["random", "pathfinder"] = "pathfinder"
    latent_init_method: Literal["predictive"] = "predictive"
    pathfinder_num_elbo_samples: int = 20
    pathfinder_maxiter: int = 20
    n_pathfinder_starts: int = 8
    pathfinder_parallel_workers: int | None = None
    pathfinder_init_scale: float | None = 0.1
    auto_preconditioner_method: Literal["map", "none", "pathfinder"] = "pathfinder"
    auto_preconditioner_maxiter: int = 200
    init_scale: float = 0.05
    compute_latent_posterior_summary: bool = True

    n_ieks_iters: int = 6


class SamplerSpec(Value):
    """Fully resolved controls for the exact particle sampler."""

    num_warmup: int = Field(default=4000, ge=0)
    num_samples: int = Field(default=1000, ge=1)
    num_chains: int = Field(default=4, ge=1)
    seed: int = Field(default=0, ge=0)
    n_particles: int = Field(default=64, ge=2)
    retain_latent_paths: bool = True
    marginal_particle_gibbs: MarginalParticleGibbsSpec = Field(
        default_factory=MarginalParticleGibbsSpec
    )


class SamplerInitialization(Value):
    """Caller-owned immutable JAX buffers used only to initialize the sampler."""

    parameter_preconditioner_chol: InstanceOf[jax.Array] | None = None
    initial_positions: InstanceOf[jax.Array] | None = None
    latent_trajectories: InstanceOf[jax.Array] | None = None
