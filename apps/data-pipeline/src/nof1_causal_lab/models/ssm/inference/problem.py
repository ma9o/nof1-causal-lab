"""Interpret nof1's scientific model through Dynestyx's public distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dynestyx as dsx
import equinox as eqx
import numpy as np
from dynestyx.inference.configs.discretizer import EulerMaruyamaConfig

from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.covariance_utils import CHOL_JITTER
from nof1_causal_lab.models.ssm.execution.dynamical_model import build_dynamical_model
from nof1_causal_lab.models.ssm.inference.conditioning import (
    ExactStateConstraints,
    compile_exact_state_constraints,
)
from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
from nof1_causal_lab.models.ssm.inference.utils import (
    prepare_model_parameters,
)
from nof1_causal_lab.models.ssm.transition_kinds import LATENT_TRANSITION_EULER_MARUYAMA
from nof1_causal_lab.utils.immutability import freeze_fields

if TYPE_CHECKING:
    import jax

    from nof1_causal_lab.models.ssm.autoreparam import Strategy
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleContext
    from nof1_causal_lab.models.ssm.inference.utils import SiteInfo
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.runtime import BoundPanel


@dataclass(frozen=True)
class ParticleProblem:
    """Exact model target together with parameter and reporting metadata."""

    runtime: ParticleTarget
    site_info: SiteInfo
    public_sites: frozenset[str]
    latent_transition_kind: str
    exact_constraints: ExactStateConstraints | None = None

    def __post_init__(self) -> None:
        freeze_fields(self)


def build_particle_problem(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    scheme: str,
    trace_key: jax.Array,
    reparam: Strategy | None,
) -> ParticleProblem:
    """Prepare the parameter transform and the exact discrete model for sampling."""
    observations, times = panel.observations, panel.times
    model = panel.model
    if scheme != LATENT_TRANSITION_EULER_MARUYAMA:
        raise ValueError(f"Particle inference requires 'euler_maruyama'; got {scheme!r}.")
    exact_constraints = compile_exact_state_constraints(
        model, observations, input_values=panel.input_values
    )
    if DistributionFamily.STUDENT_T in numeric.diffusion_families(model):
        raise ValueError(
            "Particle inference currently requires Gaussian latent diffusion for every state."
        )
    parameters, site_info, public_sites = prepare_model_parameters(
        priors, panel, trace_key, reparam
    )

    def continuous_model(position: jax.Array, runtime_times: jax.Array) -> dsx.DynamicalModel:
        return build_dynamical_model(model, parameters.constrain(position), t0=runtime_times[0])

    # Partition once to retain static metadata outside the sampler state. Every
    # parameter-dependent value in the declared model is an explicit array leaf.
    _, static_model = eqx.partition(
        continuous_model(parameters.initial_position, times), eqx.is_array
    )

    def context_fn(position: jax.Array, runtime_times: jax.Array) -> ParticleContext:
        dynamic_model: dsx.DynamicalModel = eqx.filter(
            continuous_model(position, runtime_times), eqx.is_array
        )
        return dynamic_model, runtime_times

    def declared_model(context: ParticleContext) -> dsx.DynamicalModel:
        return dsx.discretize_dynamics(
            eqx.combine(context[0], static_model),
            EulerMaruyamaConfig(covariance_jitter=CHOL_JITTER),
        )

    runtime = ParticleTarget(
        parameters,
        context_fn,
        declared_model,
        observations,
        times,
        tuple(int(index) for index in np.flatnonzero(~numeric.input_mask(model))),
    )
    return ParticleProblem(
        runtime, site_info, public_sites, LATENT_TRANSITION_EULER_MARUYAMA, exact_constraints
    )
