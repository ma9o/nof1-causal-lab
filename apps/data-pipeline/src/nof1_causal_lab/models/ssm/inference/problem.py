"""Interpret nof1's scientific model through Dynestyx's public distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dynestyx as dsx
import equinox as eqx
from dynestyx.inference.configs.discretizer import EulerMaruyamaConfig

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
from nof1_causal_lab.models.ssm.preflight import validate_observation_support_for_fit
from nof1_causal_lab.models.ssm.spec_metadata import has_student_t_diffusion
from nof1_causal_lab.models.ssm.transition_kinds import LATENT_TRANSITION_EULER_MARUYAMA

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.utils import SiteInfo


@dataclass(frozen=True)
class ParticleProblem:
    """Exact model target together with parameter and reporting metadata."""

    runtime: ParticleTarget
    site_info: SiteInfo
    public_sites: set[str]
    latent_transition_kind: str
    exact_constraints: ExactStateConstraints | None = None


def build_particle_problem(model, observations, times, *, scheme, trace_key, reparam):
    """Prepare the parameter transform and the exact discrete model for sampling."""
    if scheme != LATENT_TRANSITION_EULER_MARUYAMA:
        raise ValueError(f"Particle inference requires 'euler_maruyama'; got {scheme!r}.")
    exact_constraints = compile_exact_state_constraints(model.spec, observations)
    if has_student_t_diffusion(model.spec):
        raise ValueError(
            "Particle inference currently requires Gaussian latent diffusion for every state."
        )
    validate_observation_support_for_fit(model)
    parameters, site_info, public_sites = prepare_model_parameters(
        model, observations, times, trace_key, reparam
    )

    def continuous_model(position, runtime_times):
        return build_dynamical_model(
            model.spec, parameters.constrain(position), t0=runtime_times[0]
        )

    # Partition once to retain static metadata outside the sampler state. Every
    # parameter-dependent value in the declared model is an explicit array leaf.
    _, static_model = eqx.partition(
        continuous_model(parameters.initial_position, times), eqx.is_array
    )

    def context_fn(position, runtime_times):
        return eqx.filter(continuous_model(position, runtime_times), eqx.is_array), runtime_times

    def discrete_model(context):
        return dsx.discretize_dynamics(
            eqx.combine(context[0], static_model),
            EulerMaruyamaConfig(covariance_jitter=CHOL_JITTER),
        )

    runtime = ParticleTarget(
        parameters,
        context_fn,
        discrete_model,
        observations,
        times,
    )
    return ParticleProblem(
        runtime, site_info, public_sites, LATENT_TRANSITION_EULER_MARUYAMA, exact_constraints
    )
