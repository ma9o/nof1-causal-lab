"""Interpret nof1's scientific model through Dynestyx's particle runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import dynestyx as dsx
import equinox as eqx
import jax
from dynestyx.inference.configs.discretizer import EulerMaruyamaConfig
from dynestyx.inference.particle_runtime import (
    ParticleRuntime,
    ParticleSchedule,
)

from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.covariance_utils import CHOL_JITTER
from nof1_causal_lab.models.ssm.execution.dynamical_model import build_dynamical_model
from nof1_causal_lab.models.ssm.inference.utils import (
    prepare_model_parameters,
)
from nof1_causal_lab.models.ssm.spec_metadata import has_student_t_diffusion
from nof1_causal_lab.models.ssm.transition_kinds import LATENT_TRANSITION_EULER_MARUYAMA

# A Dynestyx model's array leaves plus its observation grid. Static callables
# and model metadata stay in the runtime closure; the sampler carries no second
# representation of drift, diffusion, initial state, or observation parameters.
type ParticleContext = tuple[dsx.DynamicalModel, jax.Array]


if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.utils import SiteInfo


@dataclass(frozen=True)
class ParticleProblem:
    """Library target together with application parameter and reporting metadata."""

    runtime: ParticleRuntime
    site_info: SiteInfo
    public_sites: set[str]
    latent_transition_kind: str


def build_particle_problem(model, observations, times, *, scheme, trace_key, reparam):
    """Prepare the application model; Dynestyx owns posterior composition."""
    if scheme != LATENT_TRANSITION_EULER_MARUYAMA:
        raise ValueError(f"Particle inference requires 'euler_maruyama'; got {scheme!r}.")
    exact_indicators = [
        indicator.name
        for indicator in numeric.observed_indicators(model.spec)
        if indicator.likelihood is not None
        and indicator.likelihood.law.family == DistributionFamily.DELTA
    ]
    if exact_indicators:
        raise ValueError(
            f"Delta observations {exact_indicators} require constraint-preserving particle "
            "proposals, which are not implemented. Continuous proposals cannot condition on "
            "exact equalities."
        )
    if has_student_t_diffusion(model.spec):
        raise ValueError(
            "Particle inference currently requires Gaussian latent diffusion for every state."
        )
    support = model.observation_support
    if support is not None and support.requires_interval_summary_handling:
        raise ValueError("Particle inference supports only point measurements.")
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

    def schedule(context):
        runtime_times = context[1]
        controls = model.transition_inputs
        if controls is not None:
            controls = controls[: runtime_times.shape[0]]
        return ParticleSchedule(runtime_times, None, controls)

    runtime = ParticleRuntime(
        parameters,
        context_fn,
        discrete_model,
        schedule,
        observations,
        times,
        marginalize_missing=False,
    )
    return ParticleProblem(runtime, site_info, public_sites, LATENT_TRANSITION_EULER_MARUYAMA)
