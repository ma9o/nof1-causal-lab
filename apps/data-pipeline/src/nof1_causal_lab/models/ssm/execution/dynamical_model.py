"""Compile scientific drift and measurement semantics into Dynestyx models.

Dynestyx owns state evolution, initial distributions, and numerical interpretation.
These adapters supply the application's causal vector field and heterogeneous
measurement semantics. Their parameters are explicit pytree leaves, so a model
can be carried through JAX transformations without closing over traced arrays.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, override

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import MultivariateNormal

from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.dynamics.spec import (
    compile_dynamics,
    pack_component_params_from_samples,
)
from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField, VectorFieldArgs
from nof1_causal_lab.models.ssm.execution.contracts import MeasurementParams
from nof1_causal_lab.models.ssm.execution.observation_model import compile_observation_model

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from dynestyx import StochasticContinuousTimeStateEvolution
    from jax.typing import ArrayLike

    from nof1_causal_lab.artifacts.likelihood import LinkFunction
    from nof1_causal_lab.distributions import DistributionFamily
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledModel
    from nof1_causal_lab.models.ssm.dynamics.spec import DynamicsSpec
    from nof1_causal_lab.models.ssm.execution.contracts import (
        LikelihoodExtraParams,
    )
    from nof1_causal_lab.models.ssm.execution.observation_dispatch import ObservationSampleFn
    from nof1_causal_lab.models.ssm.execution.observation_model import ObservationKernel
    from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor


def continuous_state_evolution(
    vector_field: VectorField,
    vf_params: tuple[Mapping[str, jax.Array], ...],
    diffusion: jax.Array,
    *,
    intervention: Intervention | None = None,
) -> dsx.StochasticContinuousTimeStateEvolution:
    """Declare the nonlinear SDE without selecting a numerical approximation."""
    intervention = Intervention.none() if intervention is None else intervention
    for clamp in intervention.variable_overrides():
        diffusion = diffusion.at[clamp.index].set(0.0)
    return vector_field.evolution(
        VectorFieldArgs(params=vf_params, intervention=intervention),
        diffusion=dsx.FullDiffusion(diffusion),
    )


class HeterogeneousObservation(dsx.ObservationModel):
    """Measurement mapping with the application's family/link and missingness rules."""

    measurement: MeasurementParams
    families: tuple[DistributionFamily, ...] = eqx.field(static=True)
    links: tuple[LinkFunction, ...] = eqx.field(static=True)
    extra_params: LikelihoodExtraParams | None = None

    def __post_init__(self) -> None:
        if self.extra_params is not None:
            object.__setattr__(self, "extra_params", MappingProxyType(dict(self.extra_params)))

    @override
    def __call__(
        self, x: jax.Array, u: jax.Array | None, t: float | int | jax.Array
    ) -> _ObservationDistribution:
        del u, t
        return self.at_predictor(self.linear_predictor(x))

    def linear_predictor(self, state: jax.Array) -> jax.Array:
        return self.measurement.lambda_mat @ state + self.measurement.manifest_means

    def at_predictor(self, predictor: jax.Array) -> _ObservationDistribution:
        """Use the same observation law with cached predictors and interval projections."""
        return _ObservationDistribution(predictor, self)


class _ObservationDistribution(dist.Distribution):
    """NumPyro distribution for one heterogeneous measurement row."""

    _predictor: jax.Array
    _measurement: MeasurementParams
    _kernel: ObservationKernel
    _sampler: ObservationSampleFn

    @property
    @override
    def support(self) -> dist.constraints.Constraint:
        return dist.constraints.real_vector

    def __init__(self, predictor: jax.Array, observation: HeterogeneousObservation) -> None:
        self._predictor = predictor
        self._measurement = observation.measurement
        compiled = compile_observation_model(
            observation.families,
            manifest_cov=self._measurement.manifest_cov,
            extra_params=observation.extra_params,
            manifest_links=observation.links,
        )
        self._kernel = compiled.kernel
        self._sampler = compiled.point_sampler.sample_point
        super().__init__(
            batch_shape=(),
            event_shape=(int(self._measurement.lambda_mat.shape[0]),),
            validate_args=False,
        )

    @property
    def mean(self) -> jax.Array:
        return self._kernel.response_fn(self._predictor)

    def sample(self, key: jax.Array | None, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        assert key is not None  # This stochastic distribution requires a NumPyro PRNG key.
        if not sample_shape:
            return self._sampler(key, self._predictor)
        from math import prod

        keys = jax.random.split(key, prod(sample_shape))
        flat = jax.vmap(lambda k: self._sampler(k, self._predictor))(keys)
        return flat.reshape((*sample_shape, *self.event_shape))

    def log_prob(self, value: ArrayLike, intermediates: list[object] | None = None) -> jax.Array:
        del intermediates
        observation = jnp.asarray(value)
        mask = ~jnp.isnan(observation)
        filled = jnp.nan_to_num(observation, nan=0.0)
        measurement = self._measurement
        return jnp.asarray(
            self._kernel.log_prob_fn(
                filled,
                self._predictor,
                measurement.manifest_cov,
                mask.astype(self._predictor.dtype),
            ),
            dtype=self._predictor.dtype,
        )


def initial_state_distribution(
    spec: CompiledModel,
    mean: jax.Array,
    covariance: jax.Array,
    *,
    input_values: jax.Array | None = None,
) -> MultivariateNormal:
    """Embed the modeled initial law in a state vector with lawless inputs."""
    endogenous = jnp.asarray(np.flatnonzero(~numeric.input_mask(spec)))
    factor = (
        jnp.zeros_like(covariance)
        .at[jnp.ix_(endogenous, endogenous)]
        .set(jnp.linalg.cholesky(covariance[jnp.ix_(endogenous, endogenous)]))
    )
    if input_values is not None:
        mean = jnp.where(numeric.input_mask(spec), input_values[0], mean)
    return MultivariateNormal(mean, scale_tril=factor)


def assemble_likelihood_inputs(
    samples: dict[str, jnp.ndarray],
    spec: CompiledModel,
    *,
    registry: Sequence[SiteDescriptor],
    dynamics: DynamicsSpec | None = None,
    intervention: Intervention | None = None,
    input_values: jax.Array | None = None,
) -> tuple[
    StochasticContinuousTimeStateEvolution,
    MeasurementParams,
    MultivariateNormal,
    LikelihoodExtraParams | None,
]:
    """Consume the canonical deterministic values from NumPyro's prior replay."""
    from nof1_causal_lab.models.ssm.parameterization import assemble_extra_params_from_registry

    dynamics_spec = spec.dynamics.spec if dynamics is None else dynamics
    compiled = spec.dynamics if dynamics is None else compile_dynamics(dynamics_spec)
    diffusion_chol = samples["diffusion"]
    evolution = continuous_state_evolution(
        vector_field=compiled.vector_field,
        vf_params=pack_component_params_from_samples(dynamics_spec, samples),
        diffusion=diffusion_chol,
        intervention=intervention,
    )
    measurement = MeasurementParams(
        lambda_mat=samples["lambda"],
        manifest_means=samples["manifest_means"],
        manifest_cov=samples["manifest_cov"],
    )
    initial = initial_state_distribution(
        spec, samples["t0_means"], samples["t0_cov"], input_values=input_values
    )
    extra = assemble_extra_params_from_registry(spec, samples, registry)
    return evolution, measurement, initial, extra or None


def build_dynamical_model(
    model_spec: CompiledModel,
    values: dict[str, jnp.ndarray],
    *,
    t0: jax.Array,
    dynamics: DynamicsSpec | None = None,
) -> dsx.DynamicalModel:
    """Construct Dynestyx's model from the compiled model and one parameter draw.

    Values include the matrices derived by assemble_model_matrices or by
    NumPyro's prior replay. An execution-only dynamics override supports paired
    edge-off admission contrasts while retaining the same measurement and initial laws.
    """
    evolution, measurement, initial, extra_params = assemble_likelihood_inputs(
        values, model_spec, registry=model_spec.site_registry, dynamics=dynamics
    )
    return dsx.DynamicalModel(
        initial_condition=initial,
        state_evolution=evolution,
        observation_model=HeterogeneousObservation(
            measurement,
            tuple(numeric.observation_families(model_spec)),
            tuple(numeric.observation_links(model_spec)),
            extra_params,
        ),
        t0=t0,
    )
